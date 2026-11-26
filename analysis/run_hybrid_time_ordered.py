"""시간순 Hybrid(ARIMA + LSTM 보정) 비교. 기존 Hybrid(SARIMAX(1,0,1)+LSTM) 설계를 같은 테스트 날짜(719건)에 맞춘 축소판이다.
ARIMA(1,0,1)는 학습 구간 일별 로그수익률에만 적합하고, 5일 누적수익률의 ARIMA 예측은 직전 1단계 예측에 phi^(h-1)을 곱해 합산한다.
LSTM은 (실제 5일 누적수익률 - ARIMA 예측)을 예측한다. 입력은 ARIMA 잔차, 수익률, 금리차 변화에 유동성 변화를 더한 것이다.
기존 Hybrid의 하이퍼파라미터 탐색(HPO)과 CNN-LSTM, 로그/정상화 이외의 변형은 포함하지 않았다.
저장: results_time_ordered/liquidity_lstm_predictions_hybrid.csv (dm_test.py _hybrid로 DM 검정)"""
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from torch.utils.data import DataLoader, TensorDataset

from LSTM.lstm_mmf.train_eval_extended import ExRateLSTM, train_model
from run_time_ordered_checks import OUT, lstm_inputs, series

SEEDS = tuple(range(42, 52))
L, H = 30, 5
MODELS = {"baseline": ["e", "r", "dspread"], "plus_mmf": ["e", "r", "dspread", "dmmf"],
          "plus_deposit": ["e", "r", "dspread", "ddeposit"]}


def fit_correction(frame, cols, target, train, test, cutoff, seed, epochs=20, lr=0.001):
    torch.manual_seed(seed)
    x = frame[cols]
    z = StandardScaler().fit(x.iloc[:cutoff].dropna()).transform(x.fillna(0))
    sc = target[train].std()
    win = lambda idx: torch.tensor(np.stack([z[i - L + 1:i + 1] for i in idx]), dtype=torch.float32)
    y = torch.tensor((target[train] / sc).reshape(-1, 1), dtype=torch.float32)
    model = ExRateLSTM(len(cols), 32, 1)
    train_model(model, DataLoader(TensorDataset(win(train), y), batch_size=32, shuffle=True),
                torch.nn.MSELoss(), torch.optim.Adam(model.parameters(), lr=lr), epochs)
    model.eval()
    with torch.no_grad():
        return model(win(test)).numpy().ravel() * sc


def main():
    frame = lstm_inputs(series("exchange_rate/USD_KRW_processed.csv", "USD_KRW"))
    deposit = series("m2/KOR/M2_details_processed.csv", "M2_수시입출식저축성예금")
    deposit.index = deposit.index + pd.offsets.MonthEnd(0) + pd.offsets.MonthEnd(1)
    dep = pd.merge_asof(frame[[]], deposit.rename("dep").to_frame(), left_index=True, right_index=True,
                        allow_exact_matches=False, tolerance=pd.Timedelta("45D"))["dep"]
    frame["r"] = np.log(frame.fx).diff()
    frame["dspread"] = frame.spread.diff()
    frame["dmmf"] = np.log(frame.mmf).diff()
    frame["ddeposit"] = np.log(dep).diff()
    cutoff = int(len(frame) * 0.8)
    fit = SARIMAX(frame.r.iloc[1:cutoff], order=(1, 0, 1)).fit(disp=False)
    phi = fit.params["ar.L1"]
    applied = fit.apply(frame.r.iloc[1:])
    frame["e"] = np.nan
    frame.loc[frame.index[1:], "e"] = (frame.r.iloc[1:] - applied.fittedvalues).to_numpy()
    one_step = pd.Series(np.nan, index=frame.index)
    one_step.iloc[1:] = applied.fittedvalues.to_numpy()
    next_pred = one_step.shift(-1).to_numpy()  # t 시점까지의 정보로 만든 t+1 예측
    arima_cum = next_pred * sum(phi ** k for k in range(H))
    cum = (np.log(frame.fx).shift(-H) - np.log(frame.fx)).to_numpy()
    feats = ["e", "r", "dspread", "dmmf", "ddeposit"]
    idx = np.arange(L, len(frame) - H)
    ok = frame[feats].notna().all(axis=1).rolling(L).sum().eq(L).to_numpy() & np.isfinite(arima_cum)
    idx = idx[ok[idx]]
    train, test = idx[idx + H < cutoff], idx[idx >= cutoff]
    target = cum - arima_cum
    last = frame.fx.to_numpy()
    out = pd.DataFrame({"origin": frame.index[test], "target_date": frame.index[test + H],
                        "actual": last[test + H], "last_fx": last[test],
                        "arima": last[test] * np.exp(arima_cum[test])})
    rows = [dict(seed=0, model=m, rmse=np.sqrt(np.mean((out[m] - out.actual) ** 2))) for m in ("last_fx", "arima")]
    for s in SEEDS:
        for name, cols in MODELS.items():
            corr = fit_correction(frame, cols, target, train, test, cutoff, s)
            out[f"{name}_{s}"] = last[test] * np.exp(arima_cum[test] + corr)
            rows.append(dict(seed=s, model=name, rmse=np.sqrt(np.mean((out[f"{name}_{s}"] - out.actual) ** 2))))
            print(rows[-1], flush=True)
    out.to_csv(OUT / "liquidity_lstm_predictions_hybrid.csv", index=False)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "liquidity_lstm_metrics_hybrid.csv", index=False)
    print(f"phi={phi:.3f} n_train={len(train)} n_test={len(test)}")
    print(res.groupby("model").rmse.agg(["mean", "std"]).to_string())


if __name__ == "__main__":
    main()
