"""정상화 입력(일별 환율 수익률, 금리차 변화, MMF 변화율) + 5일 누적수익률 목표 LSTM을 마지막 환율 유지와 비교한다.
진단용 한 가지 설정이며 하이퍼파라미터는 튜닝하지 않았다. 결과: results_time_ordered/lstm_diagnosis_stationary.csv"""
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from LSTM.lstm_mmf.train_eval_extended import ExRateLSTM, train_model
from run_time_ordered_checks import OUT, lstm_inputs, series

SEEDS = (42, 43, 44, 45, 46)
L, H = 30, 5
MODELS = {"baseline": ["r", "dspread"], "plus_mmf": ["r", "dspread", "dmmf"]}


def run(frame, cols, train, test, cutoff, seed, epochs=20, lr=0.001):
    torch.manual_seed(seed)
    x = frame[cols]
    z = StandardScaler().fit(x.iloc[:cutoff].dropna()).transform(x.fillna(0))
    fut = (np.log(frame.fx).shift(-H) - np.log(frame.fx)).to_numpy()
    sc = fut[train].std()
    win = lambda idx: torch.tensor(np.stack([z[i - L + 1:i + 1] for i in idx]), dtype=torch.float32)
    y = torch.tensor((fut[train] / sc).reshape(-1, 1), dtype=torch.float32)
    model = ExRateLSTM(len(cols), 32, 1)
    train_model(model, DataLoader(TensorDataset(win(train), y), batch_size=32, shuffle=True),
                torch.nn.MSELoss(), torch.optim.Adam(model.parameters(), lr=lr), epochs)
    model.eval()
    with torch.no_grad():
        r = model(win(test)).numpy().ravel() * sc
    return frame.fx.iloc[test].to_numpy() * np.exp(r)


def main():
    frame = lstm_inputs(series("exchange_rate/USD_KRW_processed.csv", "USD_KRW"))
    frame["r"] = np.log(frame.fx).diff()
    frame["dspread"] = frame.spread.diff()
    frame["dmmf"] = np.log(frame.mmf).diff()
    cutoff = int(len(frame) * 0.8)
    idx = np.arange(L, len(frame) - H)
    ok = frame[["r", "dspread", "dmmf"]].notna().all(axis=1).rolling(L).sum().eq(L).to_numpy()
    idx = idx[ok[idx]]
    train, test = idx[idx + H < cutoff], idx[idx >= cutoff]
    tgt = frame.fx.iloc[test + H].to_numpy()
    last = frame.fx.iloc[test].to_numpy()
    rows = [dict(model="last_fx", seed=0, rmse=np.sqrt(np.mean((last - tgt) ** 2)))]
    for name, cols in MODELS.items():
        for s in SEEDS:
            p = run(frame, cols, train, test, cutoff, s)
            rows.append(dict(model=name, seed=s, rmse=np.sqrt(np.mean((p - tgt) ** 2))))
            print(rows[-1], flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "lstm_diagnosis_stationary.csv", index=False)
    print(f"n_test={len(test)}")
    print(res.groupby("model").rmse.agg(["mean", "std"]).to_string())


if __name__ == "__main__":
    main()
