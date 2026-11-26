"""정상화 입력 + 5일 수익률 목표로 기본/MMF/수시입출식 LSTM을 10시드 비교한다(시간순 분할, 같은 테스트 날짜).
수준값 설정의 외삽 문제는 diagnose_lstm.py 참고. 예측은 liquidity_lstm_predictions_stationary.csv로 저장하고
`python dm_test.py _stationary`로 DM 검정을 한다. 수시입출식은 다음 달 말부터 사용 가능하다고 가정한다."""
import numpy as np
import pandas as pd

from diagnose_lstm_stationary import H, L, run
from run_time_ordered_checks import OUT, lstm_inputs, series

SEEDS = tuple(range(42, 52))
MODELS = {"baseline": ["r", "dspread"], "plus_mmf": ["r", "dspread", "dmmf"],
          "plus_deposit": ["r", "dspread", "ddeposit"]}


def main():
    frame = lstm_inputs(series("exchange_rate/USD_KRW_processed.csv", "USD_KRW"))
    deposit = series("m2/KOR/M2_details_processed.csv", "M2_수시입출식저축성예금")
    deposit.index = deposit.index + pd.offsets.MonthEnd(0) + pd.offsets.MonthEnd(1)
    dep = pd.merge_asof(frame[[]], deposit.rename("dep").to_frame(), left_index=True, right_index=True,
                        allow_exact_matches=False, tolerance=pd.Timedelta("45D"))["dep"]
    frame["r"] = np.log(frame.fx).diff()
    frame["dspread"] = frame.spread.diff()
    frame["dmmf"] = np.log(frame.mmf).diff()
    frame["ddeposit"] = np.log(dep).diff()  # 월 1회만 0이 아님
    feats = ["r", "dspread", "dmmf", "ddeposit"]
    cutoff = int(len(frame) * 0.8)
    idx = np.arange(L, len(frame) - H)
    ok = frame[feats].notna().all(axis=1).rolling(L).sum().eq(L).to_numpy()
    idx = idx[ok[idx]]
    train, test = idx[idx + H < cutoff], idx[idx >= cutoff]
    out = pd.DataFrame({"origin": frame.index[test], "target_date": frame.index[test + H],
                        "actual": frame.fx.iloc[test + H].to_numpy(), "last_fx": frame.fx.iloc[test].to_numpy()})
    rows = []
    for s in SEEDS:
        for name, cols in MODELS.items():
            out[f"{name}_{s}"] = run(frame, cols, train, test, cutoff, s)
            rows.append(dict(seed=s, model=name, rmse=np.sqrt(np.mean((out[f"{name}_{s}"] - out.actual) ** 2))))
            print(rows[-1], flush=True)
    rows += [dict(seed=0, model="last_fx", rmse=np.sqrt(np.mean((out.last_fx - out.actual) ** 2)))]
    out.to_csv(OUT / "liquidity_lstm_predictions_stationary.csv", index=False)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "liquidity_lstm_metrics_stationary.csv", index=False)
    print(f"n_train={len(train)} n_test={len(test)}")
    print(res.groupby("model").rmse.agg(["mean", "std"]).to_string())


if __name__ == "__main__":
    main()
