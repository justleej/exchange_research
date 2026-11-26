"""LSTM이 마지막 환율 유지보다 크게 나쁜 원인 진단: 수준 예측 vs 변화량 예측, 학습 범위 이탈 비율.
사용: python analysis/diagnose_lstm.py  (결과: results_time_ordered/lstm_diagnosis.csv)"""
import numpy as np
import pandas as pd

from run_time_ordered_checks import OUT, fit_lstm, lstm_inputs, series

SEEDS = (42, 43, 44)
MODELS = {"baseline": ["fx", "spread"], "plus_mmf": ["fx", "spread", "mmf"]}


def main():
    frame = lstm_inputs(series("exchange_rate/USD_KRW_processed.csv", "USD_KRW"))
    length, horizon = 30, 5
    cutoff = int(len(frame) * 0.8)
    origins = np.arange(length - 1, len(frame) - horizon)
    ok = frame.notna().all(axis=1).rolling(length).sum().eq(length).to_numpy()
    origins = origins[ok[origins]]
    train, test = origins[origins + horizon < cutoff], origins[origins >= cutoff]
    fx = frame.fx
    lo, hi = fx.iloc[:cutoff].min(), fx.iloc[:cutoff].max()
    tgt = fx.iloc[test + horizon].to_numpy()
    print(f"학습 환율 범위 {lo:.0f}~{hi:.0f}, 테스트 목표 중 범위 밖 {np.mean((tgt < lo) | (tgt > hi)):.1%}")
    last = fx.iloc[test].to_numpy()
    rows = [dict(model="last_fx", target="-", seed=0, rmse=np.sqrt(np.mean((last - tgt) ** 2)))]
    for target in ("level", "change"):
        for name, cols in MODELS.items():
            for s in SEEDS:
                pred = fit_lstm(frame, cols, train, test, cutoff, length, horizon, s, target)
                rows.append(dict(model=name, target=target, seed=s, rmse=np.sqrt(np.mean((pred - tgt) ** 2))))
                print(rows[-1], flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "lstm_diagnosis.csv", index=False)
    print(res.groupby(["model", "target"]).rmse.agg(["mean", "std"]).to_string())


if __name__ == "__main__":
    main()
