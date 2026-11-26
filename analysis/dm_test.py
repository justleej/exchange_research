"""Diebold-Mariano 검정(제곱오차 손실, 5거래일 앞 예측이라 HAC lag 4).
analysis/results_time_ordered/liquidity_lstm_predictions<TAG>.csv의 시드별 예측과 시드 평균(앙상블)을 비교한다.
사용: python analysis/dm_test.py [접미사, 예: _s10]. 음수 평균 차이는 앞 모델이 더 정확하다는 뜻.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

OUT = Path(__file__).resolve().parent / "results_time_ordered"
H = 5


def dm(e1, e2):
    d = e1 ** 2 - e2 ** 2
    n, lag = len(d), H - 1
    dc = d - d.mean()
    var = dc @ dc / n + 2 * sum((1 - k / (lag + 1)) * (dc[k:] @ dc[:-k]) / n for k in range(1, lag + 1))
    stat = d.mean() / np.sqrt(var / n)
    stat *= np.sqrt((n + 1 - 2 * H + H * (H - 1) / n) / n)  # Harvey 소표본 보정
    return d.mean(), stat, 2 * stats.t.sf(abs(stat), n - 1)


def main(tag=""):
    p = pd.read_csv(OUT / f"liquidity_lstm_predictions{tag}.csv")
    models = ["baseline", "plus_mmf", "plus_deposit"]
    seeds = sorted({c.rsplit("_", 1)[1] for c in p if c.startswith("baseline_")})
    err = {"last_fx": (p.last_fx - p.actual).to_numpy()}
    if "arima" in p:  # Hybrid 파일이면 ARIMA 단독도 기준으로 쓴다
        err["arima"] = (p.arima - p.actual).to_numpy()
    for m in models:
        cols = [f"{m}_{s}" for s in seeds]
        err[m] = (p[cols].mean(axis=1) - p.actual).to_numpy()  # 시드 평균 예측
        for s in seeds:
            err[f"{m}_{s}"] = (p[f"{m}_{s}"] - p.actual).to_numpy()
    rows = []
    pairs = [("plus_mmf", "baseline"), ("plus_deposit", "baseline"), ("baseline", "last_fx"),
             ("plus_mmf", "last_fx"), ("plus_deposit", "last_fx")]
    if "arima" in err:
        pairs += [("baseline", "arima"), ("plus_mmf", "arima"), ("plus_deposit", "arima"), ("arima", "last_fx")]
    for a, b in pairs:
        diff, stat, pv = dm(err[a], err[b])
        rows.append(dict(kind="seed_mean", model=a, reference=b, seed="mean", mse_diff=diff, dm=stat, p=pv))
        for s in seeds:
            ea = err[a] if a in ("last_fx", "arima") else err[f"{a}_{s}"]
            eb = err[b] if b in ("last_fx", "arima") else err[f"{b}_{s}"]
            diff, stat, pv = dm(ea, eb)
            rows.append(dict(kind="per_seed", model=a, reference=b, seed=s, mse_diff=diff, dm=stat, p=pv))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / f"dm_test{tag}.csv", index=False, float_format="%.4g")
    print(res[res.kind == "seed_mean"].to_string(index=False))
    sig = res[res.kind == "per_seed"].groupby(["model", "reference"]).apply(
        lambda g: pd.Series(dict(seeds=len(g), better=int(((g.mse_diff < 0) & (g.p < 0.05)).sum()),
                                 worse=int(((g.mse_diff > 0) & (g.p < 0.05)).sum()))), include_groups=False)
    print(sig.to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
