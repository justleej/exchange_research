"""계획 외 사후 점검: Hybrid+MMF의 방향 정확도가 MMF를 더 늦춰도 유지되는가(시점 정렬 인공물 점검).
MMF 변화(dm1)를 추가로 1일, 4일 더 늦춰(총 지연 2일, 5일) 같은 Hybrid+MMF walk-forward를 다시 돌려 방향 정확도와 RMSE를 비교한다.
지연을 늘려도 우위가 유지되면 같은 날 환율과의 시점 겹침 때문이라는 설명이 약해지고, 바로 사라지면 짧은 시차의 정보다.
설정은 run_hybrid_walkforward.py와 같다(고정, 한 번만). 결과: results_walkforward/hybrid_mmf_lag_check.csv
"""
import warnings

import numpy as np
import pandas as pd
from scipy.stats import binomtest

import run_hybrid_walkforward as h
import walkforward_daily as w
from run_time_ordered_checks import DATA

warnings.filterwarnings("ignore")
OUT = DATA.parent / "analysis" / "results_walkforward"


def main():
    d0 = w.build_daily_frame()
    fc = w.walk_forward(d0)
    base = h.add_daily_features(d0)
    rows = []
    for extra in (1, 4):
        d = base.copy()
        d["dm1"] = d.dm1.shift(extra)  # 추가 지연
        d = d.dropna(subset=["dm1"])
        origins = np.array([d.index.get_loc(t) for t in fc.index if t in d.index])
        keep = fc.index.intersection(d.index)
        pred = h.hybrid_predictions(d, origins, variants={"Hybrid+MMF": h.VARIANTS["Hybrid+MMF"]})
        y = fc.loc[keep, "y"]
        e0 = y.to_numpy()
        for m in ("ARIMA", "Hybrid+MMF"):
            e = (y - pred[m].loc[keep]).to_numpy()
            hit = (pred[m].loc[keep] > 0) == (y > 0)
            rows.append(dict(extra_lag_days=extra, total_mmf_lag_days=1 + extra, model=m, n=len(y),
                             ratio_vs_RW=np.sqrt(np.mean(e ** 2) / np.mean(e0 ** 2)), dir_acc=hit.mean(),
                             dir_binom_p=binomtest(int(hit.sum()), len(hit), 0.5).pvalue))
        print(rows[-1], flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "hybrid_mmf_lag_check.csv", index=False, float_format="%.4g")
    pd.set_option("display.width", 200)
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
