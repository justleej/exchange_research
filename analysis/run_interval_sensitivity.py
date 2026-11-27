"""예측구간 민감도 격자(H4a)와 HAR 변동성 비교(H4b). 계획: docs/analysis_plan_extra.md
- 격자: EWMA lambda {0.90, 0.94, 0.97} x conformal 보정 창 {500, 750, 1000}, 변동성 구간별 conformal(confr)과 구간 무구분 conformal(conf).
- 계획에 있던 재추정 간격(21, 63)은 격자에서 뺐다: 구간의 중심이 RW(0)이라 점 예측 재추정의 영향을 받지 않는다(계획 대비 변경, 결과 문서에 기록).
- 모든 방식은 같은 평가 표본(모든 방식이 값을 가진 원점)에서 비교한다.
결과: analysis/results_walkforward/interval_sensitivity.csv
"""
import itertools

import numpy as np
import pandas as pd

import walkforward_daily as w
from run_time_ordered_checks import DATA

OUT = DATA.parent / "analysis" / "results_walkforward"
LAMS, WINS = (0.90, 0.94, 0.97), (500, 750, 1000)


def main():
    d = w.build_daily_frame()
    fc = w.walk_forward(d)
    reg = w.regimes(d, fc.index)
    sigmas = {f"EWMA{lam}": w.ewma_sigma(d.r1, lam).reindex(fc.index) * np.sqrt(w.H) for lam in LAMS}
    sigmas["HAR"] = w.har_sigma(d, fc.index)
    variants = {}
    for (name, sig), win in itertools.product(sigmas.items(), WINS):
        if name == "HAR" and win != 750:
            continue
        iv = w.intervals(d, fc, reg, sigma=sig, cal_window=win)
        variants[(name, win)] = iv
    common = None
    for iv in variants.values():
        ok = iv[["conf_hw_80", "confr_hw_80", "conf_hw_95", "confr_hw_95"]].notna().all(axis=1)
        common = ok if common is None else common & ok
    rows = []
    for (name, win), iv in variants.items():
        iv = iv[common]
        rg = reg[common]
        for method in ("conf", "confr"):
            for lev in w.LEVELS:
                tag = int(lev * 100)
                hw = iv[f"{method}_hw_{tag}"]
                hit = iv.y.abs() <= hw
                rows.append(dict(sigma=name, window=win, method=method, level=lev, n=int(common.sum()), coverage=hit.mean(),
                                 coverage_안정=hit[rg == "안정"].mean(), coverage_고변동=hit[rg == "고변동"].mean(),
                                 halfwidth_pct=100 * hw.mean(),
                                 winkler_pct=100 * w.interval_score(iv.y.to_numpy(), hw.to_numpy(), lev).mean()))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "interval_sensitivity.csv", index=False, float_format="%.4g")
    pd.set_option("display.width", 220)
    g = res[res.method == "confr"]
    print(f"공통 표본 n={int(common.sum())}, 원점 {common[common].index.min().date()} ~ {common[common].index.max().date()}")
    print(g.round(4).to_string(index=False))
    for lev in w.LEVELS:
        s = g[g.level == lev]
        print(f"H4a level {lev}: 전체 커버리지 범위 {s.coverage.min():.3f}~{s.coverage.max():.3f}, "
              f"|목표와 차이| 최대 {np.abs(s.coverage - lev).max():.3f} (기준 0.02)")
        har = s[s.sigma == "HAR"].winkler_pct.iloc[0]
        ew = s[(s.sigma == "EWMA0.94") & (s.window == 750)].winkler_pct.iloc[0]
        print(f"H4b level {lev}: HAR Winkler {har:.4f} vs EWMA0.94 {ew:.4f} -> {'HAR 낮음' if har < ew else 'EWMA 낮음/같음'}")


if __name__ == "__main__":
    main()
