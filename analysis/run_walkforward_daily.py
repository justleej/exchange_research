"""walk-forward 일별 5거래일 평가: 점 예측(RW 대비, 방향 정확도, 안정/고변동 구간)과 예측구간(커버리지, 폭, 구간 점수).
결과: analysis/results_walkforward/*.csv (forecast_intervals_h5.csv는 의사결정 시뮬레이션의 입력)
"""
import numpy as np
import pandas as pd
from scipy import stats

import walkforward_daily as w
from dm_test import dm
from run_time_ordered_checks import DATA

OUT = DATA.parent / "analysis" / "results_walkforward"
MODELS = ["AR", "A", "B"]


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def point_table(fc, r5, label):
    rows = []
    e0 = fc.y.to_numpy()  # RW 오차는 y 자체
    rows.append(dict(sample=label, model="RW", n=len(fc), rmse_pct=100 * np.sqrt(np.mean(e0 ** 2)),
                     mae_pct=100 * np.mean(np.abs(e0)), rmse_ratio_vs_RW=1.0))
    up = (fc.y > 0).mean()
    mom = (np.sign(r5.reindex(fc.index)) == np.sign(fc.y)).mean()
    rows[0].update(dir_acc=up, dir_note="항상 상승 예측(상승 비율)")
    rows.append(dict(sample=label, model="모멘텀(부호 r5)", n=len(fc), dir_acc=mom, dir_note="5일 수익률 부호 유지"))
    for m in MODELS:
        e = fc.y.to_numpy() - fc[m].to_numpy()
        diff, stat, p = dm(e, e0)
        hit = (np.sign(fc[m]) == np.sign(fc.y))
        lo, hi = wilson(int(hit.sum()), len(hit))
        rows.append(dict(sample=label, model=m, n=len(fc), rmse_pct=100 * np.sqrt(np.mean(e ** 2)),
                         mae_pct=100 * np.mean(np.abs(e)), rmse_ratio_vs_RW=np.sqrt(np.mean(e ** 2) / np.mean(e0 ** 2)),
                         dm_vs_RW_p=p, dir_acc=hit.mean(), dir_ci_low=lo, dir_ci_high=hi, dir_note="모형 부호"))
    eA, eB = fc.y - fc["A"], fc.y - fc["B"]
    _, _, p = dm(eB.to_numpy(), eA.to_numpy())
    rows.append(dict(sample=label, model="B vs A(유동성 추가 효과)", n=len(fc), dm_vs_RW_p=p,
                     rmse_ratio_vs_RW=np.sqrt(np.mean(eB ** 2) / np.mean(eA ** 2)), dir_note="DM p는 B와 A의 차이"))
    return rows


def interval_table(iv, reg, label):
    rows = []
    for lev in w.LEVELS:
        tag = int(lev * 100)
        for method in ("const", "norm", "conf"):
            hw = iv[f"{method}_hw_{tag}"]
            ok = hw.notna()
            hit = (iv.y.abs() <= hw)[ok]
            score = w.interval_score(iv.y[ok].to_numpy(), hw[ok].to_numpy(), lev)
            row = dict(sample=label, method=method, level=lev, n=int(ok.sum()), coverage=hit.mean(),
                       mean_halfwidth_pct=100 * hw[ok].mean(), interval_score_pct=100 * score.mean())
            for g in ("안정", "고변동"):
                sel = (reg == g)[ok]
                row[f"coverage_{g}"] = hit[sel.to_numpy()].mean()
            rows.append(row)
    return rows


def main():
    OUT.mkdir(exist_ok=True)
    d = w.build_daily_frame()
    fc = w.walk_forward(d)
    reg = w.regimes(d, fc.index)
    r5 = d.r5
    iv = w.intervals(d, fc)
    valid = iv.dropna().index  # 구간 보정 표본이 쌓인 이후 공통 날짜
    fcv, ivv, regv = fc.loc[valid], iv.loc[valid], reg.loc[valid]

    pts = point_table(fcv, r5, "전체")
    for g in ("안정", "고변동"):
        sel = regv == g
        pts += point_table(fcv[sel], r5, g)
    pts += [dict(r, sample=f"{lab}~") for lab, sub in (("2015-01", fcv[fcv.index >= "2015-01-01"]),) for r in point_table(sub, r5, lab)]
    pd.DataFrame(pts).to_csv(OUT / "walkforward_point_metrics.csv", index=False, float_format="%.4g")

    ivt = interval_table(ivv, regv, "전체") + [dict(r, sample="2015-01~") for r in
                                                interval_table(ivv[ivv.index >= "2015-01-01"], regv[regv.index >= "2015-01-01"], "2015-01~")]
    pd.DataFrame(ivt).to_csv(OUT / "walkforward_interval_metrics.csv", index=False, float_format="%.4g")

    export = pd.concat([ivv, fcv.drop(columns="y"), regv.rename("regime")], axis=1)
    export.to_csv(OUT / "forecast_intervals_h5.csv", float_format="%.6g")
    pd.set_option("display.width", 230)
    print(f"origins {valid.min().date()} ~ {valid.max().date()} n={len(valid)}")
    print(pd.DataFrame(pts).round(4).to_string(index=False))
    print(pd.DataFrame(ivt).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
