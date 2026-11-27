"""헤지 시뮬레이션: 변동성 구간별 conformal 95% 반폭으로 헤지 비율을 정하면 같은 평균 헤지 비율의 고정 비율보다 위험이 줄어드는가.
계획과 판정 기준: docs/analysis_plan_hedge_direction.md (H8a, H8b). 입력: results_walkforward/forecast_intervals_h5_ext.csv(run_hybrid_oos.py가 만든다).
손익(명목 1, 5거래일) = s*(1-h)*y + s*h*캐리 - 거래비용*|h 변화|, 수출업자형 s=+1(USD 매도헤지), 수입업자형 s=-1.
캐리 = 금리차(%p)/100*5/252는 한국 금리 - 미국 금리 근사이며 실제 스왑포인트가 아니다(무료 자료의 한계).
shortcut: 위험 목표 B_t는 확장 창 중앙값 하나로 고정했고 튜닝하지 않았다. 바꾸려면 risk_target_h만 고친다.
"""
import numpy as np
import pandas as pd

import run_hybrid_oos as ho
import walkforward_daily as w
from run_time_ordered_checks import DATA

OUT = DATA.parent / "analysis" / "results_walkforward"
OOS_START = pd.Timestamp("2026-03-16")
FIXED = (0.0, 0.25, 0.5, 0.75, 1.0)


def risk_target_h(hw):
    """원점에서 아는 값만: 반폭 hw_t와 그 시점까지의 확장 창 중앙값 B_t로 h_t = clip(1 - B_t/hw_t, 0, 1)."""
    b = hw.expanding(min_periods=250).median()
    return (1 - b / hw).clip(0, 1)


def weekly_pnl(y, h, carry, side, tc):
    dh = h.diff().abs().fillna(0)
    return side * (1 - h) * y + side * h * carry - tc * dh


def metrics(p, h):
    loss = -p.to_numpy()
    k = max(1, int(np.ceil(0.05 * len(loss))))
    cum = p.cumsum().to_numpy()
    return dict(n=len(p), mean_h=h.mean(), pnl_mean_pct=100 * p.mean(), std_pct=100 * p.std(),
                cvar95_pct=100 * np.sort(loss)[-k:].mean(), maxdd_pct=100 * (np.maximum.accumulate(cum) - cum).max())


def cvar(x):
    k = max(1, int(np.ceil(0.05 * len(x))))
    return np.sort(-x)[-k:].mean()


def block_bootstrap_diff(a, b, stat, n_boot=2000, block=4, seed=0):
    """쌍체 이동 블록 부트스트랩: stat(a) - stat(b)의 분포."""
    rng = np.random.default_rng(seed)
    n = len(a)
    nb = int(np.ceil(n / block))
    out = np.empty(n_boot)
    for i in range(n_boot):
        idx = (rng.integers(0, n - block + 1, nb)[:, None] + np.arange(block)).ravel()[:n]
        out[i] = stat(a[idx]) - stat(b[idx])
    return out


def main():
    fx, sp = ho.extended_inputs()
    d0 = w.build_daily_frame(fx=fx, spread=sp)
    iv = pd.read_csv(OUT / "forecast_intervals_h5_ext.csv", parse_dates=["date"]).set_index("date")
    iv = iv.dropna(subset=["confr_hw_95"])
    carry_all = d0.spread.reindex(iv.index) / 100 * 5 / 252
    h_all = risk_target_h(iv.confr_hw_95)
    ok = h_all.notna()
    iv, carry_all, h_all = iv[ok], carry_all[ok], h_all[ok]
    rows, prim = [], []
    for side_name, side in (("수출업자형", 1), ("수입업자형", -1)):
        for phase in range(5):
            g = slice(phase, None, 5)
            y, c, hrt = iv.y.iloc[g], carry_all.iloc[g], h_all.iloc[g]
            samples = {"전체": y.index < OOS_START, "2015~": (y.index >= "2015-01-01") & (y.index < OOS_START),
                       "새 표본(2026-03-16~)": y.index >= OOS_START}
            for lab, m in samples.items():
                if m.sum() < 10:
                    continue
                yy, cc, hh = y[m], c[m], hrt[m]
                hstar = float(hrt[y.index < OOS_START].mean())  # 비교용 고정 비율: 새 표본 이전 평균(새 표본에는 쓰이지 않은 값)
                policies = {"위험 목표": hh, f"고정 h*={hstar:.2f}": pd.Series(hstar, index=yy.index)}
                policies.update({f"고정 {f}": pd.Series(f, index=yy.index) for f in FIXED})
                for tc in (0, 1e-4, 2e-4, 5e-4):
                    for pol, hser in policies.items():
                        p = weekly_pnl(yy, hser, cc, side, tc)
                        rows.append(dict(side=side_name, phase=phase, sample=lab, tc_bp=tc * 1e4, policy=pol, **metrics(p, hser)))
            if phase == 0:  # 주 판정: 전체 표본, 비용 1bp, 위상 0
                yy, cc, hh = y, c, hrt
                hstar = float(hh[y.index < OOS_START].mean())
                hh0 = hh[y.index < OOS_START]
                yy0, cc0 = yy[hh0.index], cc[hh0.index]
                pr = weekly_pnl(yy0, hh0, cc0, side, 1e-4).to_numpy()
                pf = weekly_pnl(yy0, pd.Series(hstar, index=hh0.index), cc0, side, 1e-4).to_numpy()
                for lab, stat in (("CVaR95 손실", cvar), ("주간 손익 표준편차", np.std)):  # np.std: ddof=0이지만 두 정책에 같이 적용
                    s = stat
                    dist = block_bootstrap_diff(pr, pf, s)
                    obs = s(pr) - s(pf)
                    lo, hi = np.percentile(dist, [2.5, 97.5])
                    prim.append(dict(side=side_name, metric=lab, mean_h=hstar, rt=100 * s(pr), fixed=100 * s(pf),
                                     diff_pct=100 * obs, ci_lo=100 * lo, ci_hi=100 * hi, adopt=hi < 0, n_weeks=len(pr)))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "hedge_simulation.csv", index=False, float_format="%.4g")
    pr = pd.DataFrame(prim)
    pr.to_csv(OUT / "hedge_primary.csv", index=False, float_format="%.4g")
    pd.set_option("display.width", 220)
    print(pr.round(4).to_string(index=False))
    show = res[(res.phase == 0) & (res.tc_bp == 1) & (res["sample"] == "전체")]
    print(show.drop(columns=["phase", "tc_bp", "sample"]).round(4).to_string(index=False))
    sgn = res[(res.tc_bp == 1) & (res["sample"] == "전체") & res.policy.isin(["위험 목표"]) | (res.tc_bp == 1) & (res["sample"] == "전체") & res.policy.str.startswith("고정 h*")]
    pv = sgn.pivot_table(index=["side", "phase"], columns=sgn.policy.str.startswith("고정"), values="cvar95_pct")
    print("위상별 CVaR95(위험 목표 - 고정 h*) 부호:", (pv[False] - pv[True]).round(3).to_dict())
    newp = res[(res.phase == 0) & (res.tc_bp == 1) & (res["sample"] == "새 표본(2026-03-16~)")]
    print(newp.drop(columns=["phase", "tc_bp", "sample"]).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
