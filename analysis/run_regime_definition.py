"""변동성 레짐 정의 점검. 계획과 판정 기준: docs/analysis_plan_regime_definition.md (H10a, H10b).
R0 현행(20일 변동성 67%), R1 3구간, R2 EWMA, R3 60일, R4 2상태 Markov 전환(HMM)으로 라벨만 바꿔 conformal 95% 구간 품질을 비교한다.
입력: 확장 자료(run_hybrid_oos.extended_inputs). 결과: results_walkforward/regime_definition.csv
shortcut: HMM은 수렴 실패 시 직전 모수를 쓴다(실패 횟수를 출력), 라벨 정의를 늘리려면 LABELS에 함수만 추가한다.
"""
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from statsmodels.stats.multitest import multipletests

import run_hybrid_oos as ho
import walkforward_daily as w
from run_time_ordered_checks import DATA

OUT = DATA.parent / "analysis" / "results_walkforward"
OOS_START = pd.Timestamp("2026-03-16")
LEV = 0.95


def _exceed(x, q=0.67):
    thr = x.expanding(min_periods=250).quantile(q).shift(1)
    return pd.Series(np.where(x > thr, "고변동", "안정"), index=x.index).where(thr.notna())


def r0(d):
    return w.regimes(d, d.index)


def r1(d):
    lo = d.vol20.expanding(min_periods=250).quantile(1 / 3).shift(1)
    hi = d.vol20.expanding(min_periods=250).quantile(2 / 3).shift(1)
    lab = np.where(d.vol20 > hi, "고", np.where(d.vol20 > lo, "중", "저"))
    return pd.Series(lab, index=d.index).where(hi.notna())


def r2(d):
    return _exceed(w.ewma_sigma(d.r1))


def r3(d):
    return _exceed(d.r1.rolling(60).std())


def r4(d, block=252, first=1000):
    """블록 시작까지의 자료로 추정한 모수로 전체를 필터링한다. 필터링 확률은 그날까지의 자료만 쓴다."""
    r = (100 * d.r1).to_numpy()
    prob = np.full(len(r), np.nan)
    params, fails = None, 0
    for s in range(first, len(r), block):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = MarkovRegression(r[:s], 2, switching_trend=False, switching_variance=True).fit(disp=False, maxiter=200)
            params = res.params
        except Exception:
            fails += 1
            if params is None:
                continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fr = MarkovRegression(r, 2, switching_trend=False, switching_variance=True).filter(params)
        hi_state = int(np.argmax([params[-2], params[-1]]))  # 마지막 두 모수가 상태별 분산
        prob[s: s + block] = fr.filtered_marginal_probabilities[:, hi_state][s: s + block]
    print(f"HMM 추정 실패 {fails}회")
    return pd.Series(np.where(prob > 0.5, "고변동", "안정"), index=d.index).where(~np.isnan(prob))


LABELS = {"R0 현행": r0, "R1 3구간": r1, "R2 EWMA": r2, "R3 60일": r3, "R4 HMM": r4}


def block_boot_diff(a, b, n_boot=2000, block=20, seed=0):
    rng = np.random.default_rng(seed)
    n, nb = len(a), int(np.ceil(len(a) / block))
    out = np.empty(n_boot)
    for i in range(n_boot):
        idx = (rng.integers(0, n - block + 1, nb)[:, None] + np.arange(block)).ravel()[:n]
        out[i] = a[idx].mean() - b[idx].mean()
    return out


def main():
    fx, sp = ho.extended_inputs()
    d = w.build_daily_frame(fx=fx, spread=sp)
    fc = w.walk_forward(d)
    ivs = {}
    for name, f in LABELS.items():
        lab = f(d).reindex(fc.index)
        ivs[name] = w.intervals(d, fc, lab)[f"confr_hw_{int(LEV * 100)}"]
        ivs[name + "_lab"] = lab
    hw = pd.DataFrame({k: v for k, v in ivs.items() if not k.endswith("_lab")})
    ok = hw.notna().all(axis=1)
    y = fc.y[ok].to_numpy()
    idx = fc.index[ok]
    pre = idx < OOS_START
    rows, scores = [], {}
    for name in LABELS:
        h = hw[name][ok].to_numpy()
        lab = ivs[name + "_lab"][ok].to_numpy()
        sc = w.interval_score(y, h, LEV)
        scores[name] = sc
        cov = np.abs(y) <= h
        reg_cov = {f"cov_{g}": cov[lab == g].mean() for g in sorted(set(lab))}
        rows.append(dict(regime=name, n=len(y), n_pre=int(pre.sum()), cov_all=cov.mean(), score_pre=sc[pre].mean(),
                         score_new=sc[~pre].mean(), min_regime_cov=min(reg_cov.values()), regime_cov=str({k: round(v, 3) for k, v in reg_cov.items()}),
                         share=str({g: round((lab == g).mean(), 2) for g in sorted(set(lab))})))
    res = pd.DataFrame(rows)
    p_raw, diffs = [], []
    for name in list(LABELS)[1:]:
        dist = block_boot_diff(scores[name][pre], scores["R0 현행"][pre])
        obs = scores[name][pre].mean() - scores["R0 현행"][pre].mean()
        lo, hi = np.percentile(dist, [2.5, 97.5])
        p_raw.append(min(1, 2 * min((dist >= 0).mean(), (dist <= 0).mean())))
        diffs.append((obs, lo, hi))
    holm = multipletests(p_raw, method="holm")[1]
    res["diff_vs_R0_pre"] = [np.nan] + [x[0] for x in diffs]
    res["ci_lo"] = [np.nan] + [x[1] for x in diffs]
    res["ci_hi"] = [np.nan] + [x[2] for x in diffs]
    res["p_holm"] = [np.nan] + list(holm)
    res["diff_vs_R0_new"] = [np.nan] + [scores[n][~pre].mean() - scores["R0 현행"][~pre].mean() for n in list(LABELS)[1:]]
    res["H10a"] = [False] + [bool(x[2] < 0 and p < 0.05) for x, p in zip(diffs, holm)]
    res["H10b"] = (res.cov_all.between(0.93, 0.97)) & (res.min_regime_cov >= 0.92)
    res["adopt"] = res.H10a & res.H10b
    res.to_csv(OUT / "regime_definition.csv", index=False, float_format="%.4g")
    pd.set_option("display.width", 250, "display.max_colwidth", 70)
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
