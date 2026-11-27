"""Hybrid+MMF 5거래일 방향 신호 점검. 계획과 판정 기준: docs/analysis_plan_hedge_direction.md (H9a~H9d).
입력: results_walkforward/hybrid_oos_predictions.csv(run_hybrid_oos.py), data/oos/ 의 환율(ECOS, Yahoo).
H9a 체결 지연, H9b 다른 환율 출처, H9c 비용과 캐리 후 손익, H9d 새 표본(2026-03-16~) 재현.
이항검정은 겹치는 5일 목표를 독립으로 보아 유의성이 과대하므로(기존 점검도 같음) 블록 부트스트랩과 겹치지 않는 부분표본 p를 함께 낸다.
결과: results_walkforward/hybrid_direction_oos.csv
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import binom, binomtest
from statsmodels.stats.multitest import multipletests

import run_hybrid_oos as ho
import walkforward_daily as w
from run_time_ordered_checks import DATA, series

OUT = DATA.parent / "analysis" / "results_walkforward"
OOS_START = pd.Timestamp("2026-03-16")
MODEL = "Hybrid+MMF"


def hit_tests(hit, label, rows, note=""):
    """적중 여부 Series(시간순)에 대해 이항검정(겹침 무시), 겹치지 않는 5개 위상 평균 p, 블록(5) 부트스트랩 p를 기록한다."""
    h = hit.to_numpy().astype(float)
    n = len(h)
    p_binom = binomtest(int(h.sum()), n, 0.5, alternative="greater").pvalue
    p_phase = [binomtest(int(h[k::5].sum()), len(h[k::5]), 0.5, alternative="greater").pvalue for k in range(5)]
    rng = np.random.default_rng(0)
    nb = int(np.ceil(n / 5))
    means = np.array([h[(rng.integers(0, n - 4, nb)[:, None] + np.arange(5)).ravel()[:n]].mean() for _ in range(2000)])
    p_boot = (1 + (means <= 0.5).sum()) / (1 + len(means))  # 귀무 0.5 아래에서 관측 이상이 나올 확률의 근사
    lo, hi = np.percentile(means, [2.5, 97.5])
    rows.append(dict(test=label, n=n, hit=h.mean(), p_binom=p_binom, p_nonoverlap_mean=np.mean(p_phase),
                     p_nonoverlap_max=max(p_phase), p_block_boot=p_boot, ci_lo=lo, ci_hi=hi, note=note))
    return rows[-1]


def main():
    pred = pd.read_csv(OUT / "hybrid_oos_predictions.csv", parse_dates=["date"]).set_index("date")
    fx, sp = ho.extended_inputs()
    d0 = w.build_daily_frame(fx=fx, spread=sp)
    lf = np.log(d0.fx)
    pre = pred[~pred.oos.astype(bool)]
    new = pred[pred.oos.astype(bool)]
    rows = []

    # H9a 체결 지연: 목표를 t+k ~ t+k+5로 바꾼다
    ha = []
    for k in (1, 2):
        yk = (lf.shift(-(5 + k)) - lf.shift(-k)).reindex(pre.index)
        ok = yk.notna()
        r = hit_tests(((pre[MODEL] > 0) == (yk > 0))[ok], f"H9a 지연 {k}일", rows, note="목표 = t+k에서 t+k+5")
        ha.append(r["p_binom"])
    adj = multipletests(ha, method="holm")[1]
    for r, a in zip(rows[-2:], adj):
        r["note"] += f", Holm(2) 이항 p={a:.3g}"

    # H9b 다른 출처: Yahoo 종가로 같은 날짜쌍의 5일 수익률을 다시 만든다
    yh = series("oos/usd_krw_yahoo.csv", "USD_KRW_yahoo")
    pos = pd.Series(np.arange(len(d0)), index=d0.index)
    t0 = pre.index[pre.index.isin(pos.index)]
    t5 = pd.DatetimeIndex([d0.index[pos[t] + 5] if pos[t] + 5 < len(d0) else pd.NaT for t in t0])
    ya = np.log(yh.reindex(t5, method="ffill").to_numpy()) - np.log(yh.reindex(t0, method="ffill").to_numpy())
    ya = pd.Series(ya, index=t0)
    ok = ya.notna() & (yh.index.min() <= t0)
    r = hit_tests(((pre.loc[t0, MODEL] > 0) == (ya > 0))[ok], "H9b Yahoo 종가 기준", rows,
                  note=f"ECOS 대비 방향 일치율 {((ya > 0) == (pre.loc[t0, 'y'] > 0))[ok].mean():.3f}, 표본 {t0[ok].min().date()}~")
    hit_tests(((pre.loc[t0, MODEL] > 0) == (pre.loc[t0, "y"] > 0))[ok], "H9b 대조: 같은 표본 ECOS 기준", rows)

    # H9c 비용과 캐리 후 손익: 예측 부호 +1(USD 매수)/-1(매도), 주간 겹치지 않는 격자
    carry = (d0.spread / 100 * 5 / 252).reindex(pre.index)
    for c_bp in (0, 1, 2, 5):
        for phase in range(1 if c_bp != 1 else 5):
            s = np.where(pre[MODEL] > 0, 1.0, -1.0)[phase::5]
            y = pre.y.to_numpy()[phase::5]
            c = carry.to_numpy()[phase::5]
            ds = np.abs(np.diff(s, prepend=s[0]))
            pnl = s * (y - c) - c_bp * 1e-4 * ds
            fit = sm.OLS(pnl, np.ones(len(pnl))).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
            rows.append(dict(test=f"H9c 비용 {c_bp}bp/단위 위상 {phase}", n=len(pnl), hit=np.nan, p_binom=np.nan,
                             p_nonoverlap_mean=np.nan, p_nonoverlap_max=np.nan, p_block_boot=np.nan, ci_lo=np.nan, ci_hi=np.nan,
                             note=f"평균 주간 손익 {100 * pnl.mean():.4f}%, HAC t={float(np.asarray(fit.tvalues)[0]):.2f}, 평균 포지션 전환 {ds.mean() / 2:.2f}회/주, 항상 매수 평균 {100 * (y - c).mean():.4f}%"))

    # H9d 새 표본
    nh = ((new[MODEL] > 0) == (new.y > 0))
    r = hit_tests(nh, "H9d 새 표본 Hybrid+MMF", rows, note=f"{new.index.min().date()}~{new.index.max().date()}")
    for m in ("Hybrid 기본", "ARIMA"):
        hit_tests(((new[m] > 0) == (new.y > 0)), f"H9d 새 표본 {m}(대조)", rows)
    for n_eff, lab in ((len(new), "겹치는 원점"), (len(new) // 5, "겹치지 않는 부분표본")):
        k = int(binom.isf(0.05, n_eff, 0.5)) + 1  # 단측 5% 임계 적중 수
        rows.append(dict(test=f"H9d 검정력 ({lab} n={n_eff})", n=n_eff, hit=np.nan, p_binom=np.nan, p_nonoverlap_mean=np.nan,
                         p_nonoverlap_max=np.nan, p_block_boot=np.nan, ci_lo=np.nan, ci_hi=np.nan,
                         note=f"참 적중률 0.536일 때 단측 5% 검정력 {binom.sf(k - 1, n_eff, 0.536):.2f}, 0.55일 때 {binom.sf(k - 1, n_eff, 0.55):.2f}"))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "hybrid_direction_oos.csv", index=False, float_format="%.4g")
    pd.set_option("display.width", 250, "display.max_colwidth", 140)
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
