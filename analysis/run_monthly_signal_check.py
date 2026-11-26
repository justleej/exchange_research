"""수시입출식저축성예금 성장률 -> 6개월 뒤 환율 신호(run_monthly_horizon.py, CW p=0.002)의 추가 확인.

1) 안정성: 표본 내 계수를 전반/후반으로 나누고, 표본 밖 Clark-West 기여를 연도별로 나누고, 위기 구간(2008-09~2009-03, 2020-02~2020-04)을 뺀다.
2) 다중검정 보정: 유동성 3종 x 지평 4개(12개) 전체에서 가장 큰 CW t가 우연히 이만큼 클 확률을 유동성 시계열의 순환 이동
   (자기상관은 유지, 환율과의 연결은 끊음) 순열로 구한다(최댓값 검정). 단일 검정 순열 p도 함께 본다.
3) 대조군: 같은 M2 구성요소 중 정기예적금, 현금통화 성장률로 같은 검증.
4) 금리 국면 통제: 기준금리 변화와 금리차 수준을 모형 A, B에 추가.
결과: results_power_regimes/monthly_signal_check.csv, monthly_signal_permutation.csv
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from run_monthly_horizon import HORIZONS, LIQ, OOS_START, hac_fit, make_xy, monthly_base
from run_time_ordered_checks import DATA, series

OUT = DATA.parent / "analysis" / "results_power_regimes"
NAME = "수시입출식저축성예금"
B = 300


def fast_oos(d, h, ctrl=("r", "ds")):
    """run_monthly_horizon.oos와 같은 확장 창 예측. 조정 손실차 f(Clark-West)의 시계열과 t를 돌려준다."""
    d = d.dropna(subset=["y"])
    idx, y = d.index, d.y.to_numpy()
    C = np.column_stack([np.ones(len(d)), d[list(ctrl)].to_numpy()])
    CX = np.column_stack([C, d.x.to_numpy()])
    start = np.searchsorted(idx, pd.Timestamp(OOS_START))
    f = {}
    for i in range(start, len(d)):
        m = i - h + 1
        if m < 60:
            continue
        ba = np.linalg.lstsq(C[:m], y[:m], rcond=None)[0]
        bb = np.linalg.lstsq(CX[:m], y[:m], rcond=None)[0]
        pa, pb = C[i] @ ba, CX[i] @ bb
        f[idx[i]] = (y[i] - pa) ** 2 - (y[i] - pb) ** 2 + (pa - pb) ** 2
    return pd.Series(f)


def nw_t(f, h):
    v = f.to_numpy()
    n, L = len(v), max(h - 1, 1)
    e = v - v.mean()
    s = e @ e / n + 2 * sum((1 - k / (L + 1)) * (e[k:] @ e[:-k]) / n for k in range(1, L + 1))
    return v.mean() / np.sqrt(s / n)


def liquidity_x(df, s, lag=1):
    s = s.copy()
    s.index = s.index + pd.offsets.MonthEnd(0)
    g = np.log(s).diff().reindex(df.index).shift(lag)
    return (g - g.mean()) / g.std()


def main():
    base = monthly_base()
    macro = pd.read_csv(DATA / "integrated_macro_targets.csv", parse_dates=["Date"])
    macro["Date"] = macro.Date + pd.offsets.MonthEnd(0)
    base["dr"] = macro.set_index("Date").Policy_Rate_KOR.diff().reindex(base.index)
    base["spread_lvl"] = base.spread
    xs = {k: liquidity_x(base, series(*v)) for k, v in LIQ.items()}
    det = pd.read_csv(DATA / "m2/KOR/M2_details_processed.csv", parse_dates=["observation_date"]).set_index("observation_date")
    placebo = {"정기예적금(대조)": det.iloc[:, 1].dropna(), "현금통화(대조)": det.iloc[:, 4].dropna()}
    rows = []

    # 기준 재현
    h = 6
    d = make_xy(base, xs[NAME], h)
    f = fast_oos(d, h)
    base_t = nw_t(f, h)
    rows.append(dict(check="기준 재현(CW t, 원 코드 2.886)", horizon=h, value=base_t, n=len(f)))

    # 1) 안정성
    dd = d.dropna(subset=["y"])
    for label, part in (("표본 내 전반(~2012)", dd[dd.index <= "2012-12-31"]), ("표본 내 후반(2013~)", dd[dd.index >= "2013-01-31"])):
        fit = hac_fit(part.y, part[["r", "ds", "x"]], h)
        rows.append(dict(check=label, horizon=h, value=fit.params["x"], p=fit.pvalues["x"], n=len(part)))
    share = f.groupby(f.index.year).sum() / f.sum()
    for yr, v in share.items():
        rows.append(dict(check=f"CW 기여 비중 {yr}", horizon=h, value=v))
    for label, spans in (("위기 구간 제외", [("2008-09-30", "2009-03-31"), ("2020-02-29", "2020-04-30")]),):
        keep = f.copy()
        for a, b in spans:
            keep = keep[~((keep.index >= a) & (keep.index <= b))]
        rows.append(dict(check=f"{label} CW t", horizon=h, value=nw_t(keep, h), n=len(keep)))

    # 3) 대조군, 4) 금리 국면 통제
    for k, s in placebo.items():
        for hh in (6, 12):
            ff = fast_oos(make_xy(base, liquidity_x(base, s), hh), hh)
            rows.append(dict(check=f"{k} CW t", horizon=hh, value=nw_t(ff, hh), n=len(ff)))
    for hh in (6, 12):
        ff = fast_oos(make_xy(base, xs[NAME], hh).dropna(subset=["dr"]), hh, ctrl=("r", "ds", "dr", "spread_lvl"))
        rows.append(dict(check="금리 국면 통제(기준금리 변화, 금리차 수준) CW t", horizon=hh, value=nw_t(ff, hh), n=len(ff)))

    # 2) 순열(최댓값) 검정
    rng = np.random.default_rng(42)
    n = len(base)
    obs = {(k, hh): nw_t(fast_oos(make_xy(base, xs[k], hh), hh), hh) for k in LIQ for hh in HORIZONS}
    obs_max = max(obs.values())
    perm = []
    for _ in range(B):
        t_all = {}
        for k in LIQ:
            shifted = pd.Series(np.roll(xs[k].to_numpy(), int(rng.integers(12, n - 12))), index=xs[k].index)
            for hh in HORIZONS:
                t_all[(k, hh)] = nw_t(fast_oos(make_xy(base, shifted, hh), hh), hh)
        perm.append(t_all)
    maxes = np.array([max(p.values()) for p in perm])
    single = np.array([p[(NAME, 6)] for p in perm])
    summary = [dict(test="최댓값 순열(12개 중 최대 CW t)", observed=obs_max, p=(1 + (maxes >= obs_max).sum()) / (1 + B)),
               dict(test=f"단일 순열({NAME}, h=6)", observed=obs[(NAME, 6)], p=(1 + (single >= obs[(NAME, 6)]).sum()) / (1 + B))]
    pd.DataFrame(rows).to_csv(OUT / "monthly_signal_check.csv", index=False, float_format="%.4g")
    pd.DataFrame(summary).to_csv(OUT / "monthly_signal_permutation.csv", index=False, float_format="%.4g")
    pd.set_option("display.width", 200)
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print(pd.DataFrame(summary).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
