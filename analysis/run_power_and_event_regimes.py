"""검정력(최소 탐지 효과)과 외부 기준 레짐에서의 M2/MMF -> 환율 Granger 검정.

외부 기준 레짐: 이동상관이 아니라 데이터를 보기 전에 정한 위기 구간 목록(EVENT_WINDOWS)과 VIX >= 30인 달.
EVENT_WINDOWS의 날짜는 일반적으로 알려진 위기 시기를 기준으로 한 근사값이다(월 단위 반올림). 결과를 본 뒤 바꾸지 않는다.
검정은 같은 구간(군집) 안에서만 lag를 만든다. 검정력은 시뮬레이션이다: 제한 모형(y의 자기회귀) 적합값 + 잔차 부트스트랩에
x_{t-1}이 1 표준편차 변할 때 y가 y의 표준편차의 b배 변하는 효과를 더해 F검정이 5%에서 기각되는 비율을 센다.
shortcut: y의 lag 회귀변수는 실제값으로 고정한다(조건부 검정력). 정확한 무조건 검정력이 필요하면 y를 순차 생성한다.
결과: results_power_regimes/power_mde.csv, regime_granger.csv
"""
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from run_time_ordered_checks import DATA, lstm_inputs, series

OUT = DATA.parent / "analysis" / "results_power_regimes"
EVENT_WINDOWS = {  # 이름: (시작, 끝)
    "1997_외환위기": ("1997-11-01", "1998-06-30"),
    "2008_금융위기": ("2008-09-01", "2009-03-31"),
    "2011_유럽재정위기": ("2011-08-01", "2011-10-31"),
    "2013_테이퍼탠트럼": ("2013-05-01", "2013-09-30"),
    "2015_위안화쇼크": ("2015-08-01", "2016-02-29"),
    "2020_팬데믹": ("2020-02-01", "2020-04-30"),
    "2022_연준긴축_킹달러": ("2022-03-01", "2022-10-31"),
    "2024_최근고환율": ("2024-11-01", "2025-12-31"),
}
BETAS = (0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5)
NSIM = 400


def lag_rows(d, p):
    """같은 군집 안에서 t, t-1, ..., t-p가 모두 있는 행만 남긴다."""
    g = d.groupby("grp")
    out = d[["y"]].copy()
    for k in range(1, p + 1):
        out[f"y{k}"] = g["y"].shift(k)
        out[f"x{k}"] = g["x"].shift(k)
    return out.dropna()


def f_test(f, p):
    ycols, xcols = [f"y{k}" for k in range(1, p + 1)], [f"x{k}" for k in range(1, p + 1)]
    y = f.y.to_numpy()
    Xr = np.column_stack([np.ones(len(f)), f[ycols]])
    Xu = np.column_stack([Xr, f[xcols]])
    rss = lambda X: ((y - X @ np.linalg.lstsq(X, y, rcond=None)[0]) ** 2).sum()
    rr, ru = rss(Xr), rss(Xu)
    df2 = len(f) - Xu.shape[1]
    if df2 <= 0 or ru <= 0:
        return np.nan, np.nan
    F = ((rr - ru) / p) / (ru / df2)
    return F, float(stats.f.sf(F, p, df2))


def power(f, p, rng):
    ycols = [f"y{k}" for k in range(1, p + 1)]
    Xr = np.column_stack([np.ones(len(f)), f[ycols]])
    y = f.y.to_numpy()
    coef = np.linalg.lstsq(Xr, y, rcond=None)[0]
    fit, resid = Xr @ coef, y - Xr @ coef
    scale = y.std() / f.x1.std()  # x 1 SD당 y SD의 b배
    res = {}
    for b in BETAS:
        hits = 0
        for _ in range(NSIM):
            sim = f.copy()
            sim["y"] = fit + rng.choice(resid, len(resid)) + b * scale * f.x1.to_numpy()
            hits += f_test(sim, p)[1] < 0.05
        res[b] = hits / NSIM
    return res


def mde(res):
    ok = [b for b, v in res.items() if v >= 0.8]
    return min(ok) if ok else np.nan  # nan: 격자 최대값 b=0.5에서도 80% 미만


def monthly_frame():
    fx = series("exchange_rate/USD_KRW_processed.csv", "USD_KRW").resample("ME").mean()
    m2 = series("m2/KOR/M2_KOR_processed.csv", "M2_KOR").resample("ME").last()
    m = pd.concat({"fx": fx.iloc[:-1], "m2": m2}, axis=1).dropna()
    ch = np.log(m).diff().dropna()
    vix = pd.read_csv(DATA / "integrated_macro_targets.csv", parse_dates=["Date"])
    vix = vix.assign(Date=vix.Date + pd.offsets.MonthEnd(0)).set_index("Date").VIX.reindex(ch.index)
    return ch.rename(columns={"fx": "y", "m2": "x"}), vix


def groups_by_windows(index, windows):
    g = pd.Series(np.nan, index=index)
    for i, (a, b) in enumerate(windows.values()):
        g[(index >= a) & (index <= b)] = i
    return g


def contiguous_runs(mask):
    return mask.ne(mask.shift()).cumsum().where(mask)


def run_set(name, d, delay_list, directions, rng, rows, power_rows, with_power=True):
    for delay in delay_list:
        for dname, (yc, xc) in directions.items():
            base = pd.DataFrame({"y": d[yc], "x": d[xc].groupby(d.grp).shift(delay), "grp": d.grp}).dropna()
            for p in (1, 2, 3):
                f = lag_rows(base, p)
                F, pv = f_test(f, p) if len(f) > 3 * p + 3 else (np.nan, np.nan)
                rows.append(dict(set=name, direction=dname, x_delay=delay, lag=p, n=len(f),
                                 clusters=base.grp.nunique(), F=F, p=pv))
            if with_power and delay == delay_list[0] and dname.endswith("-> FX"):
                for p in (1, 3):
                    f = lag_rows(base, p)
                    if len(f) > 3 * p + 10:
                        res = power(f, p, rng)
                        power_rows.append(dict(set=name, lag=p, n=len(f), mde_sd=mde(res),
                                               **{f"power_b{b}": v for b, v in res.items()}))


def main():
    rng = np.random.default_rng(42)
    OUT.mkdir(exist_ok=True)
    rows, prow = [], []
    # 월별 M2 -> 환율
    ch, vix = monthly_frame()
    ch["fx"], ch["m2"] = ch.y, ch.x
    sets = {"월별 전체": pd.Series(0.0, index=ch.index),
            "월별 사전지정 위기구간": groups_by_windows(ch.index, EVENT_WINDOWS),
            "월별 VIX>=30": contiguous_runs(vix.ge(30))}
    for name, g in sets.items():
        d = ch.assign(grp=g).dropna(subset=["grp"])
        run_set(name, d, (0, 1), {"M2 -> FX": ("fx", "m2"), "FX -> M2": ("m2", "fx")}, rng, rows, prow)
    # 일별 MMF -> 환율. 짧은 계열(기존 파일, 2010-12~)과 freesis에서 받은 긴 계열(2006-04~)을 모두 검정한다.
    fx = series("exchange_rate/USD_KRW_processed.csv", "USD_KRW")
    short = lstm_inputs(fx).mmf
    long_ = pd.read_csv(DATA / "m2/KOR/MMF/MMF_daily_2006_2026.csv", skiprows=4, header=None, usecols=[0, 1],
                        names=["date", "mmf"], thousands=",")
    long_ = long_.assign(date=pd.to_datetime(long_.date, format="%Y/%m/%d")).set_index("date").mmf.sort_index() / 10  # 억원 -> 십억원
    for label, mmf in (("2010-12~", short), ("2006-04~", long_)):
        dd = pd.DataFrame({"fx": np.log(fx), "mmf": np.log(mmf)}).dropna().diff().dropna()
        dd.index.name = "date"
        dsets = {f"일별 MMF({label}) 전체": pd.Series(0.0, index=dd.index),
                 f"일별 MMF({label}) 사전지정 위기구간": groups_by_windows(dd.index, EVENT_WINDOWS),
                 f"일별 MMF({label}) 최근 2024-11~2025-12": groups_by_windows(dd.index, {"r": EVENT_WINDOWS["2024_최근고환율"]})}
        if label == "2006-04~":
            crisis = {k: v for k, v in EVENT_WINDOWS.items() if k < "2015"}  # 2015 이전 구간만(2015 이후는 위 짧은 계열과 겹침)
            dsets[f"일별 MMF({label}) 2008 금융위기"] = groups_by_windows(dd.index, {"c": EVENT_WINDOWS["2008_금융위기"]})
            dsets[f"일별 MMF({label}) 2015 이전 위기구간"] = groups_by_windows(dd.index, crisis)
        for name, g in dsets.items():
            d = dd.assign(grp=g).dropna(subset=["grp"])
            run_set(name, d, (0,), {"MMF -> FX": ("fx", "mmf"), "FX -> MMF": ("mmf", "fx")}, rng, rows, prow)
    res = pd.DataFrame(rows)
    # 선행성 가설 방향(유동성 -> 환율) 검정 전체를 한 군으로 Holm 보정한다.
    fwd = res.direction.str.endswith("-> FX") & res.p.notna()
    res["p_holm"] = np.nan
    res.loc[fwd, "p_holm"] = multipletests(res.loc[fwd, "p"], method="holm")[1]
    res.to_csv(OUT / "regime_granger.csv", index=False, float_format="%.4g")
    pw = pd.DataFrame(prow)
    pw.to_csv(OUT / "power_mde.csv", index=False, float_format="%.4g")
    pd.set_option("display.width", 220)
    print(res[res.direction.str.endswith("-> FX") | res.direction.str.startswith("MMF")].to_string(index=False))
    print(pw[["set", "lag", "n", "mde_sd", "power_b0.05", "power_b0.1", "power_b0.2", "power_b0.5"]].to_string(index=False))


if __name__ == "__main__":
    main()
