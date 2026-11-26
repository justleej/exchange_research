"""임계점 회귀를 일별 MMF 2006-04 이후 자료로 확장한다(기존은 CMA+MMF 프록시 2015-12 이후 2,531일).

일별 MMF(금융투자협회 통계 포털, 2006-04~)는 CMA가 없어 프록시가 아니라 MMF 단독 변화량을 쓴다. 같은 TAR 탐색과
부트스트랩 sup-F 검정(analysis/daily_threshold_MMF/rerun_redefined.py의 sup_f)을 쓴다. 표본 세 가지를 비교한다:
2015-12 이후(기존 프록시와 비교), 2006-04 이후 전체, 2006-04~2015-11(이전 구간).
임계값 초과 빈도는 고정기간, 재정의 A/B/C, 사전지정 위기 구간 안과 밖을 Fisher 검정으로 비교한다.
사용: python analysis/run_threshold_mmf_extended.py [부트스트랩 횟수, 기본 1000] [pct]
"pct"를 주면 절대 변화량(십억원, 원) 대신 로그 변화율(%)을 쓴다. MMF 잔액이 2006년 약 70조원에서 2026년 약 280조원으로 커져
절대 임계값은 최근 기간에 기계적으로 더 자주 넘어가므로 pct가 규모 효과를 제거한 사양이다.
결과: analysis/results_power_regimes/threshold_mmf_extended[_pct].csv, threshold_mmf_breach[_pct].csv
"""
import sys

import numpy as np
import pandas as pd
from scipy import stats

from daily_threshold_MMF import rerun_redefined as thr  # noqa: E402  (BOOT=sys.argv[1], sup_f, rng)
from rerun_redefined import DATA as DAILY, DEFS, WINDOW  # noqa: E402  (analysis/anomaly)
from run_power_and_event_regimes import EVENT_WINDOWS
from run_time_ordered_checks import DATA, series

OUT = DATA.parent / "analysis" / "results_power_regimes"
BOOT = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
FIXED = ("2024-11-01", "2025-12-31")
PCT = len(sys.argv) > 2 and sys.argv[2] == "pct"
SUFFIX = "_pct" if PCT else ""


def load():
    fx = series("exchange_rate/USD_KRW_processed.csv", "USD_KRW")
    mmf = pd.read_csv(DATA / "m2/KOR/MMF/MMF_daily_2006_2026.csv", skiprows=4, header=None, usecols=[0, 1],
                      names=["date", "mmf"], thousands=",")
    mmf = mmf.assign(date=pd.to_datetime(mmf.date, format="%Y/%m/%d")).set_index("date").mmf.sort_index() / 10  # 억원 -> 십억원
    d = pd.concat({"fx": fx, "mmf": mmf}, axis=1).dropna()
    if PCT:
        d = np.log(d) * 100  # 로그 변화율(%)을 쓰기 위해 로그 수준으로 바꾼 뒤 차분한다
    d = pd.DataFrame({"date": d.index[1:], "delta_ER": d.fx.diff().dropna().to_numpy(),
                      "delta_M2": d.mmf.diff().dropna().to_numpy()})
    return d.reset_index(drop=True)


def tar_test(d, rng):
    dm, y = d.delta_M2.to_numpy(), d.delta_ER.to_numpy()
    grid = np.percentile(dm[dm > 0], np.linspace(10, 95, 85))
    X0 = np.column_stack([np.ones_like(dm), dm])
    b0 = np.linalg.lstsq(X0, y, rcond=None)[0]
    rss0 = float(((y - X0 @ b0) ** 2).sum())
    f_obs, t_best = thr.sup_f(dm, y, grid, rss0)
    fit0, e0 = X0 @ b0, y - X0 @ b0
    cnt = 0
    for _ in range(BOOT):
        ys = fit0 + e0 * rng.standard_normal(len(y))
        bs = np.linalg.lstsq(X0, ys, rcond=None)[0]
        r0 = float(((ys - X0 @ bs) ** 2).sum())
        cnt += thr.sup_f(dm, ys, grid, r0)[0] >= f_obs
    return f_obs, t_best, (cnt + 1) / (BOOT + 1)


def breach_table(d, t, sets, label):
    rows = []
    breach = d.delta_M2.to_numpy() > t
    for name, dates in sets.items():
        inn = d.date.isin(dates).to_numpy()
        a, b = breach[inn], breach[~inn]
        if len(a) == 0 or len(b) == 0:
            continue
        tab = [[int(a.sum()), int((~a).sum())], [int(b.sum()), int((~b).sum())]]
        rows.append(dict(sample=label, regime=name, n_in=len(a), breach_in_pct=a.mean() * 100, n_out=len(b),
                         breach_out_pct=b.mean() * 100, ratio=a.mean() / b.mean() if b.mean() else np.nan,
                         fisher_p=stats.fisher_exact(tab)[1]))
    return rows


def main():
    rng = np.random.default_rng(42)
    d_all = load()
    daily = pd.read_csv(DAILY, parse_dates=["date"]).dropna(subset=["FX_rate", "policy_spread"]).sort_values("date").reset_index(drop=True)
    daily["rc"] = daily.FX_rate.rolling(WINDOW).corr(daily.policy_spread)
    daily = daily.dropna(subset=["rc"])
    sets = {"fixed(2024-11~2025-12)": set(daily.loc[daily.date.between(*FIXED), "date"])}
    for name, cond in DEFS.items():
        sets[name] = set(daily.loc[cond(daily.rc), "date"])
    sets["사전지정 위기 구간"] = set(d_all.date[np.logical_or.reduce([d_all.date.between(a, b) for a, b in EVENT_WINDOWS.values()])])
    samples = {"MMF 2015-12~": d_all[d_all.date >= "2015-12-01"], "MMF 2006-04~": d_all,
               "MMF 2006-04~2015-11": d_all[d_all.date < "2015-12-01"]}
    tests, breaches = [], []
    for label, d in samples.items():
        d = d.reset_index(drop=True)
        f, t, p = tar_test(d, rng)
        tests.append(dict(sample=label, rows=len(d), start=d.date.min().date(), end=d.date.max().date(),
                          threshold=t, unit="%" if PCT else "십억원", sup_F=f, bootstrap_p=p, B=BOOT))
        print(tests[-1], flush=True)
        breaches += breach_table(d, t, sets, label)
    OUT.mkdir(exist_ok=True)
    pd.DataFrame(tests).to_csv(OUT / f"threshold_mmf_extended{SUFFIX}.csv", index=False, float_format="%.4g")
    b = pd.DataFrame(breaches)
    b.to_csv(OUT / f"threshold_mmf_breach{SUFFIX}.csv", index=False, float_format="%.4g")
    pd.set_option("display.width", 220)
    print(b.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
