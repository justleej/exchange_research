"""교차통화 복제(H3a): 수시입출식저축성예금 성장률이 원/엔(100엔), 원/유로의 6, 12개월 뒤 변화도 예측하는가. 계획: docs/analysis_plan_extra.md
같은 월별 설계(1개월 지연, 1999-06~2026-01, 모형 A=환율 월 변화+금리차 변화, 확장 창 2008-01~)에서 종속변수만 바꾼다. 금리차는 한미 금리차를 그대로 쓴다.
자료는 data/process_scripts/fetch_fx_cross_ecos.py가 받은 ECOS_cross_rates_daily.csv. 결과: results_power_regimes/signal_cross_currency.csv
"""
import numpy as np
import pandas as pd
from scipy import stats

from run_monthly_horizon import LIQ, hac_fit, make_xy, monthly_base
from run_monthly_signal_check import fast_oos, liquidity_x, nw_t
from run_time_ordered_checks import DATA, series

OUT = DATA.parent / "analysis" / "results_power_regimes"
NAME = "수시입출식저축성예금"


def main():
    usd = monthly_base()
    cross = pd.read_csv(DATA / "exchange_rate/ECOS_cross_rates_daily.csv", parse_dates=["observation_date"]).set_index("observation_date")
    sp = series("policy_rate/spread_KOR_USA_processed.csv", "RATE_SPREAD_KOR_USA").resample("ME").last()
    bases = {"USD(기준)": usd}
    for col in ("JPY100", "EUR"):
        df = pd.DataFrame({"fx": np.log(cross[col].dropna().resample("ME").last()), "spread": sp}).dropna().loc["1999-06-30":"2026-01-31"]
        df["r"], df["ds"] = df.fx.diff(), df.spread.diff()
        bases[col] = df
    liq = series(*LIQ[NAME])
    rows = []
    for cur, df in bases.items():
        x = liquidity_x(df, liq)
        for h in (6, 12):
            d = make_xy(df, x, h)
            dd = d.dropna(subset=["y"])
            fit = hac_fit(dd.y, dd[["r", "ds", "x"]], h)
            f = fast_oos(d, h)
            t = nw_t(f, h)
            rows.append(dict(currency=cur, horizon=h, n_is=len(dd), coef_per_sd=fit.params["x"], is_p=fit.pvalues["x"],
                             n_oos=len(f), cw_t=t, cw_p=float(stats.norm.sf(t))))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "signal_cross_currency.csv", index=False, float_format="%.4g")
    pd.set_option("display.width", 200)
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
