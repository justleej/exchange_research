"""run_monthly_horizon.py에서 Holm 보정 후에도 남은 결과(수시입출식저축성예금 성장률 -> 6개월 뒤 환율, CW p=0.002)의 강건성 점검.

변형: 기준, 2개월 지연, 3개월 성장률, 글로벌 통제(VIX 변화, DXY 로그 변화), 2015년 이후만 평가, 2008~2014년만 평가.
같은 표본, 같은 확장 창 방식을 쓴다. 결과: results_power_regimes/monthly_robustness.csv
"""
import numpy as np
import pandas as pd

from run_monthly_horizon import LIQ, OUT, hac_fit, make_xy, monthly_base, oos
from run_time_ordered_checks import DATA, series

NAME = "수시입출식저축성예금"


def liquidity_x(df, lag, growth_months):
    path, col = LIQ[NAME]
    s = series(path, col)
    s.index = s.index + pd.offsets.MonthEnd(0)
    g = np.log(s).diff(growth_months).reindex(df.index).shift(lag)
    return (g - g.mean()) / g.std()


def main():
    base = monthly_base()
    macro = pd.read_csv(DATA / "integrated_macro_targets.csv", parse_dates=["Date"])
    macro["Date"] = macro.Date + pd.offsets.MonthEnd(0)
    macro = macro.set_index("Date")
    base["dvix"] = macro.VIX.diff().reindex(base.index)
    base["ddxy"] = np.log(macro.DXY).diff().reindex(base.index)
    rows = []
    variants = [("기준(1개월 지연, 1개월 성장률)", 1, 1, ("r", "ds"), None, None),
                ("2개월 지연", 2, 1, ("r", "ds"), None, None),
                ("3개월 성장률", 1, 3, ("r", "ds"), None, None),
                ("글로벌 통제(VIX, DXY)", 1, 1, ("r", "ds", "dvix", "ddxy"), None, None),
                ("평가 2015-01 이후", 1, 1, ("r", "ds"), "2015-01-31", None),
                ("평가 2008~2014", 1, 1, ("r", "ds"), None, "2014-12-31")]
    for label, lag, gm, ctrl, eval_from, eval_to in variants:
        x = liquidity_x(base, lag, gm)
        for h in (6, 12):
            d = make_xy(base, x, h).dropna(subset=list(ctrl))
            dd = d.dropna(subset=["y"])
            fit = hac_fit(dd.y, dd[list(ctrl) + ["x"]], h)
            o = oos(d if eval_to is None else d[d.index <= eval_to], h, ctrl=ctrl, eval_from=eval_from)
            rows.append(dict(variant=label, horizon=h, n_is=len(dd), coef_per_sd=fit.params["x"], is_p=fit.pvalues["x"],
                             **{k: o[k] for k in ("n_oos", "ratio_B_A", "cw_t", "cw_p")}))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "monthly_robustness.csv", index=False, float_format="%.4g")
    pd.set_option("display.width", 220)
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
