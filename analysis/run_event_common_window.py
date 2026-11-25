"""Compare event-time OLS on a common calendar and adjust all shock tests."""
import numpy as np
import pandas as pd
from statsmodels.api import OLS, add_constant
from statsmodels.stats.multitest import multipletests

from fx_impact import run_final_fx_impact_pipeline as lp
from run_historical_rechecks import OUT, event_panels


def main():
    panels = {name: panel for name, _, panel in event_panels()}
    start = max(panel.event_date.min() for panel in panels.values())
    end = min(panel.event_date.max() for panel in panels.values())
    calendar = pd.date_range(start, end, freq="ME")
    cutoff = calendar[int(len(calendar) * (1 - lp.LP_TEST_FRACTION))]
    rows = []
    for name, panel in panels.items():
        panel = panel.loc[panel.event_date.between(start, end)]
        for (target, horizon), group in panel.groupby(["target", "horizon"]):
            train = group.loc[group.response_date.lt(cutoff)].sort_values("event_date")
            test = group.loc[group.event_date.ge(cutoff)]
            x = add_constant(lp.build_lp_features(train, "actual"), has_constant="add")
            row = dict(definition=name, target=target, horizon=horizon,
                       window_start=start.date(), window_end=end.date(), cutoff=cutoff.date(),
                       train_rows=len(train), test_rows=len(test),
                       coefficient=np.nan, ci_low=np.nan, ci_high=np.nan, p=np.nan)
            if len(train) >= max(len(lp.lp_feature_columns()) + 8, 20) and np.linalg.matrix_rank(x) == x.shape[1]:
                fit = OLS(train.actual_response, x, missing="raise").fit(
                    cov_type="HAC", cov_kwds={"maxlags": max(1, min(int(horizon), len(train) // 4))}
                )
                term = "fx_shock_t"
                row.update(coefficient=fit.params[term], p=fit.pvalues[term])
                row["ci_low"], row["ci_high"] = fit.conf_int().loc[term]
            rows.append(row)
    result = pd.DataFrame(rows)
    valid = result.p.notna()
    # Keep the entire planned family, including cells without an estimate.
    result["p_holm"] = multipletests(result.p.fillna(1), method="holm")[1]
    result.loc[~valid, "p_holm"] = np.nan
    OUT.mkdir(exist_ok=True)
    result.to_csv(OUT / "event_common_window.csv", index=False)
    print(f"{start.date()} to {end.date()}, cutoff {cutoff.date()}")
    print(result.groupby("definition").p.agg(tested="count", nominal=lambda p: (p < .05).sum()))
    print(result.loc[result.p_holm.lt(.05)].to_string(index=False))
    print(result.loc[result.target.eq("CSI_CCSI") & result.horizon.eq(1)].to_string(index=False))


if __name__ == "__main__":
    main()
