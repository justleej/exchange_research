"""Recheck historical threshold samples and monthly event definitions."""
from io import BytesIO
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from statsmodels.api import OLS, add_constant

from daily_threshold_MMF.rerun_redefined import BOOT, sup_f
from fx_impact import run_final_fx_impact_pipeline as lp
from rerun_redefined import DATA, DEFS, WINDOW

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/results_historical_rechecks"


def threshold_checks():
    rows = []
    for ref in ("8ec5ccb", "e381d3c"):
        raw = subprocess.check_output(
            ["git", "show", f"{ref}:data/m2/KOR/merged_daily_liquid.csv"], cwd=ROOT
        )
        d = pd.read_csv(BytesIO(raw), parse_dates=["observation_date"]).sort_values("observation_date")
        changes = d[["M2_proxy", "USD_KRW"]].diff().dropna()
        dm, y = changes.to_numpy().T
        grid = np.percentile(dm[dm > 0], np.linspace(10, 95, 85))
        x = np.column_stack([np.ones_like(dm), dm])
        fit = x @ np.linalg.lstsq(x, y, rcond=None)[0]
        error = y - fit
        f, threshold = sup_f(dm, y, grid, error @ error)
        rng = np.random.default_rng(42)
        exceedances = 0
        for _ in range(BOOT):
            draw = fit + error * rng.standard_normal(len(y))
            residual = draw - x @ np.linalg.lstsq(x, draw, rcond=None)[0]
            exceedances += sup_f(dm, draw, grid, residual @ residual)[0] >= f
        dates = d.loc[changes.index, "observation_date"]
        recent = dates.between("2024-11-01", "2025-12-31").to_numpy()
        breach = dm > threshold
        rows.append(dict(ref=ref, rows=len(y), start=dates.min().date(), end=dates.max().date(),
                         threshold_bn_krw=threshold, sup_f=f, bootstrap_p=(1 + exceedances)/(1 + BOOT),
                         bootstrap_draws=BOOT, fixed_rows=int(recent.sum()),
                         fixed_rate=breach[recent].mean(), outside_rate=breach[~recent].mean()))
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "historical_threshold.csv", index=False)
    print(result.to_string(index=False), flush=True)


def event_panels(targets=None):
    macro = lp.load_macro_dataset()
    reports = lp.REPORT_DIR
    ranking = pd.read_csv(reports / "target_selection/target_ranking.csv")
    targets = targets or ranking.loc[ranking.selected_for_final_model, "target"].tolist()
    predictions = pd.read_csv(reports / "fx_model_selection/selected_fx_predictions.csv", parse_dates=["date"])
    source = predictions.source_model.iloc[0]
    daily = pd.read_csv(DATA, parse_dates=["date"]).sort_values("date")
    corr = daily.FX_rate.rolling(WINDOW).corr(daily.policy_spread)
    valid = np.isfinite(corr)
    months = daily.date.dt.to_period("M")
    masks = {"legacy": macro.Is_Abnormal_Period.eq(1)}
    for name, rule in DEFS.items():
        fraction = rule(corr[valid]).groupby(months[valid]).mean()
        masks[name] = macro.Date.dt.to_period("M").map(fraction).ge(0.5)
    for name, mask in masks.items():
        panel = lp.build_event_time_panel(
            macro.assign(Is_Abnormal_Period=mask.astype(int)), targets, predictions, source, 0.05
        )
        yield name, mask, panel


def event_checks():
    rows, counts = [], []
    for name, mask, panel in event_panels():
        counts.append(dict(definition=name, flagged_months=int(mask.sum()),
                           panel_months=panel.event_date.nunique() if not panel.empty else 0,
                           panel_rows=len(panel)))
        if panel.empty:
            continue
        for (target, horizon), group in panel.groupby(["target", "horizon"]):
            train, test = lp.split_lp_train_test(group)
            if train.empty or test.empty:
                continue
            x = add_constant(lp.build_lp_features(train, "actual"), has_constant="add")
            fit = OLS(train.actual_response, x, missing="raise").fit(
                cov_type="HAC", cov_kwds={"maxlags": max(1, min(int(horizon), len(train)//4))}
            )
            term = "fx_shock_t"
            lo, hi = fit.conf_int().loc[term]
            rows.append(dict(definition=name, target=target, horizon=horizon, train_rows=len(train),
                             test_rows=len(test), coefficient=fit.params[term], p=fit.pvalues[term],
                             ci_low=lo, ci_high=hi))
    coefficients = pd.DataFrame(rows)
    coefficients.to_csv(OUT / "event_ols_coefficients.csv", index=False)
    counts = pd.DataFrame(counts)
    counts.to_csv(OUT / "event_sample_counts.csv", index=False)
    print(counts.to_string(index=False), flush=True)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    threshold_checks()
    event_checks()
