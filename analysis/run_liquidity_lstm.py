"""Compare liquidity features on the same chronological FX test observations."""
import numpy as np
import pandas as pd

from run_time_ordered_checks import OUT, fit_lstm, lstm_inputs, series

SEEDS = (42, 43, 44)
MODELS = {
    "baseline": ["fx", "spread"],
    "plus_mmf": ["fx", "spread", "mmf"],
    "plus_deposit": ["fx", "spread", "deposit"],
}


def paired_interval(error_a, error_b, block_size=20, draws=2000):
    """Circular block bootstrap for the paired RMSE difference on fixed predictions."""
    rng = np.random.default_rng(42)
    n = len(error_a)
    starts = rng.integers(n, size=(draws, (n + block_size - 1) // block_size))
    indices = ((starts[..., None] + np.arange(block_size)) % n).reshape(draws, -1)[:, :n]
    diff = np.sqrt(np.mean(error_a[indices] ** 2, axis=1)) - np.sqrt(np.mean(error_b[indices] ** 2, axis=1))
    return np.quantile(diff, [0.025, 0.975])


def main():
    fx = series("exchange_rate/USD_KRW_processed.csv", "USD_KRW")
    frame = lstm_inputs(fx)
    deposit = series("m2/KOR/M2_details_processed.csv", "M2_수시입출식저축성예금")
    # Assume availability after the end of the following month.
    deposit.index = deposit.index + pd.offsets.MonthEnd(0) + pd.offsets.MonthEnd(1)
    frame["deposit"] = pd.merge_asof(
        frame[[]], deposit.rename("deposit").to_frame(), left_index=True, right_index=True,
        allow_exact_matches=False, tolerance=pd.Timedelta("45D"),
    )["deposit"]
    length, horizon = 30, 5
    cutoff = int(len(frame) * 0.8)
    origins = np.arange(length - 1, len(frame) - horizon)
    complete = frame.notna().all(axis=1).rolling(length).sum().eq(length).to_numpy()
    origins = origins[complete[origins]]
    train = origins[origins + horizon < cutoff]
    test = origins[origins >= cutoff]
    if not len(train) or not len(test):
        raise ValueError("No common train or test windows")
    dates = frame.index
    out = pd.DataFrame({"origin": dates[test], "target_date": dates[test + horizon],
                        "actual": frame.fx.iloc[test + horizon].to_numpy(),
                        "last_fx": frame.fx.iloc[test].to_numpy()})
    metrics, intervals = [], []
    actual = out.actual.to_numpy()
    for seed in SEEDS:
        errors = {"last_fx": out.last_fx.to_numpy() - actual}
        for name, cols in MODELS.items():
            prediction = fit_lstm(frame, cols, train, test, cutoff, length, horizon, seed)
            out[f"{name}_{seed}"] = prediction
            errors[name] = prediction - actual
            print(f"Finished {name}, seed {seed}", flush=True)
        for name, error in errors.items():
            metrics.append(dict(seed=seed, model=name, n_train=len(train), n_test=len(test),
                                cutoff=dates[cutoff], rmse=np.sqrt(np.mean(error ** 2)),
                                mae=np.mean(np.abs(error))))
        for model in ("plus_mmf", "plus_deposit"):
            for reference in ("baseline", "last_fx"):
                low, high = paired_interval(errors[model], errors[reference])
                intervals.append(dict(seed=seed, model=model, reference=reference,
                                      rmse_delta=np.sqrt(np.mean(errors[model] ** 2))
                                      - np.sqrt(np.mean(errors[reference] ** 2)),
                                      ci_low=low, ci_high=high, block_size=20))
    OUT.mkdir(exist_ok=True)
    out.to_csv(OUT / "liquidity_lstm_predictions.csv", index=False)
    result = pd.DataFrame(metrics)
    result.to_csv(OUT / "liquidity_lstm_metrics.csv", index=False)
    pd.DataFrame(intervals).to_csv(OUT / "liquidity_lstm_paired_intervals.csv", index=False)
    print(result.groupby("model").rmse.agg(["mean", "std"]).to_string(), flush=True)


if __name__ == "__main__":
    main()
