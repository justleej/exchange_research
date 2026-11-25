from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.stats.multitest import multipletests
import torch
from torch.utils.data import DataLoader, TensorDataset

from LSTM.lstm_mmf.train_eval_extended import ExRateLSTM, train_model

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data"
OUT = ROOT / "results_time_ordered"


def series(path, column):
    return pd.read_csv(DATA / path, parse_dates=["observation_date"]).set_index(
        "observation_date"
    )[column].sort_index().dropna()


def monthly_granger(fx):
    monthly_fx = fx.resample("ME").mean()
    if fx.index.max() < monthly_fx.index.max():
        monthly_fx = monthly_fx.iloc[:-1]
    m2 = series("m2/KOR/M2_KOR_processed.csv", "M2_KOR").resample("ME").last()
    monthly = pd.concat({"fx": monthly_fx, "m2": m2}, axis=1)
    changes = np.log(monthly).diff().dropna()
    assert np.all(np.diff(changes.index.to_period("M").asi8) == 1)
    rows = []
    for delay in (0, 1, 2):
        sample = changes.assign(m2=changes.m2.shift(delay)).dropna()
        for target, predictor in (("fx", "m2"), ("m2", "fx")):
            tests = grangercausalitytests(sample[[target, predictor]], [1, 2, 3])
            for lag, result in tests.items():
                f_stat, p, df_resid, df_num = result[0]["ssr_ftest"]
                rows.append(dict(target=target, predictor=predictor, m2_delay=delay,
                                 lag=lag, f=f_stat, p=p, n=int(df_resid + 2 * lag + 1),
                                 start=sample.index[lag], end=sample.index[-1]))
    results = pd.DataFrame(rows)
    results["p_holm"] = multipletests(results.p, method="holm")[1]
    results.to_csv(OUT / "monthly_granger.csv", index=False)
    print(results.to_string(index=False), flush=True)


def lstm_inputs(fx):
    spread = series("policy_rate/spread_KOR_USA_processed.csv", "RATE_SPREAD_KOR_USA")
    mmf = pd.read_csv(DATA / "m2/KOR/MMF/MMF_daily.csv", skiprows=5,
                      header=None, usecols=[0, 1], thousands=",", names=["date", "mmf"])
    mmf.index = pd.to_datetime(mmf.pop("date"), errors="coerce")
    mmf["mmf"] = pd.to_numeric(mmf.mmf, errors="coerce")
    mmf = mmf.loc[mmf.index.notna()].dropna().sort_index()
    assert fx.index.is_unique and spread.index.is_unique and mmf.index.is_unique
    frame = fx.to_frame("fx")
    for name, values, tolerance in (("spread", spread, "45D"), ("mmf", mmf.mmf, "7D")):
        frame[name] = pd.merge_asof(
            frame[[]], values.rename(name).to_frame(), left_index=True, right_index=True,
            allow_exact_matches=False, tolerance=pd.Timedelta(tolerance)
        )[name]
    return frame.loc[mmf.index.min():mmf.index.max()]


def lstm(fx):
    frame = lstm_inputs(fx)
    length, horizon = 30, 5
    cutoff = int(len(frame) * 0.8)
    origins = np.arange(length - 1, len(frame) - horizon)
    complete = frame.notna().all(axis=1).rolling(length).sum().eq(length).to_numpy()
    origins = origins[complete[origins]]
    train = origins[origins + horizon < cutoff]
    test = origins[origins >= cutoff]
    assert len(train) and len(test)
    dates = frame.index
    assert dates[train[-1] + horizon] < dates[test[0]]
    out = pd.DataFrame({"origin": dates[test], "target_date": dates[test + horizon],
                        "actual": frame.fx.iloc[test + horizon].to_numpy(),
                        "last_fx": frame.fx.iloc[test].to_numpy()})
    for name, columns in (("baseline", ["fx", "spread"]), ("plus_mmf", ["fx", "spread", "mmf"])):
        torch.manual_seed(42)
        scaler = StandardScaler().fit(frame[columns].iloc[:cutoff].dropna())
        z = scaler.transform(frame[columns])

        def windows(indices):
            return torch.tensor(np.stack([z[i - length + 1:i + 1] for i in indices]),
                                dtype=torch.float32)

        y = torch.tensor(z[train + horizon, :1], dtype=torch.float32)
        model = ExRateLSTM(len(columns), 32, 1)
        loader = DataLoader(TensorDataset(windows(train), y), batch_size=32, shuffle=True)
        train_model(model, loader, torch.nn.MSELoss(), torch.optim.Adam(model.parameters(), lr=0.005), 50)
        model.eval()
        with torch.no_grad():
            out[name] = model(windows(test)).numpy().ravel() * scaler.scale_[0] + scaler.mean_[0]
    save_lstm_results(out, frame.spread, len(train), dates[cutoff])


def save_lstm_results(out, spread, n_train, cutoff):
    fx = series("exchange_rate/USD_KRW_processed.csv", "USD_KRW").reindex(spread.index)
    corr = fx.rolling(30).corr(spread).reindex(pd.to_datetime(out.origin)).to_numpy()
    out["weak_corr_at_origin"] = np.isfinite(corr) & (np.abs(corr) < 0.2)
    out.to_csv(OUT / "mmf_predictions_seed42.csv", index=False)
    rows = []
    for subset, sample in (("all", out), ("weak_corr_at_origin", out[out.weak_corr_at_origin])):
        for name in ("baseline", "plus_mmf", "last_fx"):
            error = sample[name] - sample.actual
            rows.append(dict(subset=subset, model=name, seed=42, n_train=n_train,
                             n_test=len(sample), cutoff=cutoff,
                             rmse=np.sqrt(np.mean(error ** 2)), mae=error.abs().mean()))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / "mmf_metrics_seed42.csv", index=False)
    print(metrics.to_string(index=False), flush=True)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    fx = series("exchange_rate/USD_KRW_processed.csv", "USD_KRW")
    monthly_granger(fx)
    lstm(fx)
