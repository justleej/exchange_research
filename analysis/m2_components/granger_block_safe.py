"""블록 경계를 넘지 않는 Granger 재검정.
기존 코드는 변화율만 블록 안에서 계산하고, lag 회귀변수는 이어붙인 행에서 만들어 서로 다른
블록의 값이 섞였다. 여기서는 t, t-1, ..., t-p 가 모두 같은 블록 안인 행만 쓴다."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "analysis" / "anomaly" / "anomaly_concatenated_dataset.csv"
OUT = Path(__file__).resolve().parent / "granger_block_safe.csv"


def lag_frame(d, cols, p):
    g = d.groupby("block_start")
    out = d[["block_start"]].copy()
    for c in cols:
        out[c] = d[c]
        for k in range(1, p + 1):
            out[f"{c}_l{k}"] = g[c].shift(k)
    return out.dropna()


def _rss(y, X):
    X = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ beta
    return float(r @ r), X.shape[1]


def granger(d, y, x, p):
    f = lag_frame(d, [y, x], p)
    ycols = [f"{y}_l{k}" for k in range(1, p + 1)]
    xcols = [f"{x}_l{k}" for k in range(1, p + 1)]
    yy = f[y].to_numpy()
    rss_r, _ = _rss(yy, f[ycols].to_numpy())
    rss_u, k_u = _rss(yy, f[ycols + xcols].to_numpy())
    n = len(f)
    F = ((rss_r - rss_u) / p) / (rss_u / (n - k_u))
    return F, float(stats.f.sf(F, p, n - k_u)), n


def main():
    df = pd.read_csv(DATA, parse_dates=["date", "block_start"]).sort_values(["block_start", "date"])
    g = df.groupby("block_start")
    df["FX"] = g["FX_rate"].pct_change()
    df["M2"] = g["M2_KOR"].pct_change()
    df = df.dropna(subset=["FX", "M2"])
    rows = []
    for p in (1, 2, 3):
        for name, y, x in (("M2 -> FX", "FX", "M2"), ("FX -> M2", "M2", "FX")):
            F, pv, n = granger(df, y, x, p)
            rows.append({"direction": name, "lag": p, "F": F, "p_value": pv, "n_obs": n})
    res = pd.DataFrame(rows)
    res.to_csv(OUT, index=False)
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
