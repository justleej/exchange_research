"""모형 의존성 점검: 같은 이상구간 자료·같은 80/20 시간분할에서 RF/XGBoost/Ridge의
SHAP 중요도 순위를 비교한다. 수준값(level)과 구간 내부 21일 로그변화(change) 두 가지로 돌린다.
기존 results_anomaly는 덮어쓰지 않고 results_model_compare/ 에 저장한다."""
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "analysis" / "anomaly" / "anomaly_concatenated_dataset.csv"
OUT = Path(__file__).resolve().parent / "results_model_compare"
FEATURES = ["policy_spread", "M2_KOR", "M2_USA"]
TARGET = "FX_rate"
LAG = 21  # 영업일 기준 약 1개월. 월별 M2를 보간한 값이라 일별 변화는 의미가 없다.


def make_level(df):
    return df[["date", TARGET] + FEATURES].dropna()


def make_change(df):
    # 블록 경계를 넘는 변화율이 생기지 않도록 블록 안에서만 차분한다.
    g = df.groupby("block_start")
    out = pd.DataFrame({"date": df["date"]})
    out[TARGET] = g[TARGET].transform(lambda s: np.log(s).diff(LAG))
    for c in ["M2_KOR", "M2_USA"]:
        out[c] = g[c].transform(lambda s: np.log(s).diff(LAG))
    out["policy_spread"] = g["policy_spread"].transform(lambda s: s.diff(LAG))
    return out.dropna()


def run(name, d):
    d = d.sort_values("date").reset_index(drop=True)
    k = int(len(d) * 0.8)
    tr, te = d.iloc[:k], d.iloc[k:]
    xtr, ytr, xte, yte = tr[FEATURES], tr[TARGET], te[FEATURES], te[TARGET]
    outside = {c: float(((xte[c] < xtr[c].min()) | (xte[c] > xtr[c].max())).mean()) for c in FEATURES}

    sc = StandardScaler().fit(xtr)
    models = {
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42),
        "Ridge": Ridge(alpha=1.0),
    }
    rows = []
    for mname, m in models.items():
        if mname == "Ridge":
            m.fit(sc.transform(xtr), ytr)
            pred = m.predict(sc.transform(xte))
            sv = shap.LinearExplainer(m, sc.transform(xtr)).shap_values(sc.transform(xte))
        else:
            m.fit(xtr, ytr)
            pred = m.predict(xte)
            sv = shap.TreeExplainer(m).shap_values(xte)
        imp = np.abs(sv).mean(axis=0)
        rank = pd.Series(imp, index=FEATURES).rank(ascending=False).astype(int)
        rows.append({
            "dataset": name, "model": mname, "n_train": len(tr), "n_test": len(te),
            "test_r2": r2_score(yte, pred),
            "test_rmse": mean_squared_error(yte, pred) ** 0.5,
            **{f"shap_{c}": imp[i] for i, c in enumerate(FEATURES)},
            **{f"rank_{c}": int(rank[c]) for c in FEATURES},
            **{f"outside_train_range_{c}": outside[c] for c in FEATURES},
        })
    return rows


def main():
    df = pd.read_csv(DATA, parse_dates=["date"])
    rows = run("level", make_level(df)) + run("change_21d", make_change(df))
    res = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    res.to_csv(OUT / "model_comparison.csv", index=False)
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
