"""모델 확장 검증: Ridge, 랜덤 포레스트, 그래디언트 부스팅, LSTM의 walk-forward 평가와 GARCH-t 변동성 구간 비교.
계획과 판정 기준: docs/analysis_plan_models.md (하이퍼파라미터 고정, 한 번만 실행).
평가 표본과 학습 규칙은 walkforward_daily.py와 같다(학습에는 목표가 원점까지 실현된 관측만).
결과: analysis/results_walkforward/model_extension_*.csv
"""
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from torch.utils.data import DataLoader, TensorDataset

import walkforward_daily as w
from dm_test import dm
from LSTM.lstm_mmf.train_eval_extended import ExRateLSTM, train_model
from run_time_ordered_checks import DATA

OUT = DATA.parent / "analysis" / "results_walkforward"
FEATS = ["r1", "r5", "r20", "r60", "ds5", "spread", "dm5", "dm20", "vol20"]
REFIT_ML, REFIT_LSTM, SEQ, SEEDS = 63, 252, 30, (0, 1, 2)
MODELS = {
    "Ridge": lambda: Ridge(alpha=10),
    "RF": lambda: RandomForestRegressor(n_estimators=200, max_depth=4, min_samples_leaf=50, random_state=0, n_jobs=1),
    "GBM": lambda: GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.03, subsample=0.7, random_state=0),
}


def add_features(d):
    d = d.copy()
    d["r60"] = np.log(d.fx).diff(60)
    d["dm20"] = np.log(d.mmf).diff(20)
    d["ds1"] = d.spread.diff()
    d["dm1"] = np.log(d.mmf).diff()
    return d.dropna(subset=FEATS + ["ds1", "dm1"])


def walk_ml(d, origins):
    """sklearn 모델: REFIT_ML 원점마다 확장 창으로 재추정하고, 다음 재추정 전까지의 원점을 한 번에 예측한다.
    구현상 한 행씩 예측하면 느려서 블록 예측으로 했다(결과는 같다). 계획의 고정 설정은 그대로다."""
    X, y = d[FEATS].to_numpy(), d.y.to_numpy()
    out = {k: np.full(len(origins), np.nan) for k in MODELS}
    for start in range(0, len(origins), REFIT_ML):
        i0 = origins[start]
        m = i0 - w.H + 1
        sc = StandardScaler().fit(X[:m])
        block = origins[start: start + REFIT_ML]
        for k, f in MODELS.items():
            model = f().fit(sc.transform(X[:m]), y[:m])
            if hasattr(model, "n_jobs"):
                model.n_jobs = 1
            out[k][start: start + len(block)] = model.predict(sc.transform(X[block]))
    return pd.DataFrame(out, index=d.index[origins])


def walk_lstm(d, origins):
    """LSTM: REFIT_LSTM 원점마다 재추정, 시드 평균. 입력은 30일 시퀀스[r1, 금리차 1일 변화, MMF 1일 로그 변화]."""
    F, y = d[["r1", "ds1", "dm1"]].to_numpy(), d.y.to_numpy()
    out = np.full(len(origins), np.nan)
    for start in range(0, len(origins), REFIT_LSTM):
        i0 = origins[start]
        m = i0 - w.H + 1  # 행 0..m-1의 목표가 i0까지 실현됨
        sc = StandardScaler().fit(F[:m])
        Z = sc.transform(F)
        train_pos = np.arange(SEQ - 1, m)
        ysc = y[train_pos].std()
        Xt = torch.tensor(np.stack([Z[p - SEQ + 1: p + 1] for p in train_pos]), dtype=torch.float32)
        yt = torch.tensor((y[train_pos] / ysc).reshape(-1, 1), dtype=torch.float32)
        block = origins[start: start + REFIT_LSTM]
        Xe = torch.tensor(np.stack([Z[p - SEQ + 1: p + 1] for p in block]), dtype=torch.float32)
        preds = []
        for seed in SEEDS:
            torch.manual_seed(seed)
            model = ExRateLSTM(3, 32, 1)
            train_model(model, DataLoader(TensorDataset(Xt, yt), batch_size=32, shuffle=True), torch.nn.MSELoss(),
                        torch.optim.Adam(model.parameters(), lr=0.001), 20)
            model.eval()
            with torch.no_grad():
                preds.append(model(Xe).numpy().ravel() * ysc)
        out[start: start + len(block)] = np.mean(preds, axis=0)
        print(f"  LSTM 재추정 {start // REFIT_LSTM + 1}/{int(np.ceil(len(origins) / REFIT_LSTM))}", flush=True)
    return pd.Series(out, index=d.index[origins], name="LSTM")


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def main():
    d0 = w.build_daily_frame()
    fc = w.walk_forward(d0)  # RW, AR, A, B (기존 선형 모형)와 같은 원점
    reg = w.regimes(d0, fc.index)
    d = add_features(d0)
    d = d.loc[d.index.isin(d0.index)]
    origins = np.array([d.index.get_loc(t) for t in fc.index if t in d.index])
    keep = fc.index.intersection(d.index)
    fc = fc.loc[keep]
    reg = reg.loc[keep]
    pred = walk_ml(d, origins)
    print("ML 완료", flush=True)
    pred["LSTM"] = walk_lstm(d, origins)
    full = pd.concat([fc, pred], axis=1)
    full["regime"] = reg

    rows = []
    names = ["Ridge", "RF", "GBM", "LSTM"]
    samples = {"전체": full, "안정": full[full.regime == "안정"], "고변동": full[full.regime == "고변동"]}
    for lab, s in samples.items():
        e0 = s.y.to_numpy()
        eA = (s.y - s["A"]).to_numpy()
        for m in names:
            e = (s.y - s[m]).to_numpy()
            _, _, p0 = dm(e, e0)
            _, _, pA = dm(e, eA)
            hit = (np.sign(s[m]) == np.sign(s.y))
            lo, hi = wilson(int(hit.sum()), len(hit))
            rows.append(dict(sample=lab, model=m, n=len(s), rmse_pct=100 * np.sqrt(np.mean(e ** 2)),
                             ratio_vs_RW=np.sqrt(np.mean(e ** 2) / np.mean(e0 ** 2)), dm_p_vs_RW=p0,
                             ratio_vs_A=np.sqrt(np.mean(e ** 2) / np.mean(eA ** 2)), dm_p_vs_A=pA,
                             dir_acc=hit.mean(), dir_ci_low=lo, dir_ci_high=hi, base_up_rate=(s.y > 0).mean()))
    res = pd.DataFrame(rows)
    res["dm_p_holm"] = multipletests(res.dm_p_vs_RW, method="holm")[1]  # 4개 모델 x 3개 표본 = 12개
    res["dir_ci_excludes_0.5"] = (res.dir_ci_low > 0.5) | (res.dir_ci_high < 0.5)
    res.to_csv(OUT / "model_extension_metrics.csv", index=False, float_format="%.4g")
    full.to_csv(OUT / "model_extension_predictions.csv", float_format="%.6g")

    # GARCH-t 변동성 구간 비교(H6)
    iv_rows = []
    sig = {"EWMA0.94": w.ewma_sigma(d0.r1).reindex(fc.index) * np.sqrt(w.H), "HAR": w.har_sigma(d0, fc.index),
           "GARCH-t": w.garch_sigma(d0, fc.index)}
    ivs = {k: w.intervals(d0, fc, reg, sigma=v) for k, v in sig.items()}
    common = np.logical_and.reduce([iv[["confr_hw_80", "confr_hw_95"]].notna().all(axis=1) for iv in ivs.values()])
    for k, iv in ivs.items():
        iv = iv[common]
        rg = reg[common]
        for lev in w.LEVELS:
            hw = iv[f"confr_hw_{int(lev * 100)}"]
            hit = iv.y.abs() <= hw
            iv_rows.append(dict(sigma=k, level=lev, n=int(common.sum()), coverage=hit.mean(), coverage_안정=hit[rg == "안정"].mean(),
                                coverage_고변동=hit[rg == "고변동"].mean(), halfwidth_pct=100 * hw.mean(),
                                winkler_pct=100 * w.interval_score(iv.y.to_numpy(), hw.to_numpy(), lev).mean()))
    ivr = pd.DataFrame(iv_rows)
    ivr.to_csv(OUT / "model_extension_intervals.csv", index=False, float_format="%.4g")
    pd.set_option("display.width", 230)
    print(res.round(4).to_string(index=False))
    print(ivr.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
