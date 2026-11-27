"""Hybrid(ARIMA + LSTM 보정) walk-forward 평가. 계획과 판정 기준: docs/analysis_plan_hybrid.md (고정 설정, 한 번만 실행).
원 코드의 하이퍼파라미터 탐색(HPO)과 CNN-LSTM은 포함하지 않는다(계획에 명시).
결과: analysis/results_walkforward/hybrid_walkforward_*.csv
"""
import warnings

import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.statespace.sarimax import SARIMAX
from torch.utils.data import DataLoader, TensorDataset

import walkforward_daily as w
from dm_test import dm
from LSTM.lstm_mmf.train_eval_extended import ExRateLSTM, train_model
from run_time_ordered_checks import DATA

warnings.filterwarnings("ignore")
OUT = DATA.parent / "analysis" / "results_walkforward"
REFIT, SEQ, SEEDS = 252, 30, (0, 1, 2)
VARIANTS = {"Hybrid 기본": ["e", "r1", "ds1"], "Hybrid+MMF": ["e", "r1", "ds1", "dm1"]}


def correction(F, y, arima_cum, m, block, seeds=SEEDS):
    """학습 행 SEQ-1..m-1의 (y - ARIMA 예측)을 LSTM으로 학습해 block 원점의 보정량을 시드 평균으로 돌려준다."""
    sc = StandardScaler().fit(F[:m])
    Z = sc.transform(F)
    pos = np.arange(SEQ - 1, m)
    tgt = y[pos] - arima_cum[pos]
    ok = np.isfinite(tgt)
    pos, tgt = pos[ok], tgt[ok]
    scale = tgt.std()
    Xt = torch.tensor(np.stack([Z[p - SEQ + 1: p + 1] for p in pos]), dtype=torch.float32)
    yt = torch.tensor((tgt / scale).reshape(-1, 1), dtype=torch.float32)
    Xe = torch.tensor(np.stack([Z[p - SEQ + 1: p + 1] for p in block]), dtype=torch.float32)
    preds = []
    for seed in seeds:
        torch.manual_seed(seed)
        model = ExRateLSTM(F.shape[1], 32, 1)
        train_model(model, DataLoader(TensorDataset(Xt, yt), batch_size=32, shuffle=True), torch.nn.MSELoss(),
                    torch.optim.Adam(model.parameters(), lr=0.001), 20)
        model.eval()
        with torch.no_grad():
            preds.append(model(Xe).numpy().ravel() * scale)
    return np.mean(preds, axis=0)


def hybrid_predictions(d, origins, max_blocks=None, variants=None):
    """블록(REFIT 원점)마다 ARIMA 계수와 LSTM 보정을 원점까지의 자료로만 다시 추정해 origins의 예측을 만든다.
    max_blocks는 테스트에서 앞쪽 블록만 돌리기 위한 인자다."""
    variants = variants or VARIANTS
    r, y = d.r1.to_numpy(), d.y.to_numpy()
    cols = {k: np.full(len(origins), np.nan) for k in ["ARIMA", *variants]}
    for n_block, start in enumerate(range(0, len(origins), REFIT)):
        if max_blocks is not None and n_block >= max_blocks:
            break
        i0 = origins[start]
        block = origins[start: start + REFIT]
        res = SARIMAX(r[: i0 + 1], order=(1, 0, 1), trend="n").fit(disp=False)
        phi = res.params[0]
        fitted = np.asarray(res.apply(r).fittedvalues)  # 고정 계수의 1단계 예측(각 시점은 과거만 사용)
        e = r - fitted
        nxt = np.r_[fitted[1:], np.nan]  # t 시점 정보로 만든 t+1 예측
        arima_cum = nxt * sum(phi ** k for k in range(w.H))
        m = i0 - w.H + 1
        cols["ARIMA"][start: start + len(block)] = arima_cum[block]
        frame = {"e": e, "r1": r, "ds1": d.ds1.to_numpy(), "dm1": d.dm1.to_numpy()}
        for name, feats in variants.items():
            F = np.column_stack([frame[f] for f in feats])
            cols[name][start: start + len(block)] = arima_cum[block] + correction(F, y, arima_cum, m, block)
        print(f"  Hybrid 재추정 {start // REFIT + 1}/{int(np.ceil(len(origins) / REFIT))} (phi={phi:.3f})", flush=True)
    return pd.DataFrame(cols, index=d.index[origins])


def add_daily_features(d0):
    return d0.assign(ds1=d0.spread.diff(), dm1=np.log(d0.mmf).diff()).dropna(subset=["ds1", "dm1"])


def main():
    d0 = w.build_daily_frame()
    fc = w.walk_forward(d0)
    reg = w.regimes(d0, fc.index)
    d = add_daily_features(d0)
    origins = np.array([d.index.get_loc(t) for t in fc.index if t in d.index])
    keep = fc.index.intersection(d.index)
    fc, reg = fc.loc[keep], reg.loc[keep]
    pred = hybrid_predictions(d, origins)
    full = pd.concat([fc[["y", "RW"]], pred], axis=1)
    full["regime"] = reg
    full.to_csv(OUT / "hybrid_walkforward_predictions.csv", float_format="%.6g")

    rows = []
    samples = {"전체": full, "안정": full[full.regime == "안정"], "고변동": full[full.regime == "고변동"]}
    for lab, s in samples.items():
        e0 = s.y.to_numpy()
        eA = (s.y - s["ARIMA"]).to_numpy()
        for m_ in ["ARIMA", *VARIANTS]:
            e = (s.y - s[m_]).to_numpy()
            _, _, p0 = dm(e, e0)
            row = dict(sample=lab, model=m_, n=len(s), rmse_pct=100 * np.sqrt(np.mean(e ** 2)),
                       ratio_vs_RW=np.sqrt(np.mean(e ** 2) / np.mean(e0 ** 2)), dm_p_vs_RW=p0)
            if m_ != "ARIMA":
                row.update(ratio_vs_ARIMA=np.sqrt(np.mean(e ** 2) / np.mean(eA ** 2)), dm_p_vs_ARIMA=dm(e, eA)[2])
            if m_ == "Hybrid+MMF":
                eB = (s.y - s["Hybrid 기본"]).to_numpy()
                row.update(ratio_vs_Hybrid기본=np.sqrt(np.mean(e ** 2) / np.mean(eB ** 2)), dm_p_vs_Hybrid기본=dm(e, eB)[2])
            hit = (s[m_] > 0) == (s.y > 0)
            row.update(dir_acc=hit.mean(), dir_binom_p=binomtest(int(hit.sum()), len(hit), 0.5).pvalue)
            rows.append(row)
    res_df = pd.DataFrame(rows)
    res_df["dm_p_vs_RW_holm9"] = multipletests(res_df.dm_p_vs_RW, method="holm")[1]
    res_df["dir_p_holm9"] = multipletests(res_df.dir_binom_p, method="holm")[1]
    res_df.to_csv(OUT / "hybrid_walkforward_metrics.csv", index=False, float_format="%.4g")
    pd.set_option("display.width", 250)
    print(res_df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
