"""이상구간(금리차 디커플링) 정의 후보를 같은 기준으로 비교한다.

기준: 연구 동기 기간(2024-11 ~ 2025-12)을 얼마나 잡는가(coverage), 레짐 날짜 중 그 기간 비중,
lift(= coverage / 전체 중 레짐 비율, 1이면 우연 수준), 블록 길이(30일 이상 블록이 충분한가). 결과는 results_regime_compare/regime_definitions.csv.
E는 최대우도가 수렴하지 않았다(ConvergenceWarning). D는 과거 창만 쓰는 사전 방식, E는 전체 표본 평활 확률이라 미래 정보를 쓴다(예측 용도로는 부적합).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "processed_daily_1995_2026_integrated.csv"
OUT = Path(__file__).resolve().parent / "results_regime_compare"
MOTIVE = (pd.Timestamp("2024-11-01"), pd.Timestamp("2025-12-31"))


def masks(df):
    fx, sp = df["FX_rate"], df["policy_spread"]
    rc = fx.rolling(30).corr(sp)
    out = {"A rc>-0.2": rc > -0.2, "B |rc|<0.2": rc.abs() < 0.2, "C rc>=0.2": rc >= 0.2}
    # D: 직전 500일 회귀(FX~spread)의 예측에서 벗어난 정도. 계수는 하루 늦춰 사용한다.
    w = 500
    beta = (fx.rolling(w).cov(sp) / sp.rolling(w).var()).shift(1)
    alpha = (fx.rolling(w).mean() - beta * sp.rolling(w).mean()).shift(1)
    scale = (fx.rolling(w).std() * np.sqrt(1 - fx.rolling(w).corr(sp) ** 2)).shift(1)
    z = (fx - (alpha + beta * sp)) / scale
    out["D |resid z|>1.5"] = z.abs() > 1.5
    # E: 환율 로그수익률의 고분산 레짐(2상태 Markov switching, 평활 확률 > 0.5)
    ret = np.log(fx).diff().dropna() * 100
    res = MarkovRegression(ret, k_regimes=2, switching_variance=True).fit(disp=False)
    high = int(np.argmax([res.params[f"sigma2[{i}]"] for i in range(2)]))
    out["E high-vol regime"] = (res.smoothed_marginal_probabilities[high] > 0.5).reindex(df.index).fillna(False)
    valid = {"A rc>-0.2": rc, "B |rc|<0.2": rc, "C rc>=0.2": rc, "D |resid z|>1.5": z}
    return out, valid


def summarize(name, dates, mask, valid):
    ok = mask.fillna(False).astype(bool)
    if valid is not None:
        ok &= valid.notna()
    d = dates[ok]
    blocks = d.diff().dt.days.ne(1).cumsum()
    lens = d.groupby(blocks).size()
    motive = (dates >= MOTIVE[0]) & (dates <= MOTIVE[1])
    return dict(definition=name, days=int(ok.sum()), share_of_days=ok.mean(), blocks=len(lens),
                median_block=float(lens.median()), blocks_ge30=int((lens >= 30).sum()),
                motive_coverage=ok[motive].mean(), lift=ok[motive].mean() / max(ok.mean(), 1e-9), regime_share_in_motive=ok[motive].sum() / max(ok.sum(), 1))


def main():
    df = pd.read_csv(SRC, parse_dates=["date"]).dropna(subset=["FX_rate", "policy_spread"]).sort_values("date").reset_index(drop=True)
    m, valid = masks(df)
    rows = [summarize(n, df["date"], mk, valid.get(n)) for n, mk in m.items()]
    OUT.mkdir(exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "regime_definitions.csv", index=False, float_format="%.3f")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
