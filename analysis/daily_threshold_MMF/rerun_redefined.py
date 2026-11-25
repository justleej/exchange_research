"""임계점 회귀 재실행.
1) CMA+MMF 프록시를 원래 방식(환율/CMA/MMF 내부 병합)으로 다시 만든다.
2) 기존과 같은 TAR 탐색으로 임계값을 구하고, 부트스트랩 sup-F로 임계 효과의 유의성을 검정한다(기존에 없었음).
3) 임계값 초과 빈도를 이상구간별(초기 정의, 고정기간, 재정의 A/B/C)로 비교한다.
결과는 results_redefined/ 에 저장하며 기존 결과는 덮어쓰지 않는다."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis" / "anomaly"))
sys.path.insert(0, str(ROOT / "data" / "process_scripts"))
import rerun_redefined as rr  # noqa: E402  (DEFS, build_blocks, WINDOW 재사용)

OUT = Path(__file__).resolve().parent / "results_redefined"
BOOT = 300
rng = np.random.default_rng(42)


def clean(s):
    return pd.to_numeric(s.astype(str).str.replace(",", "").str.strip(), errors="coerce")


def load_proxy():
    base = pd.read_csv(ROOT / "data/m2/KOR/merged_daily_liquid.csv", parse_dates=["observation_date"])[["observation_date", "USD_KRW", "MMF_total"]]
    cma = pd.read_csv(ROOT / "data/m2/KOR/CMA/운용대상별 CMA잔고 추이.csv", skiprows=2)
    cma = cma.rename(columns={"일자": "observation_date", "합계": "CMA_total"})
    cma["observation_date"] = pd.to_datetime(cma["observation_date"], format="%Y/%m/%d")
    cma["CMA_total"] = clean(cma["CMA_total"])
    d = base.merge(cma[["observation_date", "CMA_total"]].dropna(), on="observation_date", how="inner").sort_values("observation_date")
    d["M2_proxy"] = d["CMA_total"] + d["MMF_total"]
    d["delta_ER"], d["delta_M2"] = d["USD_KRW"].diff(), d["M2_proxy"].diff()
    return d.dropna().reset_index(drop=True)


def design(dm, t):
    D = (dm > t).astype(float)
    return np.column_stack([np.ones_like(dm), dm, D, dm * D])


def rss(X, y):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    return float(r @ r)


def sup_f(dm, y, grid, rss0):
    best = (-np.inf, None)
    for t in grid:
        r1 = rss(design(dm, t), y)
        F = ((rss0 - r1) / 2) / (r1 / (len(y) - 4))  # 제약 2개: D, 교호항
        if F > best[0]:
            best = (F, t)
    return best


def main():
    d = load_proxy()
    OUT.mkdir(exist_ok=True)
    dm, y = d["delta_M2"].to_numpy(), d["delta_ER"].to_numpy()
    pos = dm[dm > 0]
    grid = np.percentile(pos, np.linspace(10, 95, 85))
    X0 = np.column_stack([np.ones_like(dm), dm])
    b0, *_ = np.linalg.lstsq(X0, y, rcond=None)
    rss0 = float(((y - X0 @ b0) ** 2).sum())
    F_obs, t_best = sup_f(dm, y, grid, rss0)
    # 고정 회귀변수 부트스트랩: 귀무(선형) 하에서 y*를 만들어 sup-F 분포를 구한다.
    fit0, e0 = X0 @ b0, y - X0 @ b0
    cnt = 0
    for _ in range(BOOT):
        ys = fit0 + e0 * rng.standard_normal(len(y))
        bs, *_ = np.linalg.lstsq(X0, ys, rcond=None)
        r0 = float(((ys - X0 @ bs) ** 2).sum())
        cnt += sup_f(dm, ys, grid, r0)[0] >= F_obs
    pval = (cnt + 1) / (BOOT + 1)
    d["breach"] = dm > t_best
    print(f"rows {len(d)} | {d.observation_date.min().date()} ~ {d.observation_date.max().date()}")
    print(f"threshold {t_best:.1f} (십억원, {t_best/1000:.2f}조) | sup-F {F_obs:.2f} | bootstrap p {pval:.3f} (B={BOOT})")

    # 이상구간별 임계값 초과 빈도
    daily = pd.read_csv(ROOT / "data/processed_daily_1995_2026_integrated.csv", parse_dates=["date"]).dropna(subset=["FX_rate", "policy_spread"]).sort_values("date").reset_index(drop=True)
    daily["rc"] = daily["FX_rate"].rolling(rr.WINDOW).corr(daily["policy_spread"])
    daily = daily.dropna(subset=["rc"])
    sets = {"fixed(2024-11~2025-12)": set(daily.loc[(daily.date >= "2024-11-01") & (daily.date <= "2025-12-31"), "date"]),
            "orig(rc<=-0.2, 커밋 파일)": set(pd.read_csv(ROOT / "analysis/anomaly/anomaly_concatenated_dataset.csv", parse_dates=["date"])["date"])}
    for name, cond in rr.DEFS.items():
        sets[name] = set(daily.loc[cond(daily["rc"]), "date"])
    rows = []
    for name, ds in sets.items():
        inn = d["observation_date"].isin(ds)
        a, b = d.loc[inn, "breach"], d.loc[~inn, "breach"]
        tab = [[int(a.sum()), int((~a).sum())], [int(b.sum()), int((~b).sum())]]
        rows.append({"regime": name, "n_in": len(a), "breach_in_%": round(a.mean() * 100, 2), "n_out": len(b),
                     "breach_out_%": round(b.mean() * 100, 2), "ratio": round(a.mean() / b.mean(), 2) if b.mean() else np.nan,
                     "fisher_p": round(stats.fisher_exact(tab)[1], 4)})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "breach_by_regime.csv", index=False)
    pd.DataFrame([{"rows": len(d), "threshold_bn_krw": t_best, "sup_F": F_obs, "bootstrap_p": pval, "B": BOOT}]).to_csv(OUT / "threshold_test.csv", index=False)
    pd.set_option("display.width", 200)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
