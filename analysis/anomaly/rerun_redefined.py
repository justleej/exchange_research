"""이상구간 재정의 후 Granger(블록 안 lag)와 SHAP 모형 비교를 다시 돌린다.
기존 정의(rc <= -0.2)는 금리차-환율의 전통적(음의 상관) 날을 뽑아 디커플링과 방향이 반대였다.
기존 결과는 덮어쓰지 않고 results_redefined/ 에 저장한다."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis" / "shap_ml"))
sys.path.insert(0, str(ROOT / "analysis" / "m2_components"))
import compare_models as cm  # noqa: E402
import granger_block_safe as gb  # noqa: E402

DATA = ROOT / "data" / "processed_daily_1995_2026_integrated.csv"
OUT = Path(__file__).resolve().parent / "results_redefined"
WINDOW = 30
DEFS = {  # 이름: rolling_corr 조건
    "A_not_traditional(rc>-0.2)": lambda r: r > -0.2,
    "B_uncorrelated(|rc|<0.2)": lambda r: r.abs() < 0.2,
    "C_wrong_sign(rc>=0.2)": lambda r: r >= 0.2,
}


def build_blocks(df, mask):
    d = df[mask].copy()
    gap = d["date"].diff().dt.days.ne(1)  # 달력일이 끊기면 새 블록
    d["block_start"] = d["date"].where(gap).ffill()
    return d


def main():
    df = pd.read_csv(DATA, parse_dates=["date"]).dropna(subset=["FX_rate", "policy_spread"]).sort_values("date").reset_index(drop=True)
    df["rc"] = df["FX_rate"].rolling(WINDOW).corr(df["policy_spread"])
    df = df.dropna(subset=["rc"]).reset_index(drop=True)
    OUT.mkdir(exist_ok=True)
    stats, gr, mc = [], [], []
    for name, cond in DEFS.items():
        a = build_blocks(df, cond(df["rc"]))
        L = a.groupby("block_start").size()
        recent = (a["date"] >= "2024-11-01").mean()
        stats.append({"definition": name, "days": len(a), "blocks": len(L), "median_len": float(L.median()),
                      "blocks_ge30": int((L >= 30).sum()), "share_2024-11+": round(recent, 3)})
        for min_len in (1, 30):  # 짧은 조각을 빼고도 같은지 본다
            keep = a[a["block_start"].map(L) >= min_len].sort_values(["block_start", "date"]).copy()
            g = keep.groupby("block_start")
            keep["FX"], keep["M2"] = g["FX_rate"].pct_change(), g["M2_KOR"].pct_change()
            k = keep.dropna(subset=["FX", "M2"])
            for p in (1, 2, 3):
                for dname, y, x in (("M2->FX", "FX", "M2"), ("FX->M2", "M2", "FX")):
                    F, pv, n = gb.granger(k, y, x, p)
                    gr.append({"definition": name, "min_block_len": min_len, "direction": dname, "lag": p, "F": F, "p": pv, "n": n})
        keep = a[a["block_start"].map(L) >= 30]
        for r in cm.run("level", cm.make_level(keep)) + cm.run("change_21d", cm.make_change(keep)):
            mc.append({"definition": name, **r})
    pd.DataFrame(stats).to_csv(OUT / "block_stats.csv", index=False)
    pd.DataFrame(gr).to_csv(OUT / "granger.csv", index=False)
    pd.DataFrame(mc).to_csv(OUT / "model_comparison.csv", index=False)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print(pd.DataFrame(stats).to_string(index=False))
    g = pd.DataFrame(gr)
    print(g[g.direction == "M2->FX"].pivot_table(index=["definition", "min_block_len"], columns="lag", values="p").round(4))
    m = pd.DataFrame(mc)
    print(m[["definition", "dataset", "model", "n_train", "n_test", "test_r2", "rank_policy_spread", "rank_M2_KOR", "rank_M2_USA"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
