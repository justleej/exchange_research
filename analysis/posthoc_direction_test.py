"""계획 외 사후 점검: 모델 확장의 방향 정확도가 우연(0.5)과 다른지를 이항검정하고 12개(4개 모델 x 3개 표본)를 Holm 보정한다.
계획(docs/analysis_plan_models.md)은 신뢰구간이 0.5를 포함하는지만 보기로 했고, 이 점검은 결과를 본 뒤 추가한 것이므로 사후 점검으로 표시한다.
입력: results_walkforward/model_extension_predictions.csv. 결과: model_extension_direction_posthoc.csv
"""
import pandas as pd
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests

from run_time_ordered_checks import DATA

OUT = DATA.parent / "analysis" / "results_walkforward"
p = pd.read_csv(OUT / "model_extension_predictions.csv", parse_dates=["date"]).set_index("date")
rows = []
for lab, s in (("전체", p), ("안정", p[p.regime == "안정"]), ("고변동", p[p.regime == "고변동"])):
    for m in ("Ridge", "RF", "GBM", "LSTM"):
        hit = ((s[m] > 0) == (s.y > 0))
        t = binomtest(int(hit.sum()), len(hit), 0.5)
        rows.append(dict(sample=lab, model=m, n=len(hit), dir_acc=hit.mean(), binom_p_vs_0_5=t.pvalue))
r = pd.DataFrame(rows)
r["p_holm_12"] = multipletests(r.binom_p_vs_0_5, method="holm")[1]
r.to_csv(OUT / "model_extension_direction_posthoc.csv", index=False, float_format="%.4g")
print(r.round(4).to_string(index=False))
