"""환율 충격 파급(event-time local projection) 추론 보강.

기존 추정의 약점 세 가지를 고친다.
1) HAC가 이벤트 관측 순서에 적용돼 월 간격을 무시했다 -> 이벤트 월을 달력 간격으로 묶은 에피소드 군집 표준오차.
   같은 셀의 horizon개월 안에 이어지는 이벤트는 한 군집이라 응답 구간이 겹치는 관측이 같은 군집에 들어간다.
2) 학습/평가 분할 중복 -> 분할 없이 공통 달력 구간 전체로 추정한다(예측이 아니라 추정 목적).
3) 환율 충격 식별 부재 -> DXY, VIX, WTI 유가, 미국 정책금리 변화를 통제한 사양을 추가한다. 이것은 구조적 식별이 아니라
   관측 가능한 글로벌 요인을 통제한 것이다.
대상 변수는 기존 README의 8개로 고정한다(재선정 시 바뀌던 산업생산/경상수지 문제 제거).
결과: results_historical_rechecks/event_inference_robust.csv, event_inference_summary.csv
"""
import numpy as np
import pandas as pd
from statsmodels.api import OLS, add_constant
from statsmodels.stats.multitest import multipletests

from fx_impact import run_final_fx_impact_pipeline as lp
from run_historical_rechecks import OUT, event_panels

TARGETS = ["CSI_CCSI", "Imports", "KOSPI", "Industrial_Production", "Import_Price_Index",
           "Foreign_Bond_Investment", "Foreign_Stock_Investment", "Trade_Balance"]
MIN_CLUSTERS = 5
SPECS = {"base_hac_event_order": None, "base_cluster": False, "controls_cluster": True}


def controls():
    m = lp.load_macro_dataset().set_index("Date")
    c = pd.DataFrame({"d_dxy": np.log(m.DXY).diff(), "d_vix": m.VIX.diff(),
                      "d_wti": np.log(m.WTI_Oil.where(m.WTI_Oil > 0)).diff(), "d_us_rate": m.Policy_Rate_USA.diff()})
    return c


def episodes(event_dates, horizon):
    """달력상 이벤트 월 간격이 horizon개월을 넘으면 새 군집."""
    months = event_dates.dt.year * 12 + event_dates.dt.month
    order = months.sort_values()
    new = order.diff().fillna(np.inf).gt(horizon)
    return new.cumsum().reindex(event_dates.index)


def fit_cell(group, horizon, spec, ctrl):
    group = group.sort_values("event_date").copy()
    x = lp.build_lp_features(group, "actual")
    if SPECS[spec]:
        x = x.join(ctrl.reindex(group.event_date).set_axis(group.index))
    x = add_constant(x, has_constant="add")
    ok = x.notna().all(axis=1) & group.actual_response.notna()
    x, y, g = x[ok], group.actual_response[ok], group[ok]
    out = dict(n=len(y), coefficient=np.nan, ci_low=np.nan, ci_high=np.nan, p=np.nan, clusters=0)
    if len(y) < x.shape[1] + 8 or np.linalg.matrix_rank(x) < x.shape[1]:
        return out
    if spec == "base_hac_event_order":
        fit = OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": max(1, min(horizon, len(y) // 4))})
    else:
        groups = episodes(g.event_date, horizon)
        out["clusters"] = groups.nunique()
        if out["clusters"] < MIN_CLUSTERS:
            return out
        fit = OLS(y, x).fit(cov_type="cluster", cov_kwds={"groups": groups, "use_correction": True}, use_t=True)
    out.update(coefficient=fit.params["fx_shock_t"], p=fit.pvalues["fx_shock_t"])
    out["ci_low"], out["ci_high"] = fit.conf_int().loc["fx_shock_t"]
    return out


def main():
    ctrl = controls()
    panels = {name: panel for name, _, panel in event_panels(TARGETS)}
    start = max(p.event_date.min() for p in panels.values())
    end = min(p.event_date.max() for p in panels.values())
    rows = []
    for name, panel in panels.items():
        panel = panel.loc[panel.event_date.between(start, end)]
        assert set(panel.target) <= set(TARGETS)
        for (target, horizon), group in panel.groupby(["target", "horizon"]):
            for spec in SPECS:
                rows.append(dict(definition=name, target=target, horizon=horizon, spec=spec,
                                 window_start=start.date(), window_end=end.date(), **fit_cell(group, horizon, spec, ctrl)))
    res = pd.DataFrame(rows)
    res["p_holm"] = np.nan
    for spec in SPECS:  # 사양별로 계획된 전체 검정(8x6x4=192)을 한 군으로 보정
        sel = res.spec.eq(spec)
        p = res.loc[sel, "p"]
        res.loc[sel, "p_holm"] = multipletests(p.fillna(1), method="holm")[1]
        res.loc[sel & p.isna(), "p_holm"] = np.nan
    OUT.mkdir(exist_ok=True)
    res.to_csv(OUT / "event_inference_robust.csv", index=False)
    summary = res.groupby(["spec", "definition"]).agg(
        cells=("p", "size"), estimated=("p", "count"), nominal=("p", lambda s: int((s < .05).sum())),
        holm=("p_holm", lambda s: int((s < .05).sum())), median_clusters=("clusters", "median")).reset_index()
    summary.to_csv(OUT / "event_inference_summary.csv", index=False)
    print(f"window {start.date()} ~ {end.date()}")
    print(summary.to_string(index=False))
    key = res[res.target.eq("CSI_CCSI") & res.horizon.eq(1)]
    print(key[["definition", "spec", "n", "clusters", "coefficient", "p", "p_holm"]].to_string(index=False))


if __name__ == "__main__":
    main()
