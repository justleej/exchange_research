"""표본 밖(2026-03 이후) 원점의 Hybrid 예측과 예측구간을 만든다. 계획: docs/analysis_plan_hedge_direction.md.
기존 고정 설정 코드(run_hybrid_walkforward.hybrid_predictions)를 그대로 쓰고, 자료만 data/oos/의 무료 출처(ECOS, FRED)로 이어 붙인다.
기존 예측 파일은 덮어쓰지 않는다. 마지막 기존 블록은 다시 계산해 저장값과 같은지 확인한다(재현성 점검).
결과: results_walkforward/hybrid_oos_predictions.csv(전 구간, 새 원점은 oos=True), forecast_intervals_h5_ext.csv
"""
import numpy as np
import pandas as pd

import run_hybrid_walkforward as hw
import walkforward_daily as w
from run_time_ordered_checks import DATA, series

OUT = DATA.parent / "analysis" / "results_walkforward"


def extended_inputs():
    """기존 환율과 금리차 뒤에 새 자료를 이어 붙인다(겹치는 날짜는 기존 값을 쓴다)."""
    fx = series("exchange_rate/USD_KRW_processed.csv", "USD_KRW")
    sp = series("policy_rate/spread_KOR_USA_processed.csv", "RATE_SPREAD_KOR_USA")
    fx_new = series("oos/usd_krw_ecos_2026.csv", "USD_KRW")
    sp_new = series("oos/spread_2025_2026.csv", "RATE_SPREAD_KOR_USA")
    return pd.concat([fx, fx_new[fx_new.index > fx.index.max()]]), pd.concat([sp, sp_new[sp_new.index > sp.index.max()]])


def main():
    old = pd.read_csv(OUT / "hybrid_walkforward_predictions.csv", parse_dates=["date"]).set_index("date")
    fx, sp = extended_inputs()
    d0 = w.build_daily_frame(fx=fx, spread=sp)
    fc = w.walk_forward(d0)
    reg = w.regimes(d0, fc.index)
    iv = w.intervals(d0, fc, reg)
    iv.to_csv(OUT / "forecast_intervals_h5_ext.csv", float_format="%.6g")
    d = hw.add_daily_features(d0)
    keep = fc.index.intersection(d.index)
    fc, reg = fc.loc[keep], reg.loc[keep]
    origins = np.array([d.index.get_loc(t) for t in fc.index])
    assert (fc.index[: len(old)] == old.index).all(), "새 자료가 기존 원점을 바꿨다"
    start = (len(old) // hw.REFIT) * hw.REFIT  # 마지막 기존 블록의 시작(블록 경계는 기존 실행과 같다)
    pred = hw.hybrid_predictions(d, origins[start:])
    full = pd.concat([fc[["y", "RW"]].iloc[start:], pred], axis=1)
    full["regime"] = reg.iloc[start:]
    both = full.index.intersection(old.index)
    for c in ["ARIMA", "Hybrid 기본", "Hybrid+MMF"]:
        diff = (full.loc[both, c] - old.loc[both, c]).abs()
        print(f"재현성 {c}: 기존 블록 {len(both)}개 최대 차이 {diff.max():.2e}, 방향 일치 {((full.loc[both, c] > 0) == (old.loc[both, c] > 0)).mean():.4f}")
    new_rows = full[full.index > old.index.max()]
    out = pd.concat([old, new_rows])
    out["oos"] = out.index > old.index.max()
    out.to_csv(OUT / "hybrid_oos_predictions.csv", float_format="%.6g")
    print(f"새 원점 {len(new_rows)}개 {new_rows.index.min().date()}~{new_rows.index.max().date()}")


if __name__ == "__main__":
    main()
