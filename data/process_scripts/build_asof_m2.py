"""월별 M2를 보간하지 않고 '공표 후에만 보이는' 계단형 일별 열로 만든다.

기존 통합 파일의 M2_KOR/M2_USA는 월별 값을 선형보간해, 한 달 값이 다음 달 값과 섞여 미래 정보가 들어간다.
여기서는 해당 월 말 다음 달 말부터 값을 쓸 수 있다고 가정한다(analysis/run_liquidity_lstm.py와 같은 가정).
실제 공표일은 확인하지 않았다(shortcut: 공표일 표를 확보하면 AVAILABILITY_LAG를 날짜별 값으로 교체).
기존 파일은 덮어쓰지 않고 data/processed_daily_asof_m2.csv로 저장한다.
"""
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[1]
SRC = DATA / "processed_daily_1995_2026_integrated.csv"
OUT = DATA / "processed_daily_asof_m2.csv"
SERIES = {  # 열 이름: (파일, 원본 열)
    "M2_KOR_asof": ("m2/KOR/M2_KOR_processed.csv", "M2_KOR"),
    "M2_USA_asof": ("m2/USA/M2SL.csv", "M2SL"),
}


def asof_series(path, col, dates):
    s = pd.read_csv(DATA / path, parse_dates=["observation_date"]).set_index("observation_date")[col].dropna().sort_index()
    s.index = s.index + pd.offsets.MonthEnd(0) + pd.offsets.MonthEnd(1)  # 월 말 + 다음 달 말부터 사용 가능
    left = pd.DataFrame(index=pd.DatetimeIndex(dates))
    return pd.merge_asof(left, s.rename("v").to_frame(), left_index=True, right_index=True, allow_exact_matches=False)["v"]


def main():
    df = pd.read_csv(SRC, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    for name, (path, col) in SERIES.items():
        df[name] = asof_series(path, col, df["date"]).to_numpy()
    out = df.dropna(subset=["FX_rate", "policy_spread", "M2_KOR_asof", "M2_USA_asof"])
    out.to_csv(OUT, index=False)
    # 점검: 기존 보간열이 그날 공표 전 값을 쓴 비율(보간값과 계단값이 다른 날)
    diff = (out["M2_KOR"] - out["M2_KOR_asof"]).abs().gt(1e-6).mean()
    changed = out["M2_KOR_asof"].diff().ne(0).mean()
    print(f"saved {OUT} rows={len(out)} {out.date.min().date()}~{out.date.max().date()}")
    print(f"M2_KOR 보간열이 as-of값과 다른 날 {diff:.1%}, as-of 값이 바뀌는 날 {changed:.1%}")


if __name__ == "__main__":
    main()
