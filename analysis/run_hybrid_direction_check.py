"""계획 외 사후 점검: Hybrid+MMF의 방향 정확도(0.536, 사전 계획 Holm 보정 후 유의)가 구조적 부산물이 아닌지 본다.
입력: results_walkforward/hybrid_walkforward_predictions.csv, model_extension_predictions.csv(LSTM, 비교용).
1) 연도별 적중률, 2) 예측 방향별 적중률과 기준 비율, 3) 요일, 월말 효과, 4) 예측을 시간 이동한 대조(적중률 분포),
5) 부호를 따른 단순 전략의 평균 5일 수익률(총수익, 거래비용 없음, HAC lag 4).
결과: results_walkforward/hybrid_direction_check.csv. 계획에 없던 점검이므로 결과 문서에 사후 점검으로 표시한다.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from run_time_ordered_checks import DATA

OUT = DATA.parent / "analysis" / "results_walkforward"
MODELS = ["Hybrid 기본", "Hybrid+MMF"]


def main():
    h = pd.read_csv(OUT / "hybrid_walkforward_predictions.csv", parse_dates=["date"]).set_index("date")
    ext = pd.read_csv(OUT / "model_extension_predictions.csv", parse_dates=["date"]).set_index("date")
    h["LSTM(모델 확장)"] = ext["LSTM"].reindex(h.index)
    rows = []
    month_end = h.index.to_series().groupby(h.index.to_period("M")).rank(ascending=False) <= 3  # 월 마지막 3거래일 중 하나
    for m in ["Hybrid 기본", "Hybrid+MMF", "LSTM(모델 확장)"]:
        up = h[m] > 0
        hit = up == (h.y > 0)
        base_up = (h.y > 0).mean()
        rows.append(dict(model=m, check="전체 적중률", value=hit.mean(), note=f"n={len(h)}, 상승 비율 {base_up:.4f}, 예측 상승 비율 {up.mean():.4f}"))
        rows.append(dict(model=m, check="예측 상승일 적중률", value=hit[up].mean(), note=f"n={int(up.sum())}"))
        rows.append(dict(model=m, check="예측 하락일 적중률", value=hit[~up].mean(), note=f"n={int((~up).sum())}"))
        for yr, g in hit.groupby(hit.index.year):
            rows.append(dict(model=m, check=f"연도 {yr} 적중률", value=g.mean(), note=f"n={len(g)}"))
        for dow, g in hit.groupby(hit.index.dayofweek):
            rows.append(dict(model=m, check=f"요일 {dow}(0=월) 적중률", value=g.mean(), note=f"n={len(g)}"))
        rows.append(dict(model=m, check="월말 3거래일 적중률", value=hit[month_end].mean(), note=f"n={int(month_end.sum())}"))
        rows.append(dict(model=m, check="월말 제외 적중률", value=hit[~month_end].mean(), note=f"n={int((~month_end).sum())}"))
        # 시간 이동 대조: 예측을 k일 이동해 같은 방식으로 적중률을 계산
        p = (h[m] > 0).to_numpy()
        yy = (h.y > 0).to_numpy()
        shifts = [k for k in range(-250, 251) if abs(k) >= 10]
        dist = np.array([(np.roll(p, k) == yy).mean() for k in shifts])
        obs = hit.mean()
        rows.append(dict(model=m, check="시간 이동 대조 p(적중률 >= 관측)", value=(1 + (dist >= obs).sum()) / (1 + len(dist)),
                         note=f"이동 {len(dist)}개, 대조 평균 {dist.mean():.4f}, 최대 {dist.max():.4f}"))
        # 단순 전략: 예측 부호를 따른 5일 수익률(겹침 때문에 HAC lag 4)
        strat = np.where(up, 1, -1) * h.y
        fit = sm.OLS(strat.to_numpy(), np.ones(len(strat))).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
        rows.append(dict(model=m, check="부호 추종 평균 5일 수익률(%)", value=100 * strat.mean(),
                         note=f"t={np.asarray(fit.tvalues)[0]:.2f}, 보유(항상 매수) 평균 {100 * h.y.mean():.3f}%"))
    # 부호 확인: MMF 1일 변화(1일 지연)가 5일 수익률과 어떤 관계인가(HAC lag 4). 가설 방향은 양(유동성 증가 -> 환율 상승)이다.
    import walkforward_daily as w
    d = w.build_daily_frame()
    z = pd.DataFrame({"y": d.y, "dm1": np.log(d.mmf).diff()}).dropna()
    z = z[z.index.isin(h.index)]
    for lab, s in (("전체", z), ("~2014", z[z.index < "2015-01-01"]), ("2015~", z[z.index >= "2015-01-01"])):
        fit = sm.OLS(s.y.to_numpy(), sm.add_constant(s.dm1.to_numpy())).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
        rows.append(dict(model="MMF 1일 변화 -> 5일 수익률", check=f"회귀 계수 {lab}", value=fit.params[1],
                         note=f"t={fit.tvalues[1]:.2f}, n={len(s)}"))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "hybrid_direction_check.csv", index=False, float_format="%.4g")
    pd.set_option("display.width", 220, "display.max_colwidth", 90)
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
