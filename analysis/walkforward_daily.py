"""일별 5거래일 지평 walk-forward 평가와 예측구간(A단계 검증 모듈).

- 표본: 일별 환율(ECOS 매매기준율), 금리차, 일별 MMF(2006-04~). 토요일 관측은 뺀다.
- 특성(원점 t에서 이미 아는 값만): 환율 수익률 r1, 5일, 20일, 금리차 5일 변화(직전 관측), MMF 5일 로그 변화(1일 지연).
- 모형: RW(무변화), AR(수익률만), A(AR+금리차), B(A+MMF). 모두 선형 OLS이고 모형을 키우지 않는다.
- walk-forward: 확장 창, 21거래일마다 재추정, 추정에는 목표가 원점 t까지 실현된 관측만 쓴다.
- 구간: 중심 0(RW)에 EWMA 변동성(lambda=0.94) x sqrt(5)로 규모를 잡고, 정규 분위수와 conformal(최근 750개 표준화 오차의 분위수)을 비교한다.
- 변동성 구간: 원점의 20일 실현변동성이 과거 값들의 확장 창 67% 분위수보다 크면 고변동.
shortcut: 재추정 간격 21일, EWMA lambda, conformal 창 750은 튜닝하지 않았다. 변경하려면 상수만 바꾼다.
"""
import numpy as np
import pandas as pd
from scipy import stats

from run_time_ordered_checks import DATA, series

H = 5
REFIT = 21
MIN_TRAIN = 750
LAMBDA = 0.94
CAL_WINDOW = 750
LEVELS = (0.80, 0.95)
FEATURES = {"AR": ["r1", "r5", "r20"], "A": ["r1", "r5", "r20", "ds5"], "B": ["r1", "r5", "r20", "ds5", "dm5"]}


def build_daily_frame(fx=None, spread=None, mmf=None):
    """일별 표본. fx, spread, mmf는 날짜 인덱스의 Series(테스트에서 가짜 자료를 넣기 위한 인자)."""
    if fx is None:
        fx = series("exchange_rate/USD_KRW_processed.csv", "USD_KRW")
    if spread is None:
        spread = series("policy_rate/spread_KOR_USA_processed.csv", "RATE_SPREAD_KOR_USA")
    if mmf is None:
        m = pd.read_csv(DATA / "m2/KOR/MMF/MMF_daily_2006_2026.csv", skiprows=4, header=None, usecols=[0, 1],
                        names=["date", "mmf"], thousands=",")
        mmf = m.assign(date=pd.to_datetime(m.date, format="%Y/%m/%d")).set_index("date").mmf.sort_index() / 10
    d = pd.DataFrame({"fx": fx}).loc[mmf.index.min():]
    d = d[d.index.dayofweek < 5]
    d.index.name = "date"
    for name, s in (("spread", spread), ("mmf", mmf)):  # 직전 관측(당일 제외)만 쓴다
        d[name] = pd.merge_asof(d[[]], s.rename(name).to_frame(), left_index=True, right_index=True,
                                allow_exact_matches=False, tolerance=pd.Timedelta("45D"))[name]
    d = d.dropna()
    lf = np.log(d.fx)
    d["r1"], d["r5"], d["r20"] = lf.diff(), lf.diff(5), lf.diff(20)
    d["ds5"] = d.spread.diff(5)
    d["dm5"] = np.log(d.mmf).diff(5)
    d["y"] = lf.shift(-H) - lf
    d["vol20"] = d.r1.rolling(20).std()
    return d.dropna(subset=["r1", "r5", "r20", "ds5", "dm5", "vol20"])


def _ols(X, y):
    return np.linalg.lstsq(np.column_stack([np.ones(len(X)), X]), y, rcond=None)[0]


def walk_forward(d):
    """원점별 예측(RW=0, AR, A, B)과 실제 y. 원점 i의 학습은 s + H <= i 인 s만 쓴다."""
    d = d.reset_index()
    y = d.y.to_numpy()
    rows = []
    coefs = {}
    for i in range(MIN_TRAIN, len(d) - H):
        if (i - MIN_TRAIN) % REFIT == 0:
            m = i - H + 1
            coefs = {k: _ols(d.loc[: m - 1, cols].to_numpy(), y[:m]) for k, cols in FEATURES.items()}
        row = {"date": d.loc[i, "date"], "y": y[i], "RW": 0.0}
        for k, cols in FEATURES.items():
            row[k] = coefs[k][0] + d.loc[i, cols].to_numpy(float) @ coefs[k][1:]
        rows.append(row)
    return pd.DataFrame(rows).set_index("date")


def regimes(d, index):
    """원점의 20일 실현변동성이 과거(원점 이전) 값 67% 분위수보다 크면 고변동. 미래 값은 쓰지 않는다."""
    thr = d.vol20.expanding(min_periods=250).quantile(0.67).shift(1)
    return pd.Series(np.where(d.vol20 > thr, "고변동", "안정"), index=d.index).reindex(index)


def ewma_sigma(r1, lam=LAMBDA):
    r = r1.to_numpy()
    s2 = np.empty(len(r))
    s2[0] = np.var(r[:20]) if len(r) >= 20 else r[0] ** 2
    for i in range(1, len(r)):
        s2[i] = lam * s2[i - 1] + (1 - lam) * r[i] ** 2
    return pd.Series(np.sqrt(s2), index=r1.index)


def intervals(d, fc, reg=None, sigma=None, lam=LAMBDA, cal_window=CAL_WINDOW):
    """RW 중심 구간: 상수 변동성, EWMA 정규, EWMA conformal(conf), 변동성 구간별 conformal(confr, reg를 주면).
    원점 t의 보정에는 s + H <= t 인 s의 점수만 쓴다. confr은 같은 변동성 구간(안정/고변동)의 과거 점수만 쓴다."""
    sig = (sigma if sigma is not None else ewma_sigma(d.r1, lam).reindex(fc.index) * np.sqrt(H))
    idx = fc.index
    y = fc.y.to_numpy()
    score = np.abs(y) / sig.to_numpy()
    out = pd.DataFrame({"y": fc.y, "sigma": sig})
    const_sd = pd.Series(y).expanding(min_periods=100).std().shift(H).to_numpy()  # 실현된 y만
    for lev in LEVELS:
        z = stats.norm.ppf(0.5 + lev / 2)
        out[f"const_hw_{int(lev * 100)}"] = z * const_sd
        out[f"norm_hw_{int(lev * 100)}"] = z * sig.to_numpy()
        hw = np.full(len(idx), np.nan)
        hwr = np.full(len(idx), np.nan)
        labels = reg.reindex(idx).to_numpy() if reg is not None else None
        for i in range(len(idx)):
            past = score[max(0, i - H + 1 - cal_window): max(i - H + 1, 0)]
            if len(past) < 100:
                continue
            k = int(np.ceil((len(past) + 1) * lev))
            q = np.sort(past)[min(k, len(past)) - 1]
            hw[i] = q * sig.iloc[i]
            if labels is not None:
                lo = max(0, i - H + 1 - cal_window)
                same = past[labels[lo: max(i - H + 1, 0)] == labels[i]]
                if len(same) >= 100:
                    kr = int(np.ceil((len(same) + 1) * lev))
                    hwr[i] = np.sort(same)[min(kr, len(same)) - 1] * sig.iloc[i]
        out[f"conf_hw_{int(lev * 100)}"] = hw
        if labels is not None:
            out[f"confr_hw_{int(lev * 100)}"] = hwr
    return out


def interval_score(y, hw, lev):
    """Winkler 구간 점수(작을수록 좋다). 중심 0, 반폭 hw."""
    alpha = 1 - lev
    miss = np.maximum(np.abs(y) - hw, 0)
    return 2 * hw + (2 / alpha) * miss


def har_sigma(d, index):
    """HAR형 변동성: 다음 5일 평균 제곱수익률을 [오늘, 5일, 22일 평균 제곱수익률]에 선형 회귀(확장 창, 21일마다 재추정).
    학습에는 목표가 원점까지 실현된 관측만 쓴다. 5일 수익률의 규모 sigma = sqrt(5 * 예측 일별 분산)."""
    r2 = d.r1 ** 2
    X = pd.DataFrame({"a": r2, "b": r2.rolling(5).mean(), "c": r2.rolling(22).mean()})
    target = r2[::-1].rolling(H).mean()[::-1].shift(-1)  # t+1 ~ t+5 평균 제곱수익률
    pos = {dt: i for i, dt in enumerate(d.index)}
    Xv, yv = X.to_numpy(), target.to_numpy()
    out, beta = {}, None
    for n, dt in enumerate(index):
        i = pos[dt]
        if n % REFIT == 0:
            m = i - H + 1  # 행 0..i-H의 목표가 i까지 실현됨
            ok = np.isfinite(Xv[:m]).all(axis=1) & np.isfinite(yv[:m])
            beta = np.linalg.lstsq(np.column_stack([np.ones(ok.sum()), Xv[:m][ok]]), yv[:m][ok], rcond=None)[0] if ok.sum() > 250 else None
        if beta is None or not np.isfinite(Xv[i]).all():
            continue
        out[dt] = np.sqrt(H * max(beta[0] + Xv[i] @ beta[1:], 1e-10))
    return pd.Series(out).reindex(index)
