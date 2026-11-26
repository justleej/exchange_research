"""월별 표본이 작아 구간별 검정이 불가능했던 문제를 보완한다.

1) 상호작용 검정: 전체 월별 표본(약 370개월)에서 y_t = a + sum(y lag) + sum(b_k x_{t-k}) + c D_t + sum(g_k D_t x_{t-k})를 적합하고
   g_k = 0(위기 구간에서 유동성 효과가 다르지 않다)을 HC3 Wald로 검정한다. 구간별로 쪼개 표본 31~47개로 검정하는 대신
   전체 표본의 자기회귀 계수를 공유하므로 검정력이 높다.
2) 날짜 민감도: 사전지정 위기 구간의 시작/끝을 +-1개월, +-2개월 옮긴 변형에서도 결론이 같은지 본다.
3) 검정력: 위기 구간에서만 효과 b(y 표준편차 단위, x 1 표준편차당)를 넣고 상호작용 검정의 기각률을 시뮬레이션한다.
   shortcut: y의 lag를 실제값으로 고정한 조건부 검정력.
결과: results_power_regimes/interaction_*.csv
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from run_power_and_event_regimes import BETAS, EVENT_WINDOWS, OUT, monthly_frame

LAGS = (1, 2, 3)
NSIM = 300
SHIFTS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1), (-2, 0), (0, 2), (-1, 1), (-2, 2))  # (시작 이동, 끝 이동) 개월


def dummy(index, windows, ds, de):
    d = pd.Series(0.0, index=index)
    for a, b in windows.values():
        a = pd.Timestamp(a) + pd.DateOffset(months=ds)
        b = pd.Timestamp(b) + pd.DateOffset(months=de)
        d[(index >= a) & (index <= b + pd.offsets.MonthEnd(0))] = 1.0
    return d


def design(ch, D, p, delay):
    x = ch.x.shift(delay)
    X = pd.DataFrame(index=ch.index)
    for k in range(1, p + 1):
        X[f"y{k}"] = ch.y.shift(k)
        X[f"x{k}"] = x.shift(k)
    X["D"] = D
    for k in range(1, p + 1):
        X[f"g{k}"] = D * X[f"x{k}"]
    f = pd.concat([ch.y, X], axis=1).dropna()
    return f.iloc[:, 0], sm.add_constant(f.iloc[:, 1:])


def wald_p(y, X, p):
    fit = sm.OLS(y, X).fit(cov_type="HC3")
    names = [f"g{k}" for k in range(1, p + 1)]
    R = np.zeros((p, X.shape[1]))
    for i, n in enumerate(names):
        R[i, list(X.columns).index(n)] = 1
    return float(fit.wald_test(R, use_f=True, scalar=True).pvalue)


def power(ch, D, p, rng):
    y, X = design(ch, D, p, 0)
    restricted = X.drop(columns=[f"g{k}" for k in range(1, p + 1)])
    base = sm.OLS(y, restricted).fit()
    fit, resid = base.fittedvalues.to_numpy(), base.resid.to_numpy()
    crisis_x1 = (X["g1"]).to_numpy()
    scale = y.std() / X["x1"].std()
    out = {}
    for b in BETAS:
        hits = 0
        for _ in range(NSIM):
            ys = pd.Series(fit + rng.choice(resid, len(resid)) + b * scale * crisis_x1, index=y.index)
            hits += wald_p(ys, X, p) < 0.05
        out[b] = hits / NSIM
    return out


def main():
    rng = np.random.default_rng(42)
    ch, vix = monthly_frame()
    rows, prow = [], []
    variants = {f"창 이동({ds:+d},{de:+d})": dummy(ch.index, EVENT_WINDOWS, ds, de) for ds, de in SHIFTS}
    variants["VIX>=30"] = vix.ge(30).astype(float)
    for name, D in variants.items():
        for delay in (0, 1):
            for p in LAGS:
                y, X = design(ch, D, p, delay)
                rows.append(dict(regime=name, x_delay=delay, lag=p, n=len(y), crisis_months=int(D.sum()),
                                 p=wald_p(y, X, p)))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "interaction_tests.csv", index=False, float_format="%.4g")
    for name in ("창 이동(+0,+0)", "VIX>=30"):
        for p in (1, 3):
            pw = power(ch, variants[name], p, rng)
            ok = [b for b, v in pw.items() if v >= .8]
            prow.append(dict(regime=name, lag=p, mde_sd=min(ok) if ok else np.nan,
                             **{f"power_b{b}": v for b, v in pw.items()}))
    pd.DataFrame(prow).to_csv(OUT / "interaction_power.csv", index=False, float_format="%.4g")
    pd.set_option("display.width", 200)
    print(res.pivot_table(index="regime", columns=["x_delay", "lag"], values="p").round(3).to_string())
    print(res.groupby("regime").crisis_months.first().to_string())
    print(pd.DataFrame(prow).to_string(index=False))


if __name__ == "__main__":
    main()
