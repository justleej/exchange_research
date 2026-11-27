"""수시입출식저축성예금 6개월 신호의 구조변화(H1a)와 메커니즘(H2a) 검증. 계획: docs/analysis_plan_extra.md

H1a: 계수(x)의 Quandt-Andrews sup-Wald(분기점을 표본 15~85%에서 탐색, Newey-West lag 6). 분기점을 탐색하므로 p는
     유동성 시계열 순환 이동 순열(300회)로 구한다.
H2a: (i) 수시입출식 성장률을 기준금리 변화, KOSPI 월 수익률, 외국인 주식 순매수(1개월 지연)에 회귀(HAC lag 3),
     (ii) 같은 변수를 모형 A, B에 모두 넣은 Clark-West 재검정(h=6, 12).
결과: results_power_regimes/signal_structural_break.csv, signal_mechanism.csv
"""
import numpy as np
import pandas as pd

from run_monthly_horizon import LIQ, make_xy, monthly_base
from run_monthly_signal_check import fast_oos, liquidity_x, nw_t
from run_time_ordered_checks import DATA, series

OUT = DATA.parent / "analysis" / "results_power_regimes"
NAME = "수시입출식저축성예금"
B = 300


def nw_t_coef(y, X, j, L):
    """OLS 계수 j의 Newey-West t 값."""
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    u = y - X @ beta
    xu = X * u[:, None]
    S = xu.T @ xu
    for k in range(1, L + 1):
        G = xu[k:].T @ xu[:-k]
        S += (1 - k / (L + 1)) * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    return beta[j] / np.sqrt(V[j, j]), beta


def sup_wald(d, L=6):
    y = d.y.to_numpy()
    base = np.column_stack([np.ones(len(d)), d[["r", "ds", "x"]].to_numpy()])
    n = len(d)
    best = (-np.inf, None)
    for tau in range(int(0.15 * n), int(0.85 * n)):
        D = (np.arange(n) >= tau).astype(float)
        X = np.column_stack([base, D, d.x.to_numpy() * D])
        t, beta = nw_t_coef(y, X, X.shape[1] - 1, L)
        if t * t > best[0]:
            best = (t * t, tau, beta[3], beta[3] + beta[5])
    return best


def main():
    base = monthly_base()
    x = liquidity_x(base, series(*LIQ[NAME]))
    d = make_xy(base, x, 6).dropna(subset=["y", "r", "ds", "x"])
    stat, tau, before, after = sup_wald(d)
    rng = np.random.default_rng(42)
    n = len(base)
    perm = []
    for _ in range(B):
        xs = pd.Series(np.roll(x.to_numpy(), int(rng.integers(12, n - 12))), index=x.index)
        perm.append(sup_wald(make_xy(base, xs, 6).dropna(subset=["y", "r", "ds", "x"]))[0])
    p = (1 + (np.array(perm) >= stat).sum()) / (1 + B)
    pd.DataFrame([dict(test="Quandt-Andrews sup-Wald (x 계수, h=6)", sup_wald=stat, break_date=d.index[tau].date(),
                       coef_before=before, coef_after=after, permutation_p=p, n=len(d))]).to_csv(
        OUT / "signal_structural_break.csv", index=False, float_format="%.4g")

    # 메커니즘
    macro = pd.read_csv(DATA / "integrated_macro_targets.csv", parse_dates=["Date"])
    macro["Date"] = macro.Date + pd.offsets.MonthEnd(0)
    macro = macro.set_index("Date")
    base["dr"] = macro.Policy_Rate_KOR.diff().reindex(base.index)
    base["kret"] = np.log(macro.KOSPI).diff().reindex(base.index)
    base["fstock"] = macro.Foreign_Stock_Investment.reindex(base.index).shift(1)  # 발표 지연을 감안해 1개월 지연
    path, col = LIQ[NAME]
    s = series(path, col)
    s.index = s.index + pd.offsets.MonthEnd(0)
    g = np.log(s).diff().reindex(base.index)
    g = (g - g.mean()) / g.std()
    rows = []
    reg = pd.concat([g.rename("g"), base[["dr", "kret", "fstock"]]], axis=1).dropna()
    for var in ("dr", "kret", "fstock"):
        z = (reg[var] - reg[var].mean()) / reg[var].std()
        X = np.column_stack([np.ones(len(reg)), z.to_numpy()])
        t, beta = nw_t_coef(reg.g.to_numpy(), X, 1, 3)
        rows.append(dict(check=f"수시입출식 성장률(z) ~ {var}(z)", horizon="-", value=beta[1], t=t, n=len(reg)))
    Xall = np.column_stack([np.ones(len(reg))] + [((reg[v] - reg[v].mean()) / reg[v].std()).to_numpy() for v in ("dr", "kret", "fstock")])
    rows.append(dict(check="세 변수 동시 회귀 R^2", horizon="-",
                     value=1 - ((reg.g.to_numpy() - Xall @ np.linalg.lstsq(Xall, reg.g.to_numpy(), rcond=None)[0]) ** 2).sum()
                     / ((reg.g - reg.g.mean()) ** 2).sum(), n=len(reg)))
    ctrl = ("r", "ds", "dr", "kret", "fstock")
    for h in (6, 12):
        for label, c in (("기준(r, ds)", ("r", "ds")), ("통제(+기준금리 변화, KOSPI 수익률, 외국인 주식)", ctrl)):
            dd = make_xy(base, x, h).dropna(subset=list(c))
            f = fast_oos(dd, h, ctrl=c)
            t = nw_t(f, h)
            from scipy import stats
            rows.append(dict(check=f"CW {label}", horizon=h, value=t, p=float(stats.norm.sf(t)), n=len(f)))
    pd.DataFrame(rows).to_csv(OUT / "signal_mechanism.csv", index=False, float_format="%.4g")
    pd.set_option("display.width", 200)
    print(pd.read_csv(OUT / "signal_structural_break.csv").round(4).to_string(index=False))
    print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
