"""월·분기·연 지평 환율 예측 검증: 유동성(M2, 수시입출식저축성예금, 요구불예금)의 성장률이 h개월 뒤 환율 변화를 예측하는가.

일별 5거래일 예측은 월별로 갱신되는 유동성에 불리한 설계라, 유동성의 빈도에 맞는 월 단위 지평(h = 1, 3, 6, 12개월)으로 본다.
- 표본: 1999-06 ~ 2026-01 월말. 환율은 월말 값, 금리차는 월말 값, 유동성은 월별(다음 달 말부터 사용 가능하다고 가정해 1개월 지연).
- 표본 내: y = log FX(t+h) - log FX(t)를 [환율 월 변화, 금리차 월 변화, 유동성 성장률(z 표준화)]에 회귀, Newey-West(lag h) 표준오차.
- 표본 밖: 2008-01부터 확장 창, 모형 RW(무변화) / A(환율·금리차 변화) / B(A+유동성). Clark-West 검정(B vs A, 중첩모형, 단측)과 RMSE 비율.
- 위기 구간 상호작용: 사전지정 위기 구간 더미 D와 유동성의 상호작용 계수가 0인지 검정.
검정군별(표본 내, 표본 밖, 상호작용) Holm 보정. 결과: results_power_regimes/monthly_horizon_*.csv
shortcut: 확장 창 재추정에서 h개월 겹침 때문에 표본 밖 예측 오차가 서로 겹친다. CW p는 HAC(lag h-1)로 보정했지만 표본이 작다.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests

from run_power_and_event_regimes import EVENT_WINDOWS
from run_time_ordered_checks import DATA, series

OUT = DATA.parent / "analysis" / "results_power_regimes"
HORIZONS = (1, 3, 6, 12)
OOS_START = "2008-01-31"
LIQ = {"M2": ("m2/KOR/M2_KOR_processed.csv", "M2_KOR"),
       "수시입출식저축성예금": ("m2/KOR/M2_details_processed.csv", "M2_수시입출식저축성예금"),
       "요구불예금": ("m2/KOR/M2_details_processed.csv", "M2_요구불예금")}


def monthly_base():
    fx = series("exchange_rate/USD_KRW_processed.csv", "USD_KRW").resample("ME").last()
    sp = series("policy_rate/spread_KOR_USA_processed.csv", "RATE_SPREAD_KOR_USA").resample("ME").last()
    df = pd.DataFrame({"fx": np.log(fx), "spread": sp}).dropna().loc["1999-06-30":"2026-01-31"]
    df["r"], df["ds"] = df.fx.diff(), df.spread.diff()
    return df


def add_liquidity(df, name):
    path, col = LIQ[name]
    s = series(path, col)
    s.index = s.index + pd.offsets.MonthEnd(0)
    g = np.log(s).diff().reindex(df.index).shift(1)  # 1개월 지연: 직전 달 성장률만 안다
    x = (g - g.mean()) / g.std()
    return x


def make_xy(df, x, h):
    d = df.assign(x=x, y=df.fx.shift(-h) - df.fx).dropna(subset=["r", "ds", "x"])
    return d


def hac_fit(y, X, h):
    return sm.OLS(y, sm.add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags": max(h, 1)})


def oos(d, h, ctrl=("r", "ds"), eval_from=None):
    d = d.dropna(subset=["y"])
    idx = d.index
    start = np.searchsorted(idx, pd.Timestamp(OOS_START))
    rows = []
    for i in range(start, len(d)):
        t = idx[i]
        train = d.iloc[: max(i - h + 1, 0)]  # y가 t까지 실현된 관측만(s + h <= t)
        if len(train) < 60:
            continue
        ctrl = list(ctrl)
        xa, xb = train[ctrl], train[ctrl + ["x"]]
        fa = sm.OLS(train.y, sm.add_constant(xa)).fit()
        fb = sm.OLS(train.y, sm.add_constant(xb)).fit()
        row = d.iloc[i]
        pa = float(fa.params @ np.r_[1, row[ctrl].to_numpy(float)])
        pb = float(fb.params @ np.r_[1, row[ctrl + ["x"]].to_numpy(float)])
        rows.append((t, row.y, pa, pb))
    o = pd.DataFrame(rows, columns=["t", "y", "pa", "pb"]).set_index("t")
    if eval_from:
        o = o[o.index >= eval_from]
    ea, eb, e0 = o.y - o.pa, o.y - o.pb, o.y
    f = ea ** 2 - eb ** 2 + (o.pa - o.pb) ** 2  # Clark-West 조정 손실차
    cw = sm.OLS(f, np.ones(len(f))).fit(cov_type="HAC", cov_kwds={"maxlags": max(h - 1, 1)})
    return dict(n_oos=len(o), rmse_rw=np.sqrt((e0 ** 2).mean()), rmse_A=np.sqrt((ea ** 2).mean()),
                rmse_B=np.sqrt((eb ** 2).mean()), ratio_B_A=np.sqrt((eb ** 2).mean() / (ea ** 2).mean()),
                ratio_A_RW=np.sqrt((ea ** 2).mean() / (e0 ** 2).mean()), cw_t=float(np.asarray(cw.tvalues)[0]),
                cw_p=float(stats.norm.sf(np.asarray(cw.tvalues)[0])))


def crisis_dummy(index):
    d = pd.Series(0.0, index=index)
    for a, b in EVENT_WINDOWS.values():
        d[(index >= a) & (index <= pd.Timestamp(b) + pd.offsets.MonthEnd(0))] = 1.0
    return d


def main():
    OUT.mkdir(exist_ok=True)
    base = monthly_base()
    D = crisis_dummy(base.index)
    is_rows, oos_rows, int_rows = [], [], []
    for name in LIQ:
        x = add_liquidity(base, name)
        for h in HORIZONS:
            d = make_xy(base, x, h)
            dd = d.dropna(subset=["y"])
            fit = hac_fit(dd.y, dd[["r", "ds", "x"]], h)
            is_rows.append(dict(liquidity=name, horizon=h, n=len(dd), coef_per_sd=fit.params["x"],
                                t=fit.tvalues["x"], p=fit.pvalues["x"]))
            oos_rows.append(dict(liquidity=name, horizon=h, **oos(d, h)))
            di = dd.assign(D=D.reindex(dd.index), xD=lambda z: z.x * z.D)
            fi = hac_fit(di.y, di[["r", "ds", "x", "D", "xD"]], h)
            int_rows.append(dict(liquidity=name, horizon=h, n=len(di), crisis_months=int(di.D.sum()),
                                 coef_interaction=fi.params["xD"], p=fi.pvalues["xD"]))
    tabs = {"is": pd.DataFrame(is_rows), "oos": pd.DataFrame(oos_rows), "regime": pd.DataFrame(int_rows)}
    for key, col in (("is", "p"), ("oos", "cw_p"), ("regime", "p")):
        tabs[key]["p_holm"] = multipletests(tabs[key][col], method="holm")[1]
        tabs[key].to_csv(OUT / f"monthly_horizon_{key}.csv", index=False, float_format="%.4g")
    pd.set_option("display.width", 220)
    for key, t in tabs.items():
        print(key); print(t.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
