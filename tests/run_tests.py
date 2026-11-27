"""의존성 없는 점검 테스트(pytest 없이 실행): python tests/run_tests.py

미래 정보 누수 점검(walk-forward, EWMA, 공표 지연), 예측구간 커버리지, 자료 무결성(환율, MMF, CMA 단위와 일치)을 확인한다.
자료 파일이 없으면 해당 테스트는 건너뛴다고 출력한다.
"""
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "data" / "process_scripts"))

import walkforward_daily as w  # noqa: E402
from run_time_ordered_checks import DATA, series  # noqa: E402


def test_walkforward_ignores_future_data():
    """T 이후 자료를 흔들어도 T까지의 원점 예측과 예측구간은 같아야 한다."""
    cut = pd.Timestamp("2018-06-29")
    base = w.build_daily_frame()
    fc0 = w.walk_forward(base)
    iv0 = w.intervals(base, fc0)

    fx = series("exchange_rate/USD_KRW_processed.csv", "USD_KRW")
    sp = series("policy_rate/spread_KOR_USA_processed.csv", "RATE_SPREAD_KOR_USA")
    m = pd.read_csv(DATA / "m2/KOR/MMF/MMF_daily_2006_2026.csv", skiprows=4, header=None, usecols=[0, 1],
                    names=["date", "mmf"], thousands=",")
    mmf = m.assign(date=pd.to_datetime(m.date, format="%Y/%m/%d")).set_index("date").mmf.sort_index() / 10
    rng = np.random.default_rng(0)

    def shake(s):
        s = s.copy()
        after = s.index > cut
        s[after] = s[after] * (1 + rng.normal(0, 0.1, after.sum()))
        return s

    pert = w.build_daily_frame(fx=shake(fx), spread=shake(sp), mmf=shake(mmf))
    fc1 = w.walk_forward(pert)
    iv1 = w.intervals(pert, fc1)
    idx = fc0.index[fc0.index <= cut]
    for col in ("AR", "A", "B"):
        assert np.allclose(fc0.loc[idx, col], fc1.loc[idx, col]), f"{col} 예측이 미래 자료에 의존한다"
    for col in ("sigma", "conf_hw_80", "conf_hw_95", "const_hw_80"):
        a, b = iv0.loc[idx, col], iv1.loc[idx, col]
        assert np.allclose(a.fillna(-1), b.fillna(-1)), f"{col} 구간이 미래 자료에 의존한다"


def test_ewma_is_causal():
    r = pd.Series(np.random.default_rng(1).normal(0, 0.01, 400), index=pd.date_range("2020-01-01", periods=400))
    full, part = w.ewma_sigma(r), w.ewma_sigma(r.iloc[:300])
    assert np.allclose(full.iloc[:300], part), "EWMA가 미래 수익률을 쓴다"


def test_conformal_coverage_on_synthetic_iid():
    n = 3000
    rng = np.random.default_rng(2)
    idx = pd.date_range("2010-01-01", periods=n, freq="B")
    r1 = pd.Series(rng.normal(0, 0.007, n), index=idx)
    d = pd.DataFrame({"r1": r1})
    fc = pd.DataFrame({"y": rng.normal(0, 0.007 * np.sqrt(w.H), n)}, index=idx)
    iv = w.intervals(d, fc).dropna()
    for lev in w.LEVELS:
        cov = (iv.y.abs() <= iv[f"conf_hw_{int(lev * 100)}"]).mean()
        assert abs(cov - lev) < 0.03, f"conformal 커버리지 {cov:.3f} (목표 {lev})"


def test_interval_score_formula():
    assert np.isclose(w.interval_score(np.array([0.0]), np.array([0.01]), 0.8), 0.02)
    assert np.isclose(w.interval_score(np.array([0.03]), np.array([0.01]), 0.8), 0.02 + (2 / 0.2) * 0.02)


def test_asof_publication_lag():
    """월별 자료는 해당 월 말 다음 달 말부터 보여야 한다(당일 제외)."""
    import build_asof_m2 as b

    s = pd.DataFrame({"observation_date": ["2020-01-01"], "M2": [1.0]})
    tmp = ROOT / "analysis" / "results_walkforward" / "_asof_test.csv"
    tmp.parent.mkdir(exist_ok=True)
    s.to_csv(tmp, index=False)
    old = b.DATA
    try:
        b.DATA = tmp.parent
        v = b.asof_series(tmp.name, "M2", pd.to_datetime(["2020-02-28", "2020-02-29", "2020-03-02"]))
        assert v.isna().tolist() == [True, True, False], v.tolist()
    finally:
        b.DATA = old
        tmp.unlink(missing_ok=True)


def test_data_integrity():
    raw = DATA / "exchange_rate/주요국 통화의 대원화환율_24092601.csv"
    if not raw.exists():
        print("  (건너뜀: 원본 환율 파일 없음)")
    else:
        r = pd.read_csv(raw, encoding="utf-8-sig")
        row = r[r["계정항목"].str.contains("원/미국달러")].iloc[0].iloc[4:]
        row.index = pd.to_datetime(row.index, format="%Y/%m/%d")
        row = pd.to_numeric(row.astype(str).str.replace(",", ""), errors="coerce").dropna()
        proc = series("exchange_rate/USD_KRW_processed.csv", "USD_KRW")
        j = pd.concat([row, proc], axis=1, join="inner")
        assert len(j) == len(proc) and (j.iloc[:, 0] - j.iloc[:, 1]).abs().max() < 0.01, "가공 환율이 ECOS 원본과 다르다"
    old_mmf, new_mmf = DATA / "m2/KOR/MMF/MMF_daily.csv", DATA / "m2/KOR/MMF/MMF_daily_2006_2026.csv"
    if old_mmf.exists() and new_mmf.exists():
        o = pd.read_csv(old_mmf, skiprows=5, header=None, usecols=[0, 1], names=["d", "v"], thousands=",")
        o = o.assign(d=pd.to_datetime(o.d, errors="coerce")).dropna().set_index("d").v
        n = pd.read_csv(new_mmf, skiprows=4, header=None, usecols=[0, 1], names=["d", "v"], thousands=",")
        n = n.assign(d=pd.to_datetime(n.d, format="%Y/%m/%d")).set_index("d").v / 10
        j = pd.concat([o, n], axis=1, join="inner")
        assert len(j) > 3000 and (j.iloc[:, 0] - j.iloc[:, 1]).abs().max() <= 0.5, "MMF 새 자료와 기존 자료가 다르다"
    old_cma, new_cma = DATA / "m2/KOR/CMA/운용대상별 CMA잔고 추이.csv", DATA / "m2/KOR/CMA/CMA_balance_2006_2026.csv"
    if old_cma.exists() and new_cma.exists():
        o = pd.read_csv(old_cma, skiprows=2).rename(columns={"일자": "d", "합계": "v"})
        o = pd.to_numeric(o.assign(d=pd.to_datetime(o.d, format="%Y/%m/%d")).set_index("d").v.astype(str).str.replace(",", ""),
                          errors="coerce").dropna()
        n = pd.read_csv(new_cma, skiprows=3, header=None, thousands=",", na_values=["-"]).iloc[:, [0, 6]]
        n = n.assign(d=pd.to_datetime(n[0], format="%Y/%m/%d")).set_index("d")[6] / 10
        j = pd.concat([o, n], axis=1, join="inner")
        rel = ((j.iloc[:, 0] - j.iloc[:, 1]).abs() / j.iloc[:, 0]).max()  # 구성항목 반올림 합산 차이로 절대 일치는 아니다
        assert len(j) > 2000 and rel < 1e-4, f"CMA 새 자료와 기존 자료가 다르다(상대 오차 {rel:.2e})"


def main():
    tests = [(k, v) for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"통과  {name}")
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"실패  {name}")
            traceback.print_exc()
    print(f"{len(tests) - failed}/{len(tests)} 통과")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
