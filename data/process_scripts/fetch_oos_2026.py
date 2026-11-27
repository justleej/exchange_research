"""2026-03 이후 표본 밖 자료를 무료 출처에서 받아 data/oos/에 저장한다(기존 파일은 덮어쓰지 않는다).
- ECOS 원/달러(731Y001, 0000001)와 한국은행 기준금리(722Y001, 0101000): ECOS_API_KEY는 .env에서 읽고 출력하지 않는다.
- FRED DFF(연방기금 실효금리 일별, 공개 CSV; 기존 FEDFUNDS.csv와 값이 일치하는지 테스트로 확인), Yahoo Finance 원/달러 종가(KRW=X, 비공식, 다른 출처 대조용).
금리차 = 기준금리(일별) - 연방기금 실효금리(일별). 항목 이름은 응답으로 확인해 기대와 다르면 중단한다.
"""
import io
import os
import sys
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
KEY = os.getenv("ECOS_API_KEY")
OUT = ROOT / "data/oos"
END = pd.Timestamp.today().strftime("%Y%m%d")


def ecos(table, item, start, must_contain):
    url = f"https://ecos.bok.or.kr/api/StatisticSearch/{KEY}/json/kr/1/100000/{table}/D/{start}/{END}/{item}"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    body = r.json()
    if "StatisticSearch" not in body:
        sys.exit(f"ECOS 응답 오류: {body.get('RESULT', {}).get('MESSAGE', '알 수 없음')}")
    rows = body["StatisticSearch"]["row"]
    if must_contain not in rows[0]["ITEM_NAME1"]:
        sys.exit(f"항목 {item} 이름이 기대와 다르다: {rows[0]['ITEM_NAME1']}")
    s = pd.Series({pd.to_datetime(x["TIME"], format="%Y%m%d"): float(x["DATA_VALUE"]) for x in rows}).sort_index()
    s.index.name = "observation_date"
    return s


def main():
    if not KEY:
        sys.exit("ECOS_API_KEY가 .env에 없다.")
    OUT.mkdir(exist_ok=True)
    fx = ecos("731Y001", "0000001", "20260301", "원/미국달러").rename("USD_KRW")
    base = ecos("722Y001", "0101000", "20250101", "기준금리").rename("BASE_RATE_KOR")
    r = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFF", timeout=60)
    r.raise_for_status()
    ff = pd.read_csv(io.StringIO(r.text), parse_dates=["observation_date"]).set_index("observation_date").DFF
    # 기준금리는 변경일에만 값이 있을 수 있어 일별로 앞쪽 채움한다(기존 금리차 파일과 같은 방식은 테스트로 확인).
    days = pd.date_range("2025-01-01", fx.index.max())
    spread = (base.reindex(days).ffill() - ff.reindex(days.union(ff.index)).ffill().reindex(days)).dropna()
    spread.name = "RATE_SPREAD_KOR_USA"
    spread.index.name = "observation_date"
    fx.to_csv(OUT / "usd_krw_ecos_2026.csv")
    spread.to_csv(OUT / "spread_2025_2026.csv")
    import yfinance as yf
    y = yf.download("KRW=X", start="2003-12-01", progress=False, auto_adjust=False)["Close"].squeeze().dropna()
    y.index = pd.to_datetime(y.index).tz_localize(None)
    y.name = "USD_KRW_yahoo"
    y.index.name = "observation_date"
    y.to_csv(OUT / "usd_krw_yahoo.csv")
    print(f"ECOS 환율 {len(fx)}건 {fx.index.min().date()}~{fx.index.max().date()}")
    print(f"금리차 {len(spread)}건 {spread.index.min().date()}~{spread.index.max().date()}")
    print(f"Yahoo 환율 {len(y)}건 {y.index.min().date()}~{y.index.max().date()}")


if __name__ == "__main__":
    main()
