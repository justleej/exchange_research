"""한국은행 ECOS 일별 환율(731Y001) 중 원/엔(100엔), 원/유로를 받아 data/exchange_rate/ECOS_cross_rates_daily.csv로 저장한다.
교차통화 복제 검증(analysis/run_cross_currency.py)용이다. ECOS_API_KEY는 .env에서 읽고 출력하지 않는다.
항목 이름은 응답의 ITEM_NAME1로 확인해 기대한 통화가 아니면 중단한다.
"""
import os
import sys
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
KEY = os.getenv("ECOS_API_KEY")
ITEMS = {"JPY100": ("0000002", "일본"), "EUR": ("0000003", "유로")}  # 항목 코드, 이름에 들어 있어야 할 말
OUT = ROOT / "data/exchange_rate/ECOS_cross_rates_daily.csv"


def fetch(code, must_contain):
    url = f"https://ecos.bok.or.kr/api/StatisticSearch/{KEY}/json/kr/1/100000/731Y001/D/19990104/20260331/{code}"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    body = r.json()
    if "StatisticSearch" not in body:
        sys.exit(f"ECOS 응답 오류: {body.get('RESULT', {}).get('MESSAGE', '알 수 없음')}")
    rows = body["StatisticSearch"]["row"]
    name = rows[0]["ITEM_NAME1"]
    if must_contain not in name:
        sys.exit(f"항목 {code}의 이름이 기대와 다르다: {name}")
    s = pd.Series({pd.to_datetime(x["TIME"], format="%Y%m%d"): float(x["DATA_VALUE"]) for x in rows}).sort_index()
    print(f"{code}: {name} {len(s)}건 {s.index.min().date()}~{s.index.max().date()}")
    return s


def main():
    if not KEY:
        sys.exit("ECOS_API_KEY가 .env에 없다.")
    df = pd.DataFrame({k: fetch(*v) for k, v in ITEMS.items()})
    df.index.name = "observation_date"
    df.to_csv(OUT)
    print("saved", OUT.relative_to(ROOT), df.shape)


if __name__ == "__main__":
    main()
