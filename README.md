# 금리차 디커플링 구간에서 단기 유동성이 원/달러 환율 변동에 미치는 영향 분석

원/달러 환율은 한국 경제의 핵심 거시금융 변수다. 환율 상승은 수입물가, 기업 비용, 무역수지, 외국인 자금 흐름, 자산가격, 소비심리와 연결된다. 이 프로젝트는 전통적인 설명 변수인 한미 정책금리차가 환율을 충분히 설명하지 못하는 구간에서, M2 내부의 단기성 유동성이 환율 움직임을 설명하는 데 유의한 정보를 제공하는지 검증한다.

한 문장으로 요약하면, **단기 유동성은 모든 기간의 보편적 환율 설명 변수라기보다, 금리차와 환율의 관계가 약해지는 이상구간 레짐에서 조건부 설명력과 선행성을 보이는 변수**로 해석된다.

> 주의: 본 연구는 M2가 환율을 구조적으로 직접 유발한다고 단정하지 않는다. Granger 인과성은 시간 선행성과 예측 정보에 대한 통계적 근거이며, SHAP은 특정 모델의 예측 기여도를 설명하는 도구다.

---

## Executive Summary

| 질문 | 결론 |
| --- | --- |
| 장기적으로 환율은 어떤 변수와 움직이는가? | 장기 구간에서는 M2, CPI, 금리차 등 거시 변수와 일정한 평균 관계가 관찰된다. 특히 금리차는 전통적 환율 설명과 부합하는 방향성을 가진다. |
| 금리차 설명력이 약해지는 구간은 어떻게 잡는가? | 환율과 한미 정책금리차의 이동상관관계를 이용해 디커플링 구간을 탐지했다. 최종 분석은 1995-01-01부터 2026-03-20까지의 자료에서 추출된 668개 블록, 3,588개 관측일을 사용한다. |
| 이상구간에서 중요한 설명 변수는 무엇인가? | 랜덤 포레스트와 SHAP 분석에서 M2, 특히 단기성 자금 계열이 이상구간 환율 움직임을 설명하는 데 중요한 정보로 나타났다. |
| 왜 선형 상관은 약한데 SHAP에서는 M2가 중요한가? | 단기 유동성의 영향이 평균적인 선형 관계가 아니라 임계점 이후 커지는 비선형 구조일 가능성이 있다. 일별 단기 유동성 분석에서 약 5.1조 원 수준의 임계점이 관찰되었다. |
| M2가 환율보다 먼저 움직이는가? | 이상구간 레짐에서 M2 변화는 환율 변화를 선행하는 방향으로 유의했다. 반대로 환율이 M2를 선행하는 방향은 유의하지 않았다. 정상구간에서는 같은 관계가 유의하지 않았다. |
| 예측 모델에서도 개선되는가? | 전체 구간에서는 금리차 단독 모델이 더 안정적인 경우가 있었지만, 이상구간에서는 MMF 또는 수시입출식저축성예금을 추가한 모델이 상대적으로 더 낮은 오차를 보였다. |
| 환율 충격은 어디로 파급되는가? | 확장 분석에서는 event-time local projection으로 환율 충격 이후 소비자심리지수, 수입액, 외국인 투자, 무역수지 등 8개 변수의 1-6개월 반응을 추정했다. |

---

## 연구 질문

이 연구는 단순히 "최근 환율이 왜 올랐는가"를 묻지 않는다. 더 좁고 검증 가능한 질문으로 바꾸었다.

1. 장기적으로 원/달러 환율과 주요 거시 변수는 어떤 평균 관계를 보이는가?
2. 한미 정책금리차와 환율의 관계가 약해지는 이상구간 레짐은 어떻게 정의할 수 있는가?
3. 이상구간 레짐에서는 금리차보다 단기 유동성 변수가 환율 움직임을 더 잘 설명하는가?
4. 단기 유동성의 영향은 선형적인가, 아니면 특정 수준 이후 커지는 임계점 구조인가?
5. M2 변화는 환율 변화를 시간적으로 선행하는가?
6. 단기 유동성 변수를 추가한 예측 모델은 이상구간에서 금리차 단독 모델보다 개선되는가?
7. 환율 충격은 이후 국내 거시, 금융 변수로 어떻게 전달되는가?

---

## 분석 흐름

```mermaid
flowchart LR
    A["데이터 수집<br/>BOK, FRED, 시장자료"] --> B["전처리<br/>날짜 통일, 결측 처리, 빈도 정렬"]
    B --> C["장기 관계 분석<br/>상관관계, Random Forest"]
    C --> D["이상구간 레짐 탐지<br/>금리차-환율 이동상관"]
    D --> E["SHAP 분석<br/>이상구간 변수 기여도"]
    E --> F["M2 구성요소 분석<br/>단기성 자금 분해"]
    F --> G["임계점 검증<br/>MMF/CMA 일별 분석"]
    G --> H["Granger 검정<br/>시간 선행성 확인"]
    H --> I["예측 검증<br/>LSTM, Hybrid"]
    I --> J["환율 영향 확장<br/>Event-time local projection"]
```

---

## 데이터와 분석 단위

| 구분 | 변수 | 빈도 | 역할 |
| --- | --- | --- | --- |
| 목표 변수 | USD/KRW 환율 | 일별, 월별 | 분석 대상, 예측 대상, 이상구간 탐지 기준 |
| 금리 변수 | 한국 기준금리, 미국 정책금리, 한미 정책금리차 | 일별, 월별 | 전통적 환율 설명 변수 |
| 통화량 | 한국 M2, 미국 M2 | 월별, 필요 시 일별 정렬 | 장기 요인, 이상구간 선행성 분석 |
| M2 구성요소 | 수시입출식저축성예금, 요구불예금, MMF, 현금통화 등 | 월별 | 단기성 자금 증가 여부 분석 |
| 단기 유동성 | MMF, CMA 보조 지표 | 일별 | 임계점 회귀, SHAP 의존도 분석 |
| 확장 변수 | CPI, 산업생산, KOSPI, 무역수지, 외국인 주식/채권 투자, CSI/BSI 등 | 월별 | 환율 충격의 거시, 금융 파급효과 분석 |

주요 데이터셋:

- `data/processed_daily_1995_2026_integrated.csv`: 장기 일별 통합 데이터
- `analysis/anomaly/anomaly_concatenated_dataset.csv`: 이상구간 레짐 결합 데이터
- `analysis/LSTM/lstm_mmf/daily_dataset.csv`: MMF 기반 일별 LSTM 데이터
- `analysis/LSTM/lstm_m2_demand_deposit/daily_dataset_m2_demand_deposit.csv`: 수시입출식저축성예금 기반 일별 LSTM 데이터
- `data/integrated_macro_targets.csv`: 환율 영향 확장 분석용 월별 거시, 금융 통합 데이터

---

## 이상구간 레짐 정의

초기 분석은 2024년 11월 이후의 최근 고환율 기간을 고정된 이상기간으로 보았다. 그러나 이 방식은 시작일과 종료일 선택의 자의성이 있다. 최종 분석에서는 특정 사건을 임의로 자르기보다, **한미 정책금리차와 원/달러 환율의 관계가 실제로 약해지는 시장 상태**를 데이터 기반으로 추출했다.

사용한 핵심 기준은 다음과 같다.

- 환율과 한미 정책금리차의 30일 이동상관관계 계산
- 이동상관관계가 설정 기준 이하로 약화되는 관측치를 이상구간 후보로 분류
- 최종 분석 데이터 범위: 1995-01-01 ~ 2026-03-20
- 최종 이상구간 결합 자료: 668개 블록, 3,588개 관측일

![정상구간과 이상구간 상관관계 비교](analysis/anomaly/correlation_comparison_updated.png)

이상구간 레짐은 하나의 긴 사건이 아니다. 외환위기, 금융위기, 팬데믹, 글로벌 긴축기처럼 금리차-환율 관계가 반복적으로 약화되는 여러 시장 상태를 하나의 분석 레짐으로 묶은 것이다.

---

## 주요 결과 1. 장기 관계와 이상구간의 괴리

장기 구간에서는 환율과 주요 거시 변수 사이에 평균적인 관계가 관찰된다.

- M2와 CPI는 환율과 양의 관계를 보인다.
- 한미 정책금리차는 환율과 음의 관계를 보이며, 전통적 설명과 대체로 부합한다.
- 그러나 이상구간 레짐에서는 이 평균 관계가 충분하지 않다.

핵심은 **장기 평균 관계와 이상구간 조건부 관계를 분리해야 한다**는 점이다. 정상구간까지 포함한 전체 표본에서는 금리차나 환율 자체의 관성이 강하게 작동하지만, 금리차 설명력이 약해지는 구간에서는 단기 유동성 변수가 추가 정보를 제공한다.

---

## 주요 결과 2. SHAP 분석과 M2 구성요소

이상구간에서 랜덤 포레스트와 SHAP 분석을 수행했다. 이상구간 SHAP 분석에 사용된 관측치는 3,588개이며, 같은 구조의 baseline 분석에는 7,814개 관측치가 사용되었다.

`analysis/shap_ml/results_anomaly/importance_summary.csv` 기준 이상구간 랜덤 포레스트 중요도는 다음과 같다.

| 변수 | RF importance | mean abs SHAP |
| --- | ---: | ---: |
| `M2_KOR` | 0.963 | 18.411 |
| `policy_spread` | 0.027 | 27.568 |
| `M2_USA` | 0.010 | 11.868 |

해석상 중요한 점은 두 가지다.

1. M2는 이상구간 모형에서 강한 설명 변수로 사용되었다.
2. SHAP은 인과관계가 아니라 모델 기여도다. 따라서 이후 임계점 회귀와 Granger 검정으로 비선형성과 시간 선행성을 별도로 확인했다.

![이상구간 SHAP 요약](analysis/shap_ml/results_anomaly/shap_summary.png)

M2 총량만으로는 어떤 자금이 환율 불안과 관련되는지 알기 어렵다. 그래서 M2 내부 구성요소를 분해했다. 분석 결과는 수시입출식저축성예금, 요구불예금, MMF 등 빠르게 이동 가능한 단기성 자금에 주목해야 함을 보여준다.

![M2 구성요소 분석](analysis/m2_components/m2_components_analysis.png)

---

## 주요 결과 3. 단기 유동성 임계점 가설

선형 상관관계에서는 이상구간의 M2-환율 관계가 약하게 보일 수 있다. 그러나 SHAP에서는 M2와 단기성 자금이 중요하게 나타난다. 이 차이를 설명하기 위해 **임계점 가설**을 세웠다.

가설은 다음과 같다.

> 단기 유동성이 조금씩 증가할 때마다 환율이 같은 비율로 상승하는 것이 아니라, 특정 수준을 넘는 순간 환율 반응이 급격히 커질 수 있다.

일별 단기 유동성 데이터로 threshold regression을 수행한 결과, 기존 결과 문서 기준으로 일일 유동성이 약 5.1조 원 이상 증가하는 구간에서 환율 반응이 비선형적으로 커지는 패턴이 관찰되었다. 또한 이 임계값을 넘는 날은 이상기간에 더 자주 나타났다.

![일별 단기 유동성 임계점 회귀](analysis/daily_threshold_MMF/threshold_regression_daily.png)

MMF와 CMA를 나누어 SHAP dependence를 확인하면 MMF에서 임계점 이후 기여도가 더 뚜렷하게 나타난다. CMA는 같은 패턴이 상대적으로 명확하지 않았다.

![MMF SHAP 의존도](analysis/daily_shap_MMF/daily_shap_dependence_mmf.png)

![CMA SHAP 의존도](analysis/daily_shap_MMF/daily_shap_dependence_cma.png)

---

## 주요 결과 4. Granger 인과성 검정

Granger 인과성 검정은 단기 유동성이 환율보다 먼저 움직이는지 확인하기 위한 시계열 검정이다. 구조적 인과를 증명하지는 않지만, 한 변수의 과거값이 다른 변수의 미래 변화를 예측하는 데 유의한 정보를 제공하는지 검정한다.

검정은 이상구간과 정상구간을 나누어 수행했다. 특히 이상구간 결합 자료에서는 서로 떨어진 블록 사이의 가짜 점프를 줄이기 위해 구간 내부 변화율을 기준으로 해석한다.

| 구간 | 방향 | Lag | F-stat | p-value | 해석 |
| --- | --- | ---: | ---: | ---: | --- |
| 이상구간 | FX -> M2 | 2 | 0.1223 | 0.8849 | 유의하지 않음 |
| 이상구간 | M2 -> FX | 2 | 25.7423 | 8.28e-12 | 유의함 |
| 정상구간 | FX -> M2 | 2 | 0.0321 | 0.9685 | 유의하지 않음 |
| 정상구간 | M2 -> FX | 2 | 1.8359 | 0.1723 | 유의하지 않음 |

결론은 명확하다. **이상구간 레짐에서는 M2 변화가 환율 변화를 선행하는 방향으로 유의했지만, 정상구간에서는 같은 관계가 유의하지 않았다.** 따라서 단기 유동성의 선행성은 모든 기간의 일반 법칙이라기보다 금리차 설명력이 약해진 시장 상태에서 나타나는 조건부 현상으로 해석하는 것이 적절하다.

---

## 주요 결과 5. 예측 모델 검증

예측 모델의 목적은 모든 기간에서 환율을 가장 잘 맞히는 절대 최적 모델을 만드는 것이 아니다. 핵심은 **금리차 단독 모델과 단기 유동성 추가 모델을 비교했을 때 이상구간에서 설명력이 개선되는지** 확인하는 것이다.

모델 정의:

- Model A: `USD_KRW + RATE_SPREAD_KOR_USA`
- Model B: `USD_KRW + RATE_SPREAD_KOR_USA + 단기 유동성 변수`
- 예측 구조: 과거 30일 입력, 5영업일 ahead 예측
- 주요 평가 지표: RMSE, MAE

### MMF 기반 LSTM

| 평가 구간 | Model A RMSE | Model B RMSE | 결과 |
| --- | ---: | ---: | --- |
| 전체 구간 2010-2026 | 43.98 | 44.48 | 금리차 단독 근소 우위 |
| 고정 이상구간 2024-11 ~ 2026-03 | 21.06 | 18.77 | MMF 추가 모델 우위 |
| 이상구간 레짐 extended | 21.16 | 16.62 | MMF 추가 모델 우위 |

### 수시입출식저축성예금 기반 LSTM

| 평가 구간 | Model A RMSE | Model B RMSE | 결과 |
| --- | ---: | ---: | --- |
| 전체 구간 2010-12 ~ 2026-03 | 44.62 | 64.00 | 금리차 단독 우위 |
| 고정 이상구간 2024-11 ~ 2026-03 | 21.21 | 18.98 | 유동성 추가 모델 우위 |
| 이상구간 레짐 extended | 15.07 | 14.89 | 유동성 추가 모델 근소 우위 |

해석은 조건부다. 단기 유동성 변수는 전체 기간에서 항상 예측력을 높이지 않는다. 그러나 금리차 설명력이 약해지는 이상구간에서는 MMF와 수시입출식저축성예금 같은 단기 유동성 변수가 금리차 단독 모델보다 더 낮은 오차를 보였다.

하이브리드 모델에서는 직전 거래일 환율을 그대로 따르는 naive baseline이 매우 강한 비교 대상이었다. 환율은 자기상관이 큰 시계열이므로, 딥러닝 모델이 naive baseline을 일관되게 압도하지 못한 점만으로 단기 유동성 가설을 반박하기는 어렵다. 이 연구에서 더 중요한 비교는 **금리차 단독 모델 대비 유동성 추가 모델의 이상구간 상대 개선**이다.

---

## 주요 결과 6. 환율 영향 분석 확장

본 연구의 중심은 "단기 유동성이 환율을 설명하는가"이지만, 마지막 단계에서는 환율 충격이 이후 국내 거시, 금융 변수로 어떻게 전달되는지도 탐색했다.

최종 확장 분석은 event-time local projection 방식으로 구성했다.

- anomaly event month 수: 68
- event-time panel row 수: 2,937
- horizon: 1-6개월
- target: `CSI_CCSI`, `Imports`, `KOSPI`, `Industrial_Production`, `Import_Price_Index`, `Foreign_Bond_Investment`, `Foreign_Stock_Investment`, `Trade_Balance`
- 시나리오: event month predicted USD/KRW에 +5% 충격 적용

모형 비교 결과, predicted FX 기준 전체 평균 성능은 ElasticNet local projection이 가장 낮은 평균 RMSE를 보였다.

| 모델 | avg RMSE | avg NRMSE | targets |
| --- | ---: | ---: | ---: |
| LocalProjection ElasticNet | 1770.13 | 1.2328 | 8 |
| LocalProjection Ridge | 1779.84 | 1.2707 | 8 |
| LocalProjection OLS | 1780.42 | 1.2762 | 8 |

상대적으로 해석 안정성이 높은 target은 `CSI_CCSI`, `Foreign_Bond_Investment`, `Foreign_Stock_Investment`, `Imports`로 정리된다. `Import_Price_Index`, `Industrial_Production`, `KOSPI`, `Trade_Balance`는 모형 간 변동성과 NRMSE를 고려해 신중하게 해석해야 한다.

![환율 충격 이후 주요 변수 반응 요약](analysis/fx_impact/reports/final/event_panel/plots/response_summary_top_targets.png)

---

## 종합 해석

연구 결과는 네 문장으로 정리할 수 있다.

1. 장기 평균에서는 금리차, M2, CPI 등 거시 변수가 환율과 일정한 관계를 보인다.
2. 그러나 금리차-환율 관계가 약해지는 이상구간 레짐에서는 평균적 선형 관계만으로 설명이 부족하다.
3. 이상구간에서 M2 및 단기성 유동성은 모델 기여도, 임계점 구조, Granger 선행성, 예측 성능 비교에서 일관되게 중요한 정보로 나타났다.
4. 단기 유동성은 보편적 원인이라기보다, 금리차 설명력이 약해지는 시장 상태에서 환율 리스크를 조기에 감지할 수 있는 조건부 지표 후보로 해석된다.

---

## 한계

- **구조적 인과의 한계**: Granger 인과성은 시간 선행성과 예측 정보를 보여주지만, 제3의 외생 충격이나 글로벌 위험회피 심리가 M2와 환율을 동시에 움직였을 가능성은 남아 있다.
- **모형 의존성**: SHAP 결과는 랜덤 포레스트 모델의 예측 기여도다. 다른 모델에서도 같은 기여도가 반복된다고 단정할 수 없다.
- **빈도 불일치**: 환율과 MMF 일부 자료는 일별이지만, M2 구성요소, CPI, 산업생산 등은 월별이다. 월별 자료를 일별 인덱스에 맞춘 값은 실제 일별 관측치처럼 해석하면 안 된다.
- **이상구간 결합의 한계**: 서로 떨어진 디커플링 구간을 하나의 레짐으로 묶으면 달력상 간격이 사라질 수 있다. Granger와 event-time 분석에서는 이를 줄이기 위한 처리를 했지만 모든 분석에서 완전히 사라지는 문제는 아니다.
- **초기 결과와 최종 결과의 혼재**: 고정 이상기간 분석, 이상구간 레짐 분석, MMF+CMA 대리 지표 분석이 함께 존재한다. 최종 해석에서는 이상구간 레짐, MMF, M2 구성요소 중심 결과를 우선한다.

---

## 향후 연구

1. MMF, 수시입출식저축성예금, 요구불예금의 실시간 모니터링 체계를 구축한다.
2. 금리차-환율 이동상관, 단기 유동성 증가율, 임계값 초과 여부, DXY, VIX, 환율 변동성을 결합한 환율 리스크 조기경보 지수를 만든다.
3. 정책 이벤트, 지정학적 이벤트, 중앙은행 커뮤니케이션, 뉴스 기반 지표를 추가해 외생 충격과 유동성 효과를 분리한다.
4. 단기 유동성 기반 환율 예측과 event-time local projection을 결합해 "유동성 충격 -> 환율 급등 -> 국내 거시 변수 반응"의 연쇄 구조를 하나의 시스템으로 모델링한다.
5. 데이터 발표 지연과 실시간 관측 가능성을 반영한 순차적 out-of-sample 평가를 수행한다.

---

## 재현 방법

환경에 따라 `python` 대신 `python3`를 사용해야 할 수 있다.

```bash
pip install -r requirements.txt
```

### 1. 데이터 전처리

```bash
python data/process_scripts/process_all_indicators.py
python data/process_scripts/rebuild_daily_pipeline.py
```

### 2. 장기 관계와 이상구간 탐지

```bash
python analysis/baseline/analyze_factors.py
python analysis/anomaly/detect_anomaly_period.py
python analysis/anomaly/build_concatenated_anomaly_dataset.py
```

### 3. SHAP, M2, 임계점 분석

```bash
python analysis/shap_ml/analyze_shap.py
python analysis/m2_components/analyze_m2_components.py
python analysis/daily_threshold_MMF/analyze_daily_threshold.py
python analysis/daily_shap_MMF/analyze_daily_shap.py
```

### 4. LSTM 및 Hybrid 예측

```bash
python analysis/LSTM/lstm_mmf/prep_daily.py
python analysis/LSTM/lstm_mmf/train_eval_extended.py
python analysis/LSTM/lstm_m2_demand_deposit/prep_m2_demand_deposit.py
python analysis/LSTM/lstm_m2_demand_deposit/train_eval_extended.py
python analysis/LSTM/Hybrid/run_hybrid_log_multistep.py
python analysis/LSTM/Hybrid/run_hybrid_m2_demand_deposit.py
```

### 5. 환율 영향 확장 분석

```bash
python analysis/fx_impact/run_final_fx_impact_pipeline.py
```

---

## 주요 산출물

| 목적 | 파일 |
| --- | --- |
| 이상구간 정의 | `analysis/anomaly/period_definition.json` |
| 이상구간 결합 데이터 | `analysis/anomaly/anomaly_concatenated_dataset.csv` |
| SHAP 결과 | `analysis/shap_ml/results_anomaly/importance_summary.csv`, `analysis/shap_ml/results_anomaly/shap_summary.png` |
| M2 구성요소 | `analysis/m2_components/m2_components_analysis.png` |
| Granger 결과 | `analysis/m2_components/granger_result.md`, `analysis/m2_components/granger_findings.json` |
| 임계점 분석 | `analysis/daily_threshold_MMF/threshold_regression_daily.png` |
| MMF/CMA SHAP dependence | `analysis/daily_shap_MMF/daily_shap_dependence_mmf.png`, `analysis/daily_shap_MMF/daily_shap_dependence_cma.png` |
| MMF LSTM 결과 | `analysis/LSTM/lstm_mmf/results.json`, `analysis/LSTM/lstm_mmf/results_extended.json` |
| 수시입출식저축성예금 LSTM 결과 | `analysis/LSTM/lstm_m2_demand_deposit/results.json`, `analysis/LSTM/lstm_m2_demand_deposit/results_extended.json` |
| 환율 영향 최종 결과 | `analysis/fx_impact/result.md`, `analysis/fx_impact/reports/final/event_panel/` |
| 보고서 초안 | `reports/capstone_report_draft.md` |

---

## 참고문헌

- 홍난영, 이윤재, 이태욱. (2023). Hybrid SARIMAX-LSTM 알고리즘을 이용한 시계열 자료 예측. 한국데이터정보과학회지, 34(5), 697-709.
- Neghab, D. P., Cevik, M., & Wahab, M. I. M. (2023). Explaining exchange rate forecasts with macroeconomic fundamentals using interpretive machine learning. arXiv:2303.16149.
- 백재승, 이윤복. (2016). 국제 환율 변동으로 인한 주식시장과 거시경제변수의 상호 연관성: 한국과 미국 비교 분석. 한국산업경영학회 발표논문집.

---

## 코드 구조

| 영역 | 주요 코드 | 역할 | 대표 산출물 |
| --- | --- | --- | --- |
| 데이터 수집/전처리 | `data/process_scripts/process_all_indicators.py`, `data/process_scripts/rebuild_daily_pipeline.py`, `data/process_scripts/create_daily_integrated_dataset.py` | 원천 CSV/외부 데이터를 날짜 기준으로 정리하고 환율, 금리, M2, CPI, 생산지수 등을 통합한다. | `data/macro_dataset_processed.csv`, `data/processed_daily_1995_2026_integrated.csv` |
| 환율/금리 파생 변수 | `data/process_scripts/calculate_spread_and_forward.py`, `data/process_scripts/process_exchange_rate.py` | USD/KRW 환율 정리, 한미 금리차, 이론 환율/선물환율 관련 파생 변수를 만든다. | `data/exchange_rate/USD_KRW_processed.csv`, `data/policy_rate/spread_KOR_USA_processed.csv` |
| 이상구간 탐지 | `analysis/anomaly/detect_anomaly_period.py`, `analysis/anomaly/build_concatenated_anomaly_dataset.py`, `analysis/anomaly/generate_dynamic_periods.py` | 금리차와 환율의 이동상관관계를 이용해 디커플링 구간을 찾고, 이상구간 레짐 분석용 데이터를 만든다. | `analysis/anomaly/period_definition.json`, `analysis/anomaly/anomaly_concatenated_dataset.csv` |
| 장기 기준선 분석 | `analysis/baseline/analyze_factors.py`, `analysis/baseline/analyze_factors_extended.py` | 전체 기간의 상관관계, 변수 중요도, 장기 평균 관계를 확인한다. | `analysis/baseline/result.md`, `analysis/baseline/exchange_rate_heatmap.png` |
| SHAP/머신러닝 해석 | `analysis/shap_ml/analyze_shap.py`, `analysis/shap_ml/shap_analysis_extended.py` | Random Forest 기반 변수 중요도와 SHAP 기여도를 계산해 이상구간의 설명 변수를 해석한다. | `analysis/shap_ml/results_anomaly/importance_summary.csv`, `analysis/shap_ml/results_anomaly/shap_summary.png` |
| M2 구성요소/선행성 | `analysis/m2_components/analyze_m2_components.py`, `analysis/m2_components/granger_causality_analysis.py` | M2 내부 구성요소를 분해하고, M2와 환율 변화의 Granger 선행성을 검정한다. | `analysis/m2_components/m2_components_analysis.png`, `analysis/m2_components/granger_findings.json` |
| 임계점 검증 | `analysis/daily_threshold_MMF/analyze_daily_threshold.py`, `analysis/daily_shap_MMF/analyze_daily_shap.py` | MMF/CMA 일별 변화와 환율 반응의 비선형 임계점 구조를 검토한다. | `analysis/daily_threshold_MMF/threshold_regression_daily.png`, `analysis/daily_shap_MMF/daily_shap_dependence_mmf.png` |
| LSTM 예측 | `analysis/LSTM/lstm_mmf/train_eval_extended.py`, `analysis/LSTM/lstm_m2_demand_deposit/train_eval_extended.py` | 금리차 단독 모델과 단기 유동성 추가 모델의 환율 예측 성능을 비교한다. | `analysis/LSTM/lstm_mmf/results_extended.json`, `analysis/LSTM/lstm_m2_demand_deposit/results_extended.json` |
| Hybrid 예측 | `analysis/LSTM/Hybrid/run_hybrid_log_multistep.py`, `analysis/LSTM/Hybrid/run_hybrid_m2_demand_deposit.py`, `analysis/LSTM/Hybrid/hybrid_models.py` | ARIMA/LSTM/CNN-LSTM 계열 하이브리드 모델로 naive baseline 대비 예측 성능을 점검한다. | `analysis/LSTM/Hybrid/hybrid_mmf/results.json`, `analysis/LSTM/Hybrid/hybrid_m2/results.json` |
| 환율 영향 확장 | `analysis/fx_impact/run_final_fx_impact_pipeline.py`, `analysis/fx_impact/predict_varx.py`, `analysis/fx_impact/predict_arimax.py` | 환율 충격 이후 국내 거시, 금융 변수의 1-6개월 반응을 추정한다. | `analysis/fx_impact/result.md`, `analysis/fx_impact/reports/final/event_panel/` |
| 보고서 생성 | `reports/build_capstone_docx.py`, `reports/render_research_pipeline.py`, `reports/convert_pdfs_to_png.py` | 연구 보고서, 파이프라인 그림, 발표용 자료를 생성하거나 변환한다. | `reports/capstone_report_generated.docx`, `reports/research_pipeline_two_column_fixed.png` |

## 디렉토리 구조

```text
.
├── data/
│   ├── process_scripts/          # 데이터 수집, 정제, 통합 파이프라인
│   ├── exchange_rate/            # USD/KRW 환율 자료
│   ├── policy_rate/              # 한국/미국 정책금리 및 금리차
│   ├── m2/                       # 한국/미국 M2, MMF, CMA, 구성요소
│   ├── CPI/                      # 한국/미국 소비자물가지수
│   ├── production_index/          # 한국/미국 산업생산지수
│   ├── 10y_bond/                 # 한국/미국 10년물 금리
│   ├── KOSPI/                    # 국내 주가지수
│   ├── BOP/                      # 국제수지 관련 자료
│   ├── processed_daily_1995_2026_integrated.csv
│   └── integrated_macro_targets.csv
├── analysis/
│   ├── baseline/                 # 장기 관계 분석
│   ├── anomaly/                  # 이상구간 레짐 탐지 및 결합 데이터
│   ├── shap_ml/                  # Random Forest, SHAP 분석
│   ├── m2_components/            # M2 구성요소 및 Granger 검정
│   ├── daily_threshold_MMF/       # 일별 단기 유동성 임계점 회귀
│   ├── daily_shap_MMF/            # MMF/CMA SHAP dependence
│   ├── LSTM/                     # LSTM 및 Hybrid 환율 예측
│   ├── cpi_usa/                  # 미국 CPI 관련 보조 분석
│   └── fx_impact/                # 환율 충격의 거시, 금융 파급효과
├── reports/
│   ├── legacy/                   # 이전 보고서와 발표 자료
│   ├── capstone_report_draft.md  # 최종 보고서 초안
│   ├── final_presentation_script.md
│   └── research_pipeline_two_column_fixed.png
├── requirements.txt              # Python 의존성
├── AGENTS.md                     # 작업/협업 규칙
└── README.md                     # 연구 요약 보고서
```
