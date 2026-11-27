"""재검증 결과 그림 3종을 results CSV에서 다시 그린다(초기 정의 기준 그림을 대체하는 최종 그림).
1) 5거래일 수익률과 conformal 80/95% 예측구간(2020-01 이후)
2) 안정/고변동 구간별 RMSE 비율(RW=1)과 예측구간 커버리지
3) 수시입출식저축성예금 6개월 신호의 연도별 Clark-West 기여 비중
출력: analysis/figures/*.png. 사용: python analysis/make_figures.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from run_time_ordered_checks import DATA

ROOT = DATA.parent
RES = ROOT / "analysis"
FIG = RES / "figures"
plt.rcParams["font.family"] = ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def fig_intervals():
    d = pd.read_csv(RES / "results_walkforward/forecast_intervals_h5.csv", parse_dates=["date"]).set_index("date")
    d = d[d.index >= "2020-01-01"]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(d.index, -100 * d.confr_hw_95, 100 * d.confr_hw_95, color="#9ecae1", alpha=0.5, label="95% 구간")
    ax.fill_between(d.index, -100 * d.confr_hw_80, 100 * d.confr_hw_80, color="#3182bd", alpha=0.5, label="80% 구간")
    out = d[d.y.abs() > d.confr_hw_95]
    ax.plot(d.index, 100 * d.y, color="black", lw=0.6, label="실제 5거래일 수익률")
    ax.scatter(out.index, 100 * out.y, color="#d62728", s=10, zorder=3, label="95% 밖")
    ax.set_ylabel("%")
    ax.set_title("원/달러 5거래일 수익률과 변동성 구간별 conformal 예측구간 (중심 0)")
    ax.legend(ncol=4, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "fig1_prediction_intervals.png", dpi=150)
    plt.close(fig)


def fig_regime():
    p = pd.read_csv(RES / "results_walkforward/walkforward_point_metrics.csv")
    p = p[(p["sample"].isin(["안정", "고변동"])) & p.model.isin(["AR", "A", "B"])]
    iv = pd.read_csv(RES / "results_walkforward/walkforward_interval_metrics.csv")
    iv = iv[(iv["sample"] == "전체") & (iv.level == 0.8)]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    width = 0.25
    for i, m in enumerate(["AR", "A", "B"]):
        vals = [p[(p["sample"] == s) & (p.model == m)].rmse_ratio_vs_RW.iloc[0] for s in ("안정", "고변동")]
        axs[0].bar([x + i * width for x in range(2)], vals, width, label=f"모형 {m}")
    axs[0].axhline(1, color="gray", lw=0.8)
    axs[0].set_xticks([x + width for x in range(2)], ["안정", "고변동"])
    axs[0].set_ylim(0.99, 1.02)
    axs[0].set_title("RMSE 비율 (무변화 RW=1, 낮을수록 좋음, 세로축 확대)", fontsize=10)
    axs[0].legend(fontsize=8)
    names = {"const": "상수 변동성", "norm": "EWMA 정규", "conf": "EWMA conformal", "confr": "구간별 conformal"}
    for i, g in enumerate(("안정", "고변동")):
        axs[1].bar([x + i * 0.35 for x in range(4)], [iv[iv.method == k][f"coverage_{g}"].iloc[0] for k in names], 0.35, label=g)
    axs[1].axhline(0.8, color="red", lw=0.8, ls="--")
    axs[1].set_xticks([x + 0.175 for x in range(4)], list(names.values()), fontsize=8)
    axs[1].set_ylim(0.7, 0.92)
    axs[1].set_title("80% 구간 커버리지 (목표 0.80)")
    axs[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_regime_performance.png", dpi=150)
    plt.close(fig)


def fig_signal():
    s = pd.read_csv(RES / "results_power_regimes/monthly_signal_check.csv")
    s = s[s.check.str.startswith("CW 기여 비중")].assign(year=lambda z: z.check.str[-4:])
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.bar(s.year, 100 * s.value, color=["#d62728" if y in ("2008", "2009", "2010", "2011", "2020") else "#7f7f7f" for y in s.year])
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("전체 손실 개선 중 비중 (%)")
    ax.set_title("수시입출식저축성예금 6개월 신호: 표본 밖 개선이 나온 해 (빨강: 2008~2011, 2020)")
    plt.setp(ax.get_xticklabels(), rotation=45, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_signal_by_year.png", dpi=150)
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    fig_intervals()
    fig_regime()
    fig_signal()
    print("saved", sorted(p.name for p in FIG.glob("*.png")))


if __name__ == "__main__":
    main()
