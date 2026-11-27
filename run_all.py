"""재검증 결과를 한 번에 재현한다.

python run_all.py          # 빠른 단계(테스트, walk-forward 평가, 그림): 약 1~2분
python run_all.py --full   # 검정력, 위기 구간, 월별 지평, 신호 확인, 임계점 확장까지: 수십 분
LSTM/Hybrid(analysis/run_liquidity_lstm_stationary.py, run_hybrid_time_ordered.py)는 시간이 오래 걸려 포함하지 않는다.
한 단계가 실패하면 거기서 멈춘다. 각 스크립트는 analysis/에서 실행한다.
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FAST = [("테스트(누수, 무결성)", ["tests/run_tests.py"], ROOT),
        ("walk-forward 평가와 예측구간", ["run_walkforward_daily.py"], ROOT / "analysis"),
        ("그림 재생성", ["make_figures.py"], ROOT / "analysis")]
FULL = [("검정력과 외부 기준 레짐", ["run_power_and_event_regimes.py"], ROOT / "analysis"),
        ("레짐 상호작용", ["run_regime_interaction.py"], ROOT / "analysis"),
        ("월별 지평 예측", ["run_monthly_horizon.py"], ROOT / "analysis"),
        ("월별 지평 강건성", ["run_monthly_robustness.py"], ROOT / "analysis"),
        ("수시입출식 신호 확인(순열)", ["run_monthly_signal_check.py"], ROOT / "analysis"),
        ("임계점 MMF 확장", ["run_threshold_mmf_extended.py", "1000"], ROOT / "analysis"),
        ("임계점 프록시(CMA+MMF)", ["run_threshold_mmf_extended.py", "1000", "proxy"], ROOT / "analysis")]


def main():
    stages = FAST + (FULL if "--full" in sys.argv else [])
    for name, cmd, cwd in stages:
        t0 = time.time()
        print(f"[{name}] 시작", flush=True)
        path = cwd / cmd[0] if cwd != ROOT else ROOT / cmd[0]
        r = subprocess.run([sys.executable, str(path), *cmd[1:]], cwd=cwd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        print(f"[{name}] {'완료' if r.returncode == 0 else '실패'} ({time.time() - t0:.0f}초)", flush=True)
        if r.returncode:
            print(r.stderr[-1500:])
            sys.exit(r.returncode)


if __name__ == "__main__":
    main()
