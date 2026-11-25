"""LSTM(금리차 단독 A vs 유동성 추가 B)을 이상구간 정의별, 시드별로 다시 돌린다.
기존 train_eval_extended.py의 main()을 그대로 쓰되 PERIOD_PATH와 시드만 바꿔 호출한다.
정의: orig(커밋된 period_definition.json), A/B/C(analysis/anomaly/rerun_redefined.py의 재정의).
요약만 results_redefined/ 에 저장한다(예측 파일 등 중간 산출물은 임시 폴더에 두고 버린다)."""
import importlib.util
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis" / "anomaly"))
import rerun_redefined as rr  # noqa: E402

OUT = Path(__file__).resolve().parent / "results_redefined"
SCRIPTS = {
    "mmf": ROOT / "analysis/LSTM/lstm_mmf/train_eval_extended.py",
    "m2_demand_deposit": ROOT / "analysis/LSTM/lstm_m2_demand_deposit/train_eval_extended.py",
}
# 사용법: python rerun_redefined.py [시드,...] [정의,...(orig,A,B,C)] [모델,...(mmf,m2_demand_deposit)]
SEEDS = [int(s) for s in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["42", "43", "44"])]
ONLY_DEFS = sys.argv[2].split(",") if len(sys.argv) > 2 else None
ONLY_VARIANTS = sys.argv[3].split(",") if len(sys.argv) > 3 else None
TAG = "_".join(sys.argv[2:4]).replace(",", "") if len(sys.argv) > 2 else "all"  # 병렬 실행 시 결과 파일이 겹치지 않게


def regime_periods(tmp: Path) -> dict:
    daily = pd.read_csv(rr.DATA, parse_dates=["date"]).dropna(subset=["FX_rate", "policy_spread"]).sort_values("date").reset_index(drop=True)
    daily["rc"] = daily["FX_rate"].rolling(rr.WINDOW).corr(daily["policy_spread"])
    daily = daily.dropna(subset=["rc"]).reset_index(drop=True)
    paths = {"orig": ROOT / "analysis/anomaly/period_definition.json"}
    for name, cond in rr.DEFS.items():
        a = rr.build_blocks(daily, cond(daily["rc"]))
        blocks = [{"start": str(g["date"].min().date()), "end": str(g["date"].max().date()), "days": len(g)} for _, g in a.groupby("block_start")]
        p = tmp / f"{name.split('_')[0]}.json"
        p.write_text(json.dumps({"anomaly_blocks_for_analysis": blocks}), encoding="utf-8")
        paths[name.split("_")[0]] = p
    return paths


def main():
    OUT.mkdir(exist_ok=True)
    rows = []
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        periods = regime_periods(td)
        periods = {k: v for k, v in periods.items() if ONLY_DEFS is None or k in ONLY_DEFS}
        for variant, script in SCRIPTS.items():
            if ONLY_VARIANTS is not None and variant not in ONLY_VARIANTS:
                continue
            spec = importlib.util.spec_from_file_location(f"lstm_{variant}", script)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            data_path = mod.DATA_PATH
            # 진행 상황을 로그에 남기기 위한 래퍼(계산 결과에는 영향 없음)
            _tune, _eval, cnt = mod.tune_hyperparams, mod.evaluate_block, [0]

            def tune(*a, _f=_tune, **k):
                t = time.time()
                r = _f(*a, **k)
                print(f"  tune done {round(time.time() - t)}s", flush=True)
                return r

            def ev(*a, _f=_eval, **k):
                cnt[0] += 1
                if cnt[0] % 100 == 0:
                    print(f"  evaluated blocks {cnt[0]}", flush=True)
                return _f(*a, **k)

            mod.tune_hyperparams, mod.evaluate_block = tune, ev
            for defname, pp in periods.items():
                for seed in SEEDS:
                    run_dir = td / variant / defname / str(seed)
                    run_dir.mkdir(parents=True)
                    mod.DATA_PATH, mod.OUT_DIR, mod.PERIOD_PATH = data_path, run_dir, pp
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    t0 = time.time()
                    cnt[0] = 0
                    print(f"start {variant} {defname} seed={seed}", flush=True)
                    try:
                        mod.main()
                        r = json.loads((run_dir / "results_extended.json").read_text(encoding="utf-8"))
                        row = {"variant": variant, "definition": defname, "seed": seed, "train_rows": r["train_rows"],
                               "blocks": r["evaluated_blocks"], "rmse_A": r["weighted_rmse_model_a"], "rmse_B": r["weighted_rmse_model_b"],
                               "mae_A": r["weighted_mae_model_a"], "mae_B": r["weighted_mae_model_b"], "sec": round(time.time() - t0)}
                    except Exception as e:  # 블록이 모두 짧은 정의 등
                        row = {"variant": variant, "definition": defname, "seed": seed, "error": str(e)[:80], "sec": round(time.time() - t0)}
                    rows.append(row)
                    pd.DataFrame(rows).to_csv(OUT / f"lstm_runs_{TAG}.csv", index=False)  # 중간 저장
                    print(row, flush=True)
    df = pd.DataFrame(rows)
    ok = df.dropna(subset=["rmse_A"]) if "rmse_A" in df else df.iloc[0:0]
    if len(ok):
        ok = ok.assign(gain=ok["rmse_A"] - ok["rmse_B"])
        s = ok.groupby(["variant", "definition"]).agg(blocks=("blocks", "first"), rmse_A=("rmse_A", "mean"), rmse_B=("rmse_B", "mean"),
                                                       gain_mean=("gain", "mean"), gain_sd=("gain", "std"), B_better=("gain", lambda x: int((x > 0).sum())), runs=("gain", "size")).round(2)
        s.to_csv(OUT / f"lstm_summary_{TAG}.csv")
        print(s.to_string())


if __name__ == "__main__":
    main()
