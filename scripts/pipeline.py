import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
import yaml


def sh(cmd: str):
    print(f"[cmd] {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True)
    if r.returncode != 0:
        raise RuntimeError(f"Command failed ({r.returncode}): {cmd}")


def load_yaml(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yaml(obj, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)


def load_json(p: Path, default):
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return default


def save_json(obj, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def rt(r): return f"{r:02d}"
def now(): return datetime.now().strftime("%Y%m%d_%H%M%S")


def make_train_cfg_from_template(tpl_path: Path, out_path: Path,
                                 base_jsonl: str, lora_prev: str, out_run: str):
    cfg = load_yaml(tpl_path)
    # 覆寫三個欄位（其餘維持你的模板設定）
    # 1) datasets[0].path -> 本輪要吃的資料
    if "datasets" in cfg and cfg["datasets"]:
        cfg["datasets"][0]["path"] = base_jsonl
    else:
        cfg["datasets"] = [{"path": base_jsonl, "type": "alpaca"}]
    # 2) lora_model_dir -> 指向上一輪的輸出（第一輪可保留模板的或設為空/初始LoRA）
    cfg["lora_model_dir"] = lora_prev
    # 3) output_dir -> 本輪的新輸出
    cfg["output_dir"] = out_run
    dump_yaml(cfg, out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/star.yml")
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = load_yaml(cfg_path)
    if args.rounds is not None:
        cfg["rounds"] = int(args.rounds)
    if args.seed is not None:
        cfg["seed"] = int(args.seed)

    runs_dir = Path(cfg["paths"]["runs_dir"])
    runs_dir.mkdir(parents=True, exist_ok=True)
    outs_dir = Path(cfg["paths"]["outputs_dir"])
    outs_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(cfg["paths"]["data_dir"])
    data_dir.mkdir(parents=True, exist_ok=True)
    state_path = Path(cfg["paths"]["state_file"])
    state_path.parent.mkdir(parents=True, exist_ok=True)

    model_base = cfg["model"]["base"]
    lora_tag = cfg["model"]["lora_tag"]     # e.g. "s2_easy"
    out_tag = cfg["model"]["out_tag"]      # e.g. "q1_5b"

    # 狀態
    state = load_json(state_path, {"round": 0, "history": []}) if args.resume else {
        "round": 0, "history": []}
    start_r = state["round"] + 1

    # active 初始
    active = Path(cfg["data"]["active_path"])
    active.parent.mkdir(parents=True, exist_ok=True)
    if not active.exists():
        active.write_text("", encoding="utf-8")

    tpl_cfg = Path(cfg["paths"]["train_cfg_template"])
    cfg_dir = Path(cfg["paths"]["train_cfg_dir"])

    total = int(cfg["rounds"])
    seed = int(cfg["seed"])

    for r in range(start_r, total + 1):
        print(f"\n=== STaR Round {r}/{total} ===")

        # ---- Train：每輪產生一份 s2_easy_{r}.yml 並開練 ----
        # 本輪訓練吃「上一輪 merge 後的檔案」，也就是 data/star_R1_{r:02d}.jsonl
        train_data_this_round = str(data_dir / f"star_R1_{rt(r)}.jsonl")
        # 上一輪的 LoRA 目錄（第一輪可由模板決定）
        lora_prev = str(
            runs_dir / f"{lora_tag}_{r-1}") if r > 1 else load_yaml(tpl_cfg).get("lora_model_dir", "")
        out_run = str(runs_dir / f"{lora_tag}_{r}")   # 本輪輸出

        per_round_cfg = cfg_dir / f"{lora_tag}_{rt(r)}.yml"
        make_train_cfg_from_template(
            tpl_path=tpl_cfg,
            out_path=per_round_cfg,
            base_jsonl=train_data_this_round,   # datasets[0].path
            lora_prev=lora_prev,                # lora_model_dir
            out_run=out_run                     # output_dir
        )

        train_cmd_tpl = cfg["commands"].get("train_cmd", "")
        if train_cmd_tpl:
            sh(train_cmd_tpl.format(train_cfg=str(per_round_cfg)))

        # ---- Test：輸出到 outputs/<out_tag>/s{seed}_{r}/ ----
        test_out_dir = outs_dir / f"{out_tag}/s{seed}_{rt(r)}"
        test_out_dir.mkdir(parents=True, exist_ok=True)

        # 這裡推理用本輪的 LoRA 目錄（out_run）
        test_cmd = cfg["commands"]["test_cmd"].format(
            base=model_base, lora_dir=out_run,
            subset=cfg["test"]["subset"], n=cfg["test"]["n"], seed=seed,
            max_new_tokens=cfg["test"]["max_new_tokens"], temperature=cfg["test"]["temperature"],
            tries=cfg["test"]["tries"], fewshot=cfg["test"]["fewshot"],
            test_out_dir=str(test_out_dir)
        )
        sh(test_cmd)

        alpaca_in = test_out_dir / "star_train_round1.jsonl"
        if not alpaca_in.exists():
            raise FileNotFoundError(f"[test] missing: {alpaca_in}")

        # ---- Verify：outputs/star_round{r}.jsonl ----
        verify_out = outs_dir / f"star_round{rt(r)}.jsonl"
        sh(cfg["commands"]["verify_cmd"].format(
            alpaca_in=str(alpaca_in), verify_out=str(verify_out)
        ))
        if not verify_out.exists():
            raise FileNotFoundError(f"[verify] missing: {verify_out}")

        # ---- Merge：data/star_R1_{r+1}.jsonl （把 active + verify 合併去重）----
        next_r = r + 1
        merged_out = data_dir / f"star_R1_{rt(next_r)}.jsonl"
        merge_inputs = []
        if active.exists() and active.stat().st_size > 0:
            merge_inputs.append(str(active))
        merge_inputs.append(str(verify_out))

        sh(cfg["commands"]["merge_cmd"].format(
            merged_out=str(merged_out),
            merge_inputs=" ".join(merge_inputs)
        ))
        if not merged_out.exists():
            raise FileNotFoundError(f"[merge] missing: {merged_out}")

        # 更新 active 與狀態
        cfg["data"]["active_path"] = str(merged_out)
        dump_yaml(cfg, cfg_path)
        active = merged_out

        state["round"] = r
        state["history"].append({
            "round": r,
            "seed": seed,
            "lora_prev": lora_prev,
            "lora_now": out_run,
            "test_dir": str(test_out_dir),
            "verify_out": str(verify_out),
            "active_after_merge": str(merged_out),
            "timestamp": now()
        })
        save_json(state, state_path)

        print(
            f"[done] r{rt(r)} → verify: {verify_out} | merged→ {merged_out} | lora: {out_run}")

    print("\nAll rounds complete 🎉")


if __name__ == "__main__":
    main()
