#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import random
import argparse
import time
from typing import Optional, Dict, Any, List
import torch

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig
from peft import PeftModel

BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
MATHRM_RE = re.compile(r"\\mathrm\{([^}]*)\}")
FINAL_RE = re.compile(
    r"(?mi)^\s*(?:final(?:\s*answer)?|answer|ans)\s*[:=]\s*(.+?)\s*$")

FORMAT_ONLY_DEMO = """
Example format:
1) <brief step>
2) <brief step>
...
Final: <single-line final answer>
""".strip()

MATH_DEMOS = [
    {
        "problem": "Solve for x: 2x + 3 = 11.",
        "steps": "1) 2x = 11 - 3 = 8\n2) x = 8 / 2 = 4\nFinal: x = 4"
    },
    {
        "problem": "Differentiate f(x) = sin(x^2) with respect to x.",
        "steps": "1) Let u = x^2, then d/du[sin u] = cos u\n2) du/dx = 2x\n3) f'(x) = cos(x^2) * 2x\nFinal: f'(x) = 2x*cos(x^2)"
    }
]

SOLVE_INSTR = (
    "Solve ONLY the given problem. Do NOT copy any example answers. "
    "Show concise steps (≤8 lines). End with one line exactly starting with 'Final:'."
)

CORRECT_INSTR = (
    "The correct final answer is '{ANS}'. Give a new set of verifiable steps that logically lead to "
    "'Final: {ANS}'. Do NOT output any other final value. End with exactly 'Final: {ANS}'."
)

FINAL_ONLY_INSTR = (
    "Output ONLY the final answer in one line exactly as: Final: <answer>. "
    "Do NOT print any other text."
)


def extract_boxed(ans: str) -> str:
    if not isinstance(ans, str):
        return ""
    m = BOXED_RE.search(ans)
    final = m.group(1) if m else ans.strip()
    final = MATHRM_RE.sub(r"\\1", final)
    final = final.replace("\\,", "").replace(
        "\\;", "").replace("\\!", "").replace("\\:", "")
    return final.strip()


def extract_final_from_text(txt: str) -> str:
    if not isinstance(txt, str):
        return ""
    matches = FINAL_RE.findall(txt) if txt else []
    return matches[-1].strip() if matches else ""


def normalize_token(s: str) -> str:
    s = s.strip().replace("\\frac", "frac").replace(
        "\\sqrt", "sqrt").replace("\\pi", "pi").replace(" ", "")
    if s.endswith("."):
        s = s[:-1]
    return s


def try_float(x: str):
    try:
        return float(x)
    except Exception:
        return None


def try_sympy_equal(a: str, b: str):
    try:
        import sympy as sp
    except Exception:
        return None
    try:
        A = sp.simplify(sp.sympify(a))
        B = sp.simplify(sp.sympify(b))
        return bool(sp.simplify(A - B) == 0)
    except Exception:
        return None


def finals_equal(pred: str, gold: str) -> bool:
    if not pred or not gold:
        return False
    ap, ag = normalize_token(pred), normalize_token(gold)
    if ap == ag:
        return True
    fp, fg = try_float(ap), try_float(ag)
    if fp is not None and fg is not None:
        return abs(fp - fg) <= 1e-6
    res = try_sympy_equal(ap, ag)
    return bool(res) if isinstance(res, bool) else False


def build_preamble(fewshot: str):
    if fewshot == "none":
        return ""
    if fewshot == "format":
        return f"Follow this output format:\n{FORMAT_ONLY_DEMO}\n\n"
    # "math"
    parts = ["Here are two EXAMPLES of the expected style. Do NOT copy their answers."]
    for j, ex in enumerate(MATH_DEMOS, 1):
        parts.append(
            f"Example {j} — Problem:\n{ex['problem']}\nSolution:\n{ex['steps']}")
    return "\n\n".join(parts) + "\n\n"


def build_user_prompt(problem: str, fewshot: str) -> str:
    pre = build_preamble(fewshot)
    return f"{pre}{SOLVE_INSTR}\n\nProblem:\n{problem.strip()}\n\nResponse:"


def build_corrective_prompt(problem: str, correct_ans: str, fewshot: str) -> str:
    pre = build_preamble(fewshot)
    spec = CORRECT_INSTR.replace("{ANS}", correct_ans)
    return f"{pre}{spec}\n\nProblem:\n{problem.strip()}\n\nResponse:"


def build_final_only_prompt(problem: str) -> str:
    return f"{FINAL_ONLY_INSTR}\n\nProblem:\n{problem.strip()}\n\nFinal:"


def maybe_apply_chat_template(tok, user_prompt: str):
    messages = [
        {"role": "system", "content": "You are a careful, concise math tutor. Show ≤8 steps and finish with 'Final:'."},
        {"role": "user", "content": user_prompt}
    ]
    try:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return user_prompt + "\n"


def set_seed_all(seed: int):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(base_model: str, lora_dir: Optional[str], load_in_4bit: bool = True, trust_remote_code: bool = True):
    quant = None
    if load_in_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    tok = AutoTokenizer.from_pretrained(
        base_model, use_fast=True, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=trust_remote_code,
        quantization_config=quant
    )
    if lora_dir:
        model = PeftModel.from_pretrained(model, lora_dir)
    model.eval()
    return tok, model


@torch.no_grad()
def generate(model, tok, prompt_text: str, max_new_tokens=256, temperature=0.0, top_p=0.95, top_k=50):
    text = maybe_apply_chat_template(tok, prompt_text)
    inputs = tok(text, return_tensors="pt").to(model.device)

    gen_cfg = GenerationConfig.from_model_config(model.config)
    gen_cfg.do_sample = temperature > 0.0
    gen_cfg.temperature = float(temperature)
    gen_cfg.top_p = float(top_p)
    gen_cfg.top_k = int(top_k)

    gen = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        generation_config=gen_cfg,
        use_cache=True,
        eos_token_id=tok.eos_token_id,
    )
    return tok.decode(gen[0], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--subset", default="default",
                    choices=["default", "extended"])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--tries", type=int, default=3,
                    help="attempts in first pass before correction")
    ap.add_argument("--fewshot", default="format",
                    choices=["none", "format", "math"])
    ap.add_argument("--out-dir", default="outputs/star_round1_test_v3")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    report_path = os.path.join(args.out_dir, "report.jsonl")
    alpaca_path = os.path.join(args.out_dir, "star_train_round1.jsonl")

    print(f"[info] loading dataset open-r1/OpenR1-Math-220k ({args.subset})…")
    ds = load_dataset("open-r1/OpenR1-Math-220k", args.subset, split="train")

    idx = list(range(len(ds)))
    set_seed_all(args.seed)
    random.shuffle(idx)
    idx = idx[:args.n]

    print(f"[info] sampling {len(idx)} problems (seed={args.seed})")
    tok, model = load_model_and_tokenizer(
        args.base_model, args.lora, load_in_4bit=True, trust_remote_code=True)

    kept_for_train: List[Dict[str, Any]] = []
    n_ok = n_fix = n_probe_fix = 0
    with open(report_path, "w", encoding="utf-8") as frep:
        for k, i in enumerate(idx, 1):
            ex = ds[i]
            problem, gold_ans = ex.get("problem", ""), ex.get("answer", "")
            gold_final = extract_boxed(gold_ans)

            # multi-try first pass
            pred_final1 = ""
            out1 = ""
            ok1 = False
            for t in range(args.tries):
                out1 = generate(model, tok, build_user_prompt(
                    problem, args.fewshot), max_new_tokens=args.max_new_tokens, temperature=args.temperature)
                pred_final1 = extract_final_from_text(out1)
                ok1 = finals_equal(pred_final1, gold_final)
                # if ok1 or pred_final1:  # keep if at least produced a Final; otherwise retry
                #     break
                if ok1:
                    break
            # Final-only probe if no Final yet
            probe_out = ""
            probe_final = ""
            probe_ok = False
            if not pred_final1:
                probe_out = generate(model, tok, build_final_only_prompt(
                    problem), max_new_tokens=64, temperature=0.0)
                probe_final = extract_final_from_text(probe_out)
                probe_ok = finals_equal(probe_final, gold_final)

            # corrective pass
            out2 = ""
            pred_final2 = ""
            ok2 = False
            if not ok1:
                corr_prompt = build_corrective_prompt(
                    problem, gold_final, args.fewshot)
                out2 = generate(
                    model, tok, corr_prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
                pred_final2 = extract_final_from_text(out2)
                ok2 = finals_equal(pred_final2, gold_final)

            row = {
                "idx": int(i),
                "problem": problem,
                "gold_final": gold_final,
                "gen_solution": out1,
                "gen_final": pred_final1,
                "gen_ok": bool(ok1),
                "probe_solution": probe_out,
                "probe_final": probe_final,
                "probe_ok": bool(probe_ok),
                "corr_solution": out2,
                "corr_final": pred_final2,
                "corr_ok": bool(ok2),
            }
            frep.write(json.dumps(row, ensure_ascii=False) + "\n")

            if ok1:
                kept_for_train.append({
                    "instruction": "Solve the problem with concise steps and end with a single line starting 'Final:'.",
                    "input": problem,
                    "output": out1.strip() if out1.strip() else f"Final: {gold_final}"
                })
                n_ok += 1
            elif ok2:
                kept_for_train.append({
                    "instruction": "Solve the problem with concise steps and end with a single line starting 'Final:'.",
                    "input": problem,
                    "output": out2.strip() if out2.strip() else f"Final: {gold_final}"
                })
                n_fix += 1
            elif probe_ok:
                out3 = generate(model, tok, build_corrective_prompt(
                    problem, gold_final, args.fewshot), max_new_tokens=args.max_new_tokens, temperature=args.temperature)
                kept_for_train.append({
                    "instruction": "Solve the problem with concise steps and end with a single line starting 'Final:'.",
                    "input": problem,
                    "output": out3.strip() if out3.strip() else f"Final: {gold_final}"
                })
                n_probe_fix += 1

            print(
                f"[{k}/{len(idx)}] ok={n_ok} | fixed={n_fix} | probe_fix={n_probe_fix} | kept={len(kept_for_train)}")

    with open(alpaca_path, "w", encoding="utf-8") as f:
        for r in kept_for_train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n[done] report: {report_path}")
    print(f"[done] next-round SFT data (alpaca): {alpaca_path}")
    print(
        f"[stats] ok={n_ok} | corrected ok={n_fix} | probe_fix={n_probe_fix} | kept total={len(kept_for_train)}")


if __name__ == "__main__":
    main()
