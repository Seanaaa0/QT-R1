#!/usr/bin/env python3
# verify_star_integrity_relaxed.py
# Keeps items where:
#  - Last non-empty line is a single 'Final:' line (ignoring placeholder finals like '<...>')
#  - There are at least 2 non-empty step lines before Final
import argparse, json, re, sys
from pathlib import Path

FINAL_RE = re.compile(r"(?mi)^\s*final\s*:\s*(.+?)\s*$")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    args = ap.parse_args()

    p = Path(args.inp)
    if not p.exists():
        print(f"[error] not found: {p}", file=sys.stderr)
        sys.exit(2)

    total = kept = 0
    dropped = {"no_final":0,"final_not_last":0,"too_few_steps":0}

    with p.open("r", encoding="utf-8") as f, open(args.out, "w", encoding="utf-8") as fo:
        for line in f:
            if not line.strip(): continue
            total += 1
            ex = json.loads(line)
            out = ex.get("output","")
            lines = [l.rstrip() for l in out.splitlines()]
            while lines and lines[-1].strip() == "":
                lines.pop()
            if not lines:
                dropped["no_final"] += 1
                continue
            text = "\n".join(lines)
            matches = list(FINAL_RE.finditer(text))
            if not matches:
                dropped["no_final"] += 1
                continue
            # Select the last non-placeholder Final
            target = None
            for m in reversed(matches):
                val = m.group(1).strip()
                if "<" in val or ">" in val or "single-line" in val:
                    continue
                target = m
                break
            if target is None:
                # fall back to the last Final match
                target = matches[-1]
            fin_line_idx = text[:target.start()].count("\n")
            if fin_line_idx != len(lines)-1:
                dropped["final_not_last"] += 1
                continue
            # require at least 2 non-empty step lines
            steps = [s for s in lines[:fin_line_idx] if s.strip()]
            if len(steps) < 2:
                dropped["too_few_steps"] += 1
                continue
            fo.write(json.dumps(ex, ensure_ascii=False) + "\n")
            kept += 1

    print({"total": total, "kept": kept, "dropped": dropped, "out": args.out})

if __name__ == "__main__":
    main()
