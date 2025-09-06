import argparse, json, sys
from pathlib import Path

def read_jsonl(p):
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                yield json.loads(line)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("inputs", nargs="+")
    args = ap.parse_args()
    seen=set(); kept=[]
    for ip in args.inputs:
        p=Path(ip)
        if not p.exists():
            print(f"[warn] missing: {p}", file=sys.stderr); continue
        for row in read_jsonl(p):
            key=row.get("input") or row.get("problem") or json.dumps(row, ensure_ascii=False)
            if key in seen: continue
            seen.add(key); kept.append(row)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out,"w",encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False)+"\n")
    print({"merged": len(kept), "out": args.out})

if __name__=="__main__":
    main()
