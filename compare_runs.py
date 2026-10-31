import json

with open("logs/runs.jsonl") as f:
    runs = [json.loads(line) for line in f]

runs.sort(key=lambda r: r["accuracy"], reverse=True)

print(f"{'run_id':<18}{'C':>8}{'max_iter':>10}{'accuracy':>10}{'time':>8}")
for r in runs:
    print(
        f"{r['run_id']:<18}{r['C']:>8}{r['max_iter']:>10}"
        f"{r['accuracy']:>10}{r['train_time_sec']:>8}"
    )
