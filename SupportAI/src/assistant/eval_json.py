import json, argparse
from sklearn.metrics import f1_score
from .infer import load, predict


FIELDS = ["category", "priority", "product", "sentiment"]

def main(data: str, adapter_path: str, base_model: str = "mistralai/Mistral-7B-Instruct-v0.2"):
    tok, model = load(base_model, adapter_path)
    y_true, y_pred, valid = {f: [] for f in FIELDS}, {f: [] for f in FIELDS}, []


total = 0
with open(data, "r", encoding="utf-8") as f:
for line in f:
ex = json.loads(line)
total += 1
msg = ex["message"]
gold = ex["expected"]
out = predict(msg, tok, model)
try:
pj = json.loads(out)
valid.append(1)
except Exception:
pj = {k: None for k in FIELDS}
valid.append(0)
for k in FIELDS:
y_true[k].append(gold.get(k))
y_pred[k].append(pj.get(k))


print(f"Samples: {total}")
print(f"JSON validity: {sum(valid)/len(valid):.3f}")
for k in FIELDS:
# Remove None rows for f1
pairs = [(t, p) for t, p in zip(y_true[k], y_pred[k]) if p is not None]
if pairs:
yt, yp = zip(*pairs)
print(f"{k} F1: {f1_score(yt, yp, average='macro'):.3f}")
else:
print(f"{k} F1: 0.000 (no valid preds)")


if __name__ == "__main__":
ap = argparse.ArgumentParser()
ap.add_argument("--data", type=str, default="data/synthetic_val_pairs.jsonl")
ap.add_argument("--adapter_path", type=str, required=True)
ap.add_argument("--base_model", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
args = ap.parse_args()
main(args.data, args.adapter_path, args.base_model)