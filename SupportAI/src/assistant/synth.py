from __future__ import annotations
import json, random, argparse, os
from typing import List, Dict
from .formatting import CATEGORIES, PRODUCTS, PRIORITIES, SENTIMENTS, format_prompt, TARGET_JSON_SCHEMA

random.seed(42)

NAMES = ["Alex", "Sam", "Taylor", "Jordan", "Priya", "Omar", "Lee", "Chen", "Ivy"]
ISSUES = {
    "billing": [
    "I was charged twice",
    "My invoice shows the wrong amount",
    "Refund still not processed",
    ],
    "login": [
    "2FA code never arrives",
    "Password reset link expired",
    "Locked out after update",
    ],
    "bug": [
    "Export to CSV crashes",
    "Page goes blank on save",
    "Mobile app freezes on login",
    ],
    "feature_request": [
    "Need dark mode",
    "Please add SSO with Okta",
    "Custom roles for teams",
    ],
    "shipping": [
    "Package stuck in transit",
    "Wrong item received",
    "Return label not working",
    ],
}

TEMPLATES = [
"Hi team, I'm {name} on the {product} plan. {issue}. This is really {feeling}!",
"Hello, {issue}. I'm using {product} and it's getting {feeling}. Please fix.",
"My company is on {product}. {issue}. Priority should be {priority}.",
"I tried support but no luck: {issue}. Using {product}.",
]

FEELINGS = {
"negative": ["frustrating", "unacceptable", "blocking", "bad"],
"neutral": ["inconvenient", "annoying"],
"positive": ["okay now", "resolved after retry"],
}

def make_example() -> Dict:
    category = random.choice(CATEGORIES)
    product = random.choice(PRODUCTS)
    sentiment = random.choices(SENTIMENTS, weights=[0.6, 0.3, 0.1])[0]
    priority = random.choices(PRIORITIES, weights=[0.3, 0.4, 0.2, 0.1])[0]
    issue = random.choice(ISSUES[category])
    name = random.choice(NAMES)
    feeling = random.choice(FEELINGS[sentiment])

    text = random.choice(TEMPLATES).format(
    name=name, product=product, issue=issue, feeling=feeling, priority=priority
    )

    record = {
        "message": text,
        "expected": {
        "category": category,
        "priority": priority,
        "product": product,
        "sentiment": sentiment,
        "summary": issue,
        },
    }
    return record

def to_sft(example: Dict) -> Dict:
    prompt = format_prompt(example["message"])
    response = json.dumps(example["expected"], ensure_ascii=False)
    return {"text": prompt + response}

def main(n_train: int, n_val: int, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    train = [to_sft(make_example()) for _ in range(n_train)]
    val = [to_sft(make_example()) for _ in range(n_val)]
    with open(os.path.join(out_dir, "synthetic_train.jsonl"), "w", encoding="utf-8") as f:
        for ex in train: f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, "synthetic_val.jsonl"), "w", encoding="utf-8") as f:
        for ex in val: f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Wrote {len(train)} train and {len(val)} val to {out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=5000)
    ap.add_argument("--n_val", type=int, default=500)
    ap.add_argument("--out_dir", type=str, default="data")
    args = ap.parse_args()
    main(args.n_train, args.n_val, args.out_dir)


