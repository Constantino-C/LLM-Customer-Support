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
    "Please add functionality of extra users",
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
"Hello, {issue}. I'm using {product} and it's {feeling}. Please fix.",
"My company is on {product}. {issue}. Priority should be {priority}.",
"I tried support but no luck: {issue}. Using {product}.",
]

FEELINGS = {
"negative": ["frustrating", "unacceptable", "blocking", "bad"],
"neutral": ["inconvenient", "annoying", "not as expected", "mediocre"],
"positive": ["just ok", "resolved after retry", "improved", "better than before"],
}

PRIORITY_WEIGHTS = {
    ("billing", "negative"):  [0.05, 0.25, 0.45, 0.25],
    ("billing", "neutral"):   [0.15, 0.50, 0.30, 0.05],
    ("billing", "positive"):  [0.40, 0.50, 0.09, 0.01],

    ("login", "negative"):    [0.05, 0.25, 0.50, 0.20],
    ("login", "neutral"):     [0.20, 0.55, 0.20, 0.05],
    ("login", "positive"):    [0.50, 0.40, 0.09, 0.01],

    ("bug", "negative"):      [0.10, 0.30, 0.40, 0.20],
    ("bug", "neutral"):       [0.25, 0.50, 0.20, 0.05],
    ("bug", "positive"):      [0.50, 0.40, 0.08, 0.02],

    ("feature_request", "negative"): [0.50, 0.45, 0.05, 0.00],
    ("feature_request", "neutral"):  [0.60, 0.35, 0.05, 0.00],
    ("feature_request", "positive"): [0.70, 0.28, 0.02, 0.00],

    ("shipping", "negative"):  [0.10, 0.35, 0.35, 0.20],
    ("shipping", "neutral"):   [0.25, 0.50, 0.20, 0.05],
    ("shipping", "positive"):  [0.50, 0.40, 0.09, 0.01],
}

SENTIMENT_WEIGHTS = {
    "billing" :  [0.6,0.3,0.1],
    "login" :   [0.4, 0.4, 0.2],
    "bug" :  [0.6, 0.3, 0.1],
    "feature_request" :  [0.15, 0.45, 0.4],
    "shipping" : [0.55, 0.35, 0.1]
}

def make_example() -> Dict:
    category = random.choice(CATEGORIES)
    product = random.choice(PRODUCTS)
    sentiment_weights = SENTIMENT_WEIGHTS.get(category)
    sentiment = random.choices(SENTIMENTS, weights=sentiment_weights)[0]
    priority_weights = PRIORITY_WEIGHTS.get((category, sentiment))
    priority = random.choices(PRIORITIES, weights=priority_weights)[0]
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
    ap.add_argument("--n_train", type=int, default=1000)
    ap.add_argument("--n_val", type=int, default=200)
    ap.add_argument("--out_dir", type=str, default="data")
    args = ap.parse_args()
    main(args.n_train, args.n_val, args.out_dir)


