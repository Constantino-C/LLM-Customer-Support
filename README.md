# Customer Support AI Assistant 🤝 

This project demonstrates how **fine-tuned language models (using LoRA methods)** can transform messy customer support feedback into **structured records**.  

Inneficiently handled customer feedback leads to:

- **Lost revenue**: frustrated customers churn quickly when problems aren’t solved.
- **Reputation damage**: negative experiences spread via media.
- **Slower productivity**: prodactivity falls when logged errors get buried in unstructured text.

In a businesses with many thousands of support messages/ customer feedback, extracting actionable insights from this data is critical. Manual processing is slow and prone to error. This system automates the process by classifying incoming messages into categories (e.g., *billing*, *login*, *bug*), assigning priorities, detecting sentiment, and summarising the issue.

By structuring support feedback into clean records, businesses can:

- Direct urgent issues immediately to the right teams.
- Monitor specific problems and errors.
- Quantify sentiment across different customer profiles.
- Feed structured data into dashboards and analytics pipelines.

---

## Project Features ✨

- **Generating synthetic and realistic data** simulating customer support scenarios.
- **LoRA methods** for fine-tuning of LLMs for domain-specific adaptation.
- **Schema-guided prompts** to guide structured outputs.
- **Evaluation pipeline** with validity checks and F1 scores.

---

## Quickstart

### ⚙️ Environment setup

conda create -n customer-support python=3.11 -y <br>
conda activate customer-support <br>
pip install -e . <br>

### 🛠️ Generate synthetic data

python -m SupportAI.synth --n_train 5000 --n_val 500 --out_dir data <br>

### 🎯 Fine-tune with LoRA

python -m SupportAI.train_lora --config configs/train.yaml<br>

### 📊 Evaluate the adapter

python -m SupportAI.eval_json --data data/synthetic_val_pairs.jsonl --adapter_path outputs/lora-adapter <br>

### 💻 Run the demo

python app.py --adapter_path outputs/lora-adapter<br>