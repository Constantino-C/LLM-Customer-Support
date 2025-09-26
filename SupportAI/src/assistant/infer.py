import argparse, yaml
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from .formatting import format_prompt


def load(base_model: str, adapter_path: str):
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.float32,low_cpu_mem_usage=True).to("cpu")
    model = PeftModel.from_pretrained(model, adapter_path).to("cpu")
    model.eval()
    return tokenizer, model


def predict(message: str, tokenizer: AutoTokenizer, model: PeftModel , max_new_tokens=256, temperature=0.1, top_p=0.9):
    prompt = format_prompt(message)
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=temperature>0, temperature=temperature, top_p=top_p
        )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
    # Return only the assistant JSON after the last [ASSISTANT]\n
    if "[ASSISTANT]\n" in text:
        text = text.split("[ASSISTANT]\n")[-1]
    return text.strip()


def main(config_path: str, max_new_tokens: int, temperature: float, top_p: float):

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    adapter_path = cfg['save_dir']
    base_model = cfg['base_model']

    tok, model = load(base_model, adapter_path)
    print("Type a support message, or Ctrl+C to exit.\n")
    message = input(">> ")
    out = predict(message, tok, model, max_new_tokens, temperature, top_p)
    print(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_path", type=str ,default="configs/train.yaml")
    ap.add_argument("--max_new_tokens", type=int ,default=256)
    ap.add_argument("--temperature", type=float ,default=0.1)
    ap.add_argument("--top_p", type=float ,default=0.9)
    args = ap.parse_args()
    main(args.config_path, args.max_new_tokens, args.temperature, args.top_p )

