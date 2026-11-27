"""Closed-book evaluation: for each test document, attach its knowledge adapter and answer the dataset
questions without the document.

Configurations (same base model):
  k   base + LoRA_k              (untrained student)
  ks  base + LoRA_k + LoRA_s     (trained student)

  CUDA_VISIBLE_DEVICES=0 python rodel/evaluate.py --set hotpotqa --knowledge-tag t3 --student runs/seq/ckpt/s3/epoch1/s \
      --tag seq_r3 --run-dir runs/seq --shard 0 --nshards 8

Writes {run-dir}/preds/{tag}/{set}.shard{i}.jsonl, one row per document:
  {"id", "set", "official": [{"id", "question", "answers", "extra", "pred_<c>", "f1_<c>", "valid_<c>", "ntok_<c>"}]}
(f1 = token F1 against the closest reference; finished units are skipped on restart).
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (DEFAULT_MAX_NEW_TOKENS, DEFAULT_MODEL, INFER_SAMPLING, best_f1, chat_frame, knowledge_dir,
                    load_units, parse_answer, preds_path, read_jsonl, student_prompt)

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--set", required=True)
ap.add_argument("--knowledge-tag", required=True, help="knowledge adapters to attach ({run-dir}/knowledge/{tag}/{set}/)")
ap.add_argument("--student", default="", help="student adapter directory (LoRA_s); without it only configuration k is run")
ap.add_argument("--tag", required=True, help="name of the predictions")
ap.add_argument("--configs", default="k,ks")
ap.add_argument("--model", default=DEFAULT_MODEL)
ap.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS, help="length limit of an answer")
ap.add_argument("--data-dir", default="data")
ap.add_argument("--run-dir", required=True)
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--limit", type=int, default=0, help="only the first N units of the shard")
a = ap.parse_args()

import torch
from peft import PeftModel, set_peft_model_state_dict
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

dev = "cuda"
units = load_units(a.data_dir, a.set)[a.shard::a.nshards]
if a.limit:
    units = units[:a.limit]
out = preds_path(a.run_dir, a.tag, a.set, a.shard)
os.makedirs(os.path.dirname(out), exist_ok=True)
done = set(r["id"] for r in read_jsonl(out)) if os.path.exists(out) else set()
weights = lambda u: os.path.join(knowledge_dir(a.run_dir, a.knowledge_tag, a.set, u["id"]), "adapter_model.safetensors")
todo = [u for u in units if u["id"] not in done and os.path.exists(weights(u))]
print(f"{a.set} shard {a.shard}/{a.nshards}: {len(units)} units, {len(todo)} to do", flush=True)
if not todo:
    sys.exit()

tok = AutoTokenizer.from_pretrained(a.model)
pad = tok.pad_token_id or tok.eos_token_id
eos = tok.eos_token_id
enc = lambda t: tok(t, add_special_tokens=False)["input_ids"]
base = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(dev)
model = PeftModel.from_pretrained(base, os.path.dirname(weights(todo[0])), adapter_name="k")
configs = a.configs.split(",")
if a.student:
    model.load_adapter(a.student, adapter_name="s")
else:
    configs = [c for c in configs if c == "k"]
model.eval()
for p in model.parameters():
    p.requires_grad = False
HEAD, TAIL = chat_frame(tok)
ADAPTERS = {"k": ["k"], "ks": ["k", "s"]}


@torch.no_grad()
def answer(questions):
    outs = []
    for s in range(0, len(questions), 16):
        chunk = [enc(HEAD + student_prompt(q["question"]) + TAIL) for q in questions[s:s + 16]]
        L = max(len(p) for p in chunk)
        ids = torch.full((len(chunk), L), pad, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, p in enumerate(chunk):
            ids[i, L - len(p):] = torch.tensor(p)
            mask[i, L - len(p):] = 1
        torch.manual_seed(7 + s)
        g = model.generate(input_ids=ids.to(dev), attention_mask=mask.to(dev), max_new_tokens=a.max_new_tokens,
                           do_sample=True, pad_token_id=pad, eos_token_id=eos, **INFER_SAMPLING)
        outs += [tok.decode(x[L:], skip_special_tokens=True) for x in g]
    return outs


for k, u in enumerate(todo):
    t0 = time.time()
    res = set_peft_model_state_dict(model, load_file(weights(u), device=dev), adapter_name="k")
    assert not res.unexpected_keys
    official = [{"id": q["id"], "question": q["question"], "answers": q["answers"], "extra": q.get("extra", {})}
                for q in u["official"]]
    for c in configs:
        model.base_model.set_adapter(ADAPTERS[c])
        for p in model.parameters():
            p.requires_grad = False
        for o, text in zip(official, answer(u["official"])):
            valid, _, pred = parse_answer(text)
            o[f"pred_{c}"] = pred
            o[f"f1_{c}"] = best_f1(pred, o["answers"])
            o[f"valid_{c}"] = int(valid)
            o[f"ntok_{c}"] = len(enc(text))
    row = {"id": u["id"], "set": a.set, "official": official, "sec": round(time.time() - t0, 1)}
    with open(out, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[{k + 1}/{len(todo)}] {u['id']} "
          + " ".join(f"{c} {100 * sum(o[f'f1_{c}'] for o in official) / len(official):.1f}" for c in configs)
          + f" ({row['sec']}s)", flush=True)
