"""Distillation: one knowledge adapter (LoRA_k) per document, fitted on the teacher's answer set of that document.

Training pairs are (question without the document -> the teacher's full output, i.e. reasoning trace + answer).
The prompt is encoded by the frozen base model and the adapter is active on the target tokens only; the loss
is the token-mean cross-entropy on the target. Every document starts from a freshly initialised adapter.

  CUDA_VISIBLE_DEVICES=0 python rodel/distill.py --set hotpotqa --tag t0 --run-dir runs/seq --shard 0 --nshards 8

Reads  {run-dir}/answers/{tag}/{set}.shard{i}.jsonl
Writes {run-dir}/knowledge/{tag}/{set}/{unit id}/   (PEFT adapter)
       {run-dir}/knowledge/{tag}/{set}.shard{i}.jsonl   (one row per finished unit; finished units are skipped on restart)
"""
import argparse
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn as nn

torch.backends.cuda.enable_cudnn_sdp(False)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (DEFAULT_MAX_NEW_TOKENS, DEFAULT_MODEL, answers_path, ce_loss, chat_frame, distill_log_path,
                    distill_logits, fix_linear_attn_cache, knowledge_dir, load_units, lora_targets, read_jsonl,
                    student_prompt)


def reset_lora(model):
    for m in model.modules():
        if hasattr(m, "lora_A") and isinstance(m.lora_A, nn.ModuleDict):
            for k in m.lora_A:
                nn.init.kaiming_uniform_(m.lora_A[k].weight, a=math.sqrt(5))
            for k in m.lora_B:
                nn.init.zeros_(m.lora_B[k].weight)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True)
    ap.add_argument("--tag", required=True, help="answer sets to distill; also names the knowledge adapters")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS, help="length limit used for synthesis")
    ap.add_argument("--max-target", type=int, default=0,
                    help="truncate longer targets to this many tokens (0 = --max-new-tokens + 76)")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="only the first N units of the shard")
    a = ap.parse_args()
    dev = "cuda"
    max_target = a.max_target or a.max_new_tokens + 76

    units = load_units(a.data_dir, a.set)[a.shard::a.nshards]
    if a.limit:
        units = units[:a.limit]
    answers = {r["id"]: r["pairs"] for r in read_jsonl(answers_path(a.run_dir, a.tag, a.set, a.shard))}
    log = distill_log_path(a.run_dir, a.tag, a.set, a.shard)
    done = set(r["id"] for r in read_jsonl(log)) if os.path.exists(log) else set()
    todo = [u for u in units if u["id"] not in done]
    print(f"{a.set} shard {a.shard}/{a.nshards}: {len(units)} units, {len(todo)} to do", flush=True)
    if not todo:
        return

    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    pad = tok.pad_token_id or tok.eos_token_id
    enc = lambda t: tok(t, add_special_tokens=False)["input_ids"]
    head, tail = chat_frame(tok)
    base = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(dev)
    fix_linear_attn_cache(base)
    model = get_peft_model(base, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05,
                                            target_modules=lora_targets(base)))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()                 # adapter weights are trained in fp32, the base model stays bf16
    trainable = [p for p in model.parameters() if p.requires_grad]

    for k, u in enumerate(todo):
        t0 = time.time()
        pairs = answers.get(u["id"], [])
        reset_lora(model)
        prompts, targets, truncated = [], [], 0
        for p in pairs:
            prompts.append(enc(head + student_prompt(p["q"]) + tail))
            y = enc(p["full"])
            if len(y) > max_target:
                y = y[:max_target]
                truncated += 1
            else:
                y = y + [tok.eos_token_id]
            targets.append(y)
        opt = torch.optim.AdamW(trainable, lr=a.lr, weight_decay=0.01)
        rng = random.Random(1234)
        losses = []
        model.train()
        for ep in range(a.epochs):
            order = list(range(len(targets)))
            rng.shuffle(order)
            total, nb = 0.0, 0
            for s in range(0, len(order), a.batch):
                idx = order[s:s + a.batch]
                logits, y_ids, y_mask = distill_logits(model, [prompts[i] for i in idx], [targets[i] for i in idx], pad, dev)
                loss = ce_loss(logits, y_ids, y_mask)
                if not torch.isfinite(loss):
                    opt.zero_grad()
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
                opt.zero_grad()
                total += loss.item()
                nb += 1
            losses.append(total / max(nb, 1))
        model.eval()

        d = knowledge_dir(a.run_dir, a.tag, a.set, u["id"])
        os.makedirs(d, exist_ok=True)
        tmp = os.path.join(d, "adapter_model.safetensors.tmp")
        save_file({n: v.detach().to(torch.bfloat16).cpu().contiguous() for n, v in get_peft_model_state_dict(model).items()},
                  tmp, metadata={"format": "pt"})
        os.replace(tmp, os.path.join(d, "adapter_model.safetensors"))
        model.peft_config["default"].save_pretrained(d)
        row = {"id": u["id"], "n_pairs": len(pairs), "truncated": truncated,
               "loss_first": losses[0] if pairs else None, "loss_last": losses[-1] if pairs else None,
               "sec": round(time.time() - t0, 1)}
        with open(log, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        loss_txt = f"loss {row['loss_first']:.3f}->{row['loss_last']:.3f}" if pairs else "no pairs"
        print(f"[{k + 1}/{len(todo)}] {u['id']} pairs {len(pairs)} {loss_txt} {row['sec']}s", flush=True)


if __name__ == "__main__":
    main()
