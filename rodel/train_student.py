"""Student update of RoDeL-Seq: train the student adapter (LoRA_s) with GRPO while the teacher is frozen.

The knowledge adapters of the training documents are prepared beforehand with the current teacher
(synthesize.py + distill.py) and stay fixed. Per step, each process takes one training document: its
knowledge adapter is attached, the student samples G answers per dataset question without the document,
and each answer is rewarded with its token F1 (a fixed negative reward if it is format-invalid). The
advantage is normalised within the G answers of a question; the loss is the advantage-weighted,
length-normalised log-likelihood of the sampled answers. Gradients are averaged over the processes.

  python -m torch.distributed.run --nproc_per_node 8 rodel/train_student.py \
      --sets train_hotpotqa train_musique --knowledge-tag t0 --run-dir runs/seq --out runs/seq/ckpt/s1

Units with "split": "val" are held out and used for validation only.
Writes {out}/epoch{e}/s (PEFT adapter) and {out}/log.jsonl (per-step statistics, validations).
"""
import argparse
import datetime
import json
import math
import os
import random
import sys
import time

import torch
import torch.distributed as dist
from safetensors.torch import load_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (DEFAULT_MAX_NEW_TOKENS, DEFAULT_MODEL, INFER_SAMPLING, INVALID_REWARD, best_f1, chat_frame,
                    fix_linear_attn_cache, knowledge_dir, left_pad, load_units, lora_targets, parse_answer, student_prompt)

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--sets", nargs="+", required=True, help="training unit files")
ap.add_argument("--knowledge-tag", required=True, help="knowledge adapters of the training documents")
ap.add_argument("--model", default=DEFAULT_MODEL)
ap.add_argument("--out", required=True)
ap.add_argument("--init-student", default="", help="continue from this LoRA_s directory instead of a new adapter")
ap.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS, help="length limit of a student answer")
ap.add_argument("--epochs", type=int, default=1)
ap.add_argument("--samples", type=int, default=4, help="G: sampled answers per dataset question")
ap.add_argument("--max-q", type=int, default=8, help="dataset questions used per document and epoch")
ap.add_argument("--lr", type=float, default=1e-5)
ap.add_argument("--rank", type=int, default=64)
ap.add_argument("--grad-budget", type=int, default=2048, help="padded tokens (batch x longest) per forward+backward chunk")
ap.add_argument("--val-every", type=int, default=28, help="also validate every N steps (0 = per epoch only)")
ap.add_argument("--limit", type=int, default=0, help="only the first N training and N validation units per set")
ap.add_argument("--max-steps", type=int, default=0, help="stop after this many steps (0 = full epochs)")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--data-dir", default="data")
ap.add_argument("--run-dir", required=True)
a = ap.parse_args()

dist.init_process_group("nccl", timeout=datetime.timedelta(hours=4))
R, W = dist.get_rank(), dist.get_world_size()
dev = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")
torch.cuda.set_device(dev)


def log(*x):
    if R == 0:
        print(f"[student {time.strftime('%m-%d %H:%M:%S')}]", *x, flush=True)


# ---------------- data ----------------
def adapter_dir(u):
    return knowledge_dir(a.run_dir, a.knowledge_tag, u["set"], u["id"])


train_units, val_units = [], []
for S in a.sets:
    rows = [u for u in load_units(a.data_dir, S) if os.path.exists(os.path.join(adapter_dir(u), "adapter_model.safetensors"))]
    tr = [u for u in rows if u.get("split", "train") == "train"]
    va = [u for u in rows if u.get("split") == "val"]
    if a.limit:
        tr, va = tr[:a.limit], va[:a.limit]
    train_units += tr
    val_units += va
    log(f"{S}: {len(tr)} training units ({sum(len(u['official']) for u in tr)} questions), {len(va)} validation units")

# ---------------- model: base + k (swapped per document) + s ----------------
from peft import LoraConfig, PeftModel, set_peft_model_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained(a.model)
pad = tok.pad_token_id or tok.eos_token_id
eos = tok.eos_token_id
enc = lambda t: tok(t, add_special_tokens=False)["input_ids"]
base = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(dev)
fix_linear_attn_cache(base)
model = PeftModel.from_pretrained(base, adapter_dir(train_units[0]), adapter_name="k")
model.add_adapter("s", LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.0, task_type="CAUSAL_LM",
                                  target_modules=lora_targets(base)))


def is_s(n):
    return ".lora_A.s." in n or ".lora_B.s." in n


def freeze():
    for n, p in model.named_parameters():
        p.requires_grad = is_s(n)


def to_fp32():
    for n, p in model.named_parameters():
        if is_s(n):
            p.data = p.data.float()


freeze()
to_fp32()
if a.init_student:
    res = set_peft_model_state_dict(model, load_file(os.path.join(a.init_student, "adapter_model.safetensors"), device=str(dev)),
                                    adapter_name="s")
    assert not res.unexpected_keys, res.unexpected_keys[:3]
    to_fp32()
    log(f"loaded LoRA_s from {a.init_student}")
s_params = [p for n, p in model.named_parameters() if is_s(n)]


def activate():
    """Student = base + k + s; only s is trainable."""
    model.base_model.set_adapter(["k", "s"])
    freeze()


def load_k(u):
    sd = load_file(os.path.join(adapter_dir(u), "adapter_model.safetensors"), device=str(dev))
    res = set_peft_model_state_dict(model, sd, adapter_name="k")
    assert not res.unexpected_keys, res.unexpected_keys[:3]


log(f"LoRA_s trainable parameters {sum(p.numel() for p in s_params) / 1e6:.1f}M")
opt = torch.optim.AdamW(s_params, lr=a.lr, weight_decay=0.0)
model.config.use_cache = True
model.eval()          # eval mode throughout: sampling and scoring use the same policy (no adapter dropout)

HEAD, TAIL = chat_frame(tok)


def student_ids(q):
    return enc(HEAD + student_prompt(q) + TAIL)


@torch.no_grad()
def sample(prompts, n, temperature, top_p, top_k, seed=None):
    """n completions per prompt (prompt-major) as token lists; the end-of-sequence token is kept."""
    outs = []
    for s in range(0, len(prompts), 8):
        chunk = [p for p in prompts[s:s + 8] for _ in range(n)]
        ids, mask = left_pad(chunk, pad, dev)
        if seed is not None:
            torch.manual_seed(seed + s)
        gen = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=a.max_new_tokens, do_sample=True,
                             temperature=temperature, top_p=top_p, top_k=top_k, pad_token_id=pad, eos_token_id=eos)
        for row in gen[:, ids.size(1):].tolist():
            y = []
            for t in row:
                if t == eos:
                    y.append(t)
                    break
                if t == pad:
                    break
                y.append(t)
            outs.append(y)
    return outs


def chunks(prefixes, ys, budget):
    """Index groups, sorted by length, with at most `budget` padded tokens (batch x longest sequence) each."""
    L = lambda i: len(prefixes[i]) + len(ys[i])
    order = sorted(range(len(ys)), key=L)
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and (j == i or (j - i + 1) * L(order[j]) <= budget):
            j += 1
        yield order[i:j]
        i = j


def seq_logps(prefixes, ys, idx):
    """log p(y_l | prefix, y_<l) for the sequences in idx (one forward); 1-D tensors in idx order."""
    ids, mask = left_pad([prefixes[r] + ys[r] for r in idx], pad, dev)
    keep = max(len(ys[r]) for r in idx) + 1
    logits = model(input_ids=ids, attention_mask=mask, logits_to_keep=keep, use_cache=False).logits.float()
    lp = torch.log_softmax(logits, -1)
    out = []
    for b, r in enumerate(idx):
        ly = len(ys[r])
        pos = torch.arange(keep - ly - 1, keep - 1, device=dev)       # the logits predicting y_l sit one position earlier
        out.append(lp[b, pos, ids[b, -ly:]])
    return out


def validate(tag):
    """Held-out documents: one sampled answer per dataset question with the inference decoding."""
    res = []
    for u in val_units[R::W]:
        load_k(u)
        activate()
        qs = u["official"]
        ys = sample([student_ids(q["question"]) for q in qs], 1, seed=7, **INFER_SAMPLING)
        for q, y in zip(qs, ys):
            valid, _, pred = parse_answer(tok.decode(y, skip_special_tokens=True))
            res.append({"set": u["set"], "f1": best_f1(pred, q["answers"]), "valid": int(valid)})
    allres = [None] * W
    dist.all_gather_object(allres, res)
    allres = [x for l in allres for x in l]
    by = {}
    for x in allres:
        by.setdefault(x["set"], []).append(x)
    by["all"] = allres
    rep = {s: {"n": len(v), "f1": 100 * sum(x["f1"] for x in v) / max(1, len(v)),
               "valid": 100 * sum(x["valid"] for x in v) / max(1, len(v))} for s, v in by.items()}
    log(f"VAL {tag}: " + " | ".join(f"{s} {v['f1']:.1f} (n={v['n']}, valid {v['valid']:.0f}%)" for s, v in rep.items()))
    if R == 0:
        with open(os.path.join(a.out, "log.jsonl"), "a") as f:
            f.write(json.dumps({"val": tag, "rep": rep}) + "\n")


# ---------------- training ----------------
os.makedirs(a.out, exist_ok=True)
rng = random.Random(a.seed)
validate("epoch0")
step, stop = 0, False
for ep in range(1, a.epochs + 1):
    order = list(range(len(train_units)))
    rng.shuffle(order)
    n_steps = math.ceil(len(order) / W)
    total = min(n_steps * a.epochs, a.max_steps) if a.max_steps else n_steps * a.epochs
    for st in range(n_steps):
        i = st * W + R
        u = train_units[order[i % len(order)]]              # processes past the end repeat a unit
        load_k(u)
        qs = list(u["official"])
        random.Random(a.seed * 1000 + ep * 100 + order[i % len(order)]).shuffle(qs)
        qs = qs[:a.max_q]
        sp = [student_ids(q["question"]) for q in qs]
        # 1. the student samples G answers per question (no document in the prompt)
        activate()
        ys = sample(sp, a.samples, 1.0, 1.0, 0)
        ys = [y if y else [eos] for y in ys]
        qidx = [qi for qi in range(len(qs)) for _ in range(a.samples)]
        parsed = [parse_answer(tok.decode(y, skip_special_tokens=True)) for y in ys]
        valid = [pz[0] for pz in parsed]
        sp_rep = [sp[qi] for qi in qidx]
        # 2. reward and group-normalised advantage
        rew = [best_f1(pz[2], qs[qi]["answers"]) if pz[0] else INVALID_REWARD for pz, qi in zip(parsed, qidx)]
        A = [None] * len(ys)
        zero_std = 0.0
        for qi in range(len(qs)):
            rs = [r for r in range(len(ys)) if qidx[r] == qi]
            m = sum(rew[r] for r in rs) / len(rs)
            sd = (sum((rew[r] - m) ** 2 for r in rs) / len(rs)) ** 0.5
            zero_std += float(sd < 1e-6)
            for r in rs:
                A[r] = (rew[r] - m) / (sd + 1e-4)
        zero_std /= max(1, len(qs))
        # 3. policy gradient on LoRA_s: every answer length-normalised and equally weighted
        activate()
        opt.zero_grad(set_to_none=True)
        loss = 0.0
        for idx in chunks(sp_rep, ys, a.grad_budget):
            with torch.enable_grad():
                part = sum(-A[r] * ls.sum() / len(ys[r]) for r, ls in zip(idx, seq_logps(sp_rep, ys, idx))) / len(ys)
                part.backward()
            loss += part.item()
        for p in s_params:
            if p.grad is not None:
                dist.all_reduce(p.grad)
                p.grad /= W
        gn = torch.nn.utils.clip_grad_norm_(s_params, 1.0)
        opt.step()
        step += 1
        stat = torch.tensor([loss, sum(valid) / len(valid), sum(len(y) for y in ys) / len(ys), float(gn),
                             sum(rew) / len(rew), zero_std], device=dev)
        dist.all_reduce(stat)
        stat /= W
        if R == 0:
            names = ["loss", "valid", "mean_len", "gnorm", "reward", "zero_std"]
            rec = {"ep": ep, "step": step, **{n: v for n, v in zip(names, stat.tolist())}, "unit": u["id"]}
            with open(os.path.join(a.out, "log.jsonl"), "a") as f:
                f.write(json.dumps(rec) + "\n")
            if step % 10 == 0 or step == 1:
                log(f"ep {ep} step {step}/{total} loss {rec['loss']:.4f} reward {rec['reward']:.3f} valid {rec['valid']:.2f} "
                    f"len {rec['mean_len']:.0f} zero_std {rec['zero_std']:.2f} gnorm {rec['gnorm']:.2f}")
        if a.val_every and step % a.val_every == 0:
            validate(f"ep{ep} step{step}")
        if a.max_steps and step >= a.max_steps:
            stop = True
            break
    if R == 0:
        model.save_pretrained(os.path.join(a.out, f"epoch{ep}"), selected_adapters=["s"])
        log(f"saved LoRA_s -> {a.out}/epoch{ep}/s")
    dist.barrier()
    validate(f"epoch{ep}")
    if stop:
        break
log("STUDENT UPDATE DONE")
dist.barrier()
dist.destroy_process_group()
