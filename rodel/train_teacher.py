"""Teacher update: train the teacher adapter (LoRA_t) with set-level GRPO.

One frozen base model carries three adapters: t (teacher), k (knowledge, re-initialised for every answer
set) and s (student). Per step, each process takes one training document:
  1. synthesis: with the document, the teacher samples M answer sets over all synthetic questions;
  2. distillation: each answer set is distilled into a fresh LoRA_k (format-invalid answers are left out);
  3. under each LoRA_k the student answers the dataset questions without the document (greedy decoding);
     the reward of an answer set is the student's mean reward minus lambda x its share of invalid answers;
  4. the advantage is normalised over the M answer sets of the document; the loss is the advantage-weighted,
     length-normalised log-likelihood of every answer of each set, given the document.
Gradients are averaged over the processes; one optimizer step per training step.

Default (RoDeL-Seq): the student is frozen (--init-student, or an untrained student without it).
--joint (RoDeL-Sim): the student is updated in the same step. Under each LoRA_k it also samples G answers
per dataset question and LoRA_s receives the GRPO update of train_student.py.

  python -m torch.distributed.run --nproc_per_node 8 rodel/train_teacher.py \
      --sets train_hotpotqa train_musique --init-student runs/seq/ckpt/s1/epoch1/s --out runs/seq/ckpt/t1

Units with "split": "val" are held out. Validation: the teacher answers the synthetic questions once with the
inference decoding, the answers are distilled, and the dataset questions are answered with configurations
k (base + LoRA_k) and ks (base + LoRA_k + LoRA_s).
Writes {out}/step{N}/{t,s} and {out}/epoch{e}/{t,s} (PEFT adapters) and {out}/log.jsonl.
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

torch.backends.cuda.enable_cudnn_sdp(False)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (DEFAULT_MAX_NEW_TOKENS, DEFAULT_MODEL, INFER_SAMPLING, INVALID_REWARD, best_f1, ce_loss, chat_frame,
                    distill_logits, fix_linear_attn_cache, left_pad, load_questions, load_units, lora_targets,
                    parse_answer, student_prompt, teacher_prompt)

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--sets", nargs="+", required=True, help="training unit files")
ap.add_argument("--model", default=DEFAULT_MODEL)
ap.add_argument("--out", required=True)
ap.add_argument("--joint", action="store_true", help="also update the student in every step (RoDeL-Sim)")
ap.add_argument("--init-teacher", default="", help="continue from this LoRA_t directory")
ap.add_argument("--init-student", default="", help="LoRA_s directory: the frozen student, or the starting point with --joint")
ap.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS, help="length limit of every generation")
ap.add_argument("--epochs", type=int, default=1)
ap.add_argument("--answer-sets", type=int, default=4, help="M: answer sets sampled per document")
ap.add_argument("--samples", type=int, default=4, help="G: sampled student answers per dataset question (--joint)")
ap.add_argument("--max-q", type=int, default=8, help="dataset questions used per document and step")
ap.add_argument("--invalid-weight", type=float, default=2.0, help="lambda: weight of the share of invalid answers in the set reward")
ap.add_argument("--lr", type=float, default=1e-5)
ap.add_argument("--rank-t", type=int, default=64)
ap.add_argument("--rank-s", type=int, default=64)
ap.add_argument("--rank-k", type=int, default=16)
ap.add_argument("--k-epochs", type=int, default=5)
ap.add_argument("--k-lr", type=float, default=2e-4)
ap.add_argument("--k-batch", type=int, default=4)
ap.add_argument("--grad-budget", type=int, default=4096, help="padded tokens (batch x longest) per forward+backward chunk")
ap.add_argument("--ckpt-len", type=int, default=6000,
                help="teacher gradient: use gradient checkpointing when a sequence is longer than this many tokens")
ap.add_argument("--val-every", type=int, default=28, help="validate and save every N steps (0 = per epoch only)")
ap.add_argument("--limit", type=int, default=0, help="only the first N training and N validation units per set")
ap.add_argument("--max-steps", type=int, default=0, help="stop after this many steps (0 = full epochs)")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--data-dir", default="data")
a = ap.parse_args()
FREEZE_S = not a.joint

dist.init_process_group("nccl", timeout=datetime.timedelta(hours=4))
R, W = dist.get_rank(), dist.get_world_size()
dev = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")
torch.cuda.set_device(dev)


def log(*x):
    if R == 0:
        print(f"[teacher {time.strftime('%m-%d %H:%M:%S')}]", *x, flush=True)


# ---------------- data ----------------
train_units, val_units, SYN = [], [], {}
for S in a.sets:
    rows = load_units(a.data_dir, S)
    Q = load_questions(a.data_dir, S)
    for u in rows:
        SYN[u["id"]] = Q.get(u["id"], [])
    rows = [u for u in rows if SYN[u["id"]]]
    tr = [u for u in rows if u.get("split", "train") == "train"]
    va = [u for u in rows if u.get("split") == "val"]
    if a.limit:
        tr, va = tr[:a.limit], va[:a.limit]
    train_units += tr
    val_units += va
    log(f"{S}: {len(tr)} training / {len(va)} validation units; "
        f"synthetic questions per unit {sum(len(SYN[u['id']]) for u in rows) / len(rows):.1f}")

# ---------------- model: base + t + k + s ----------------
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained(a.model)
pad = tok.pad_token_id or tok.eos_token_id
eos = tok.eos_token_id
enc = lambda t: tok(t, add_special_tokens=False)["input_ids"]
base = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(dev)
fix_linear_attn_cache(base)
TARGETS = lora_targets(base)
cfg = lambda r, dropout=0.0: LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=dropout, task_type="CAUSAL_LM", target_modules=TARGETS)
model = get_peft_model(base, cfg(a.rank_t), adapter_name="t")
model.add_adapter("k", cfg(a.rank_k, 0.05))
model.add_adapter("s", cfg(a.rank_s))


def adapter_of(n):
    for x in ("t", "k", "s"):
        if f".lora_A.{x}." in n or f".lora_B.{x}." in n:
            return x
    return None


P = {"t": [], "k": [], "s": []}
for n, p in model.named_parameters():
    x = adapter_of(n)
    if x:
        p.data = p.data.float()
        P[x].append((n, p))
    p.requires_grad = False


def freeze(train=None):
    for x in P:
        for _, p in P[x]:
            p.requires_grad = (x == train)


def activate(names, train=None):
    model.base_model.set_adapter(names)
    freeze(train)


def reset_k():
    with torch.no_grad():
        for n, p in P["k"]:
            if ".lora_A." in n:
                torch.nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            else:
                p.zero_()


def snapshot_k():
    return [p.detach().clone() for _, p in P["k"]]


def load_k(snap):
    with torch.no_grad():
        for (_, p), q in zip(P["k"], snap):
            p.copy_(q)


opt_t = torch.optim.AdamW([p for _, p in P["t"]], lr=a.lr, weight_decay=0.0)
opt_s = torch.optim.AdamW([p for _, p in P["s"]], lr=a.lr, weight_decay=0.0)
for x, d in (("s", a.init_student), ("t", a.init_teacher)):
    if d:
        res = set_peft_model_state_dict(model, load_file(os.path.join(d, "adapter_model.safetensors"), device=str(dev)), adapter_name=x)
        assert not res.unexpected_keys, res.unexpected_keys
        for _, p in P[x]:
            p.data = p.data.float()
        log(f"loaded LoRA_{x} from {d}")
model.config.use_cache = True
model.eval()
log("parameters: " + " ".join(f"{x} {sum(p.numel() for _, p in P[x]) / 1e6:.1f}M" for x in P))

HEAD, TAIL = chat_frame(tok)


def student_ids(q):
    return enc(HEAD + student_prompt(q) + TAIL)


def teacher_ids(u, q):
    return enc(HEAD + teacher_prompt(u, q) + TAIL)


@torch.no_grad()
def sample(prompts, n, temperature, top_p, top_k, seed=None, bs=8, greedy=False):
    """n completions per prompt (prompt-major) as token lists; the end-of-sequence token is kept."""
    outs = []
    for s in range(0, len(prompts), bs):
        chunk = [p for p in prompts[s:s + bs] for _ in range(n)]
        ids, mask = left_pad(chunk, pad, dev)
        if seed is not None:
            torch.manual_seed(seed + s)
        if greedy:
            kw = dict(do_sample=False, temperature=1.0, top_p=1.0, top_k=50)
        else:
            kw = dict(do_sample=True, temperature=temperature, top_p=top_p, top_k=top_k)
        gen = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=a.max_new_tokens, pad_token_id=pad, eos_token_id=eos, **kw)
        for row in gen[:, ids.size(1):].tolist():
            y = []
            for t in row:
                if t == eos:
                    y.append(t)
                    break
                if t == pad:
                    break
                y.append(t)
            outs.append(y if y else [eos])
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
        pos = torch.arange(keep - ly - 1, keep - 1, device=dev)
        out.append(lp[b, pos, ids[b, -ly:]])
    return out


def reward(text, answers):
    """Student reward: token F1 of a valid answer, a fixed negative reward otherwise -> (reward, valid)."""
    valid, _, pred = parse_answer(text)
    return (best_f1(pred, answers) if valid else INVALID_REWARD), valid


def train_k(pairs):
    """Distil (prompt ids, target ids) pairs into a fresh LoRA_k; returns the mean loss of the last epoch."""
    reset_k()
    activate(["k"], train="k")
    k_params = [p for _, p in P["k"]]
    opt = torch.optim.AdamW(k_params, lr=a.k_lr, weight_decay=0.01)
    rng = random.Random(1234)
    last = 0.0
    model.train()                                           # dropout of LoRA_k; t and s have none
    for ep in range(a.k_epochs):
        order = list(range(len(pairs)))
        rng.shuffle(order)
        total, nb = 0.0, 0
        for s in range(0, len(order), a.k_batch):
            idx = order[s:s + a.k_batch]
            with torch.enable_grad():
                logits, y_ids, y_mask = distill_logits(model, [pairs[i][0] for i in idx], [pairs[i][1] for i in idx], pad, dev)
                loss = ce_loss(logits, y_ids, y_mask)
                if not torch.isfinite(loss):
                    opt.zero_grad(set_to_none=True)
                    continue
                loss.backward()
            torch.nn.utils.clip_grad_norm_(k_params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            total += loss.item()
            nb += 1
        last = total / max(1, nb)
    model.eval()
    freeze()
    return last


def greedy_answers(names, sp, qs_gt):
    """One greedy answer per dataset question under the adapters `names` -> (rewards, valid flags, lengths)."""
    activate(names)
    ys = sample(sp, 1, 1.0, 1.0, 0, greedy=True)
    rf = [reward(tok.decode(y, skip_special_tokens=True), q["answers"]) for y, q in zip(ys, qs_gt)]
    return [x[0] for x in rf], [x[1] for x in rf], [len(y) for y in ys]


def synthesize(u, n_sets, temperature, top_p, top_k, seed=None):
    """n_sets answers for every synthetic question of u -> (questions, answer sets as lists of token lists)."""
    qs = SYN[u["id"]]
    activate(["t"])
    ys = sample([teacher_ids(u, q) for q in qs], n_sets, temperature, top_p, top_k, seed=seed, bs=4)
    return qs, [[ys[j * n_sets + i] for j in range(len(qs))] for i in range(n_sets)]


def distill(qs, S):
    """Answer set S (aligned with qs) -> (LoRA_k snapshot, share of invalid answers, last loss)."""
    pairs, invalid = [], 0
    for q, y in zip(qs, S):
        if not parse_answer(tok.decode(y, skip_special_tokens=True))[0]:
            invalid += 1
            continue
        pairs.append((student_ids(q), (y if y[-1] == eos else y + [eos])))
    if not pairs:
        reset_k()
        return snapshot_k(), 1.0, float("nan")
    loss = train_k(pairs)
    return snapshot_k(), invalid / len(S), loss


def validate(tag):
    res = []
    for u in val_units[R::W]:
        qs, sets = synthesize(u, 1, seed=11, **INFER_SAMPLING)
        snap, b, _ = distill(qs, sets[0])
        load_k(snap)
        sp = [student_ids(q["question"]) for q in u["official"]]
        for c, names in (("k", ["k"]), ("ks", ["k", "s"])):
            activate(names)
            ys = sample(sp, 1, seed=7, **INFER_SAMPLING)
            for q, y in zip(u["official"], ys):
                r, valid = reward(tok.decode(y, skip_special_tokens=True), q["answers"])
                res.append({"set": u["set"], "config": c, "f1": max(r, 0.0), "valid": int(valid), "invalid_t": b})
    allres = [None] * W
    dist.all_gather_object(allres, res)
    allres = [x for l in allres for x in l]
    rep = {}
    for c in ("k", "ks"):
        by = {}
        for x in allres:
            if x["config"] == c:
                by.setdefault(x["set"], []).append(x)
        by["all"] = [x for x in allres if x["config"] == c]
        rep[c] = {s: {"n": len(v), "f1": 100 * sum(x["f1"] for x in v) / max(1, len(v)),
                      "valid": 100 * sum(x["valid"] for x in v) / max(1, len(v))} for s, v in by.items()}
        log(f"VAL {tag} [{c}]: " + " | ".join(f"{s} {v['f1']:.1f} (n={v['n']}, valid {v['valid']:.0f}%)" for s, v in rep[c].items()))
    ks = [x for x in allres if x["config"] == "k"]
    rep["teacher_invalid"] = 100 * sum(x["invalid_t"] for x in ks) / max(1, len(ks))
    log(f"VAL {tag}: teacher invalid {rep['teacher_invalid']:.0f}%")
    if R == 0:
        with open(os.path.join(a.out, "log.jsonl"), "a") as f:
            f.write(json.dumps({"val": tag, "rep": rep}) + "\n")


def save(name):
    if R == 0:
        model.save_pretrained(os.path.join(a.out, name), selected_adapters=["t", "s"])
        log(f"saved t, s -> {a.out}/{name}")
    dist.barrier()


def make_steps(units, rng):
    """Steps of W units from the same unit file (one per process); the last step of a file is filled by repeats."""
    by = {}
    for u in units:
        by.setdefault(u["set"], []).append(u)
    steps = []
    for us in by.values():
        rng.shuffle(us)
        for i in range(0, len(us), W):
            chunk = us[i:i + W]
            while len(chunk) < W:
                chunk.append(us[rng.randrange(len(us))])
            steps.append(chunk)
    rng.shuffle(steps)
    return steps


# ---------------- training ----------------
os.makedirs(a.out, exist_ok=True)
rng = random.Random(a.seed)
validate("epoch0")
step, stop = 0, False
for ep in range(1, a.epochs + 1):
    steps = make_steps(train_units, rng)
    total = min(len(steps) * a.epochs, a.max_steps) if a.max_steps else len(steps) * a.epochs
    for st, chunk in enumerate(steps):
        t0 = time.time()
        u = chunk[R]
        qs_gt = list(u["official"])
        random.Random(a.seed * 1000 + ep * 100 + st).shuffle(qs_gt)
        qs_gt = qs_gt[:a.max_q]
        # 1. synthesis: M answer sets over the synthetic questions of the document
        qs, sets = synthesize(u, a.answer_sets, 1.0, 1.0, 0)
        t_gen = time.time() - t0
        # 2. distil each set into a fresh LoRA_k; 3. the student answers under each LoRA_k
        opt_s.zero_grad(set_to_none=True)
        sp = [student_ids(q["question"]) for q in qs_gt]
        Ri, bi, ces, s_rews, s_zero, s_groups, s_valid, s_len, n_all = [], [], [], [], 0, 0, 0, 0, 0
        for i in range(a.answer_sets):
            snap, b, ce = distill(qs, sets[i])
            load_k(snap)
            bi.append(b)
            ces.append(ce)
            if FREEZE_S:                                    # frozen student: greedy answers only
                rew, vl, ln = greedy_answers(["k", "s"], sp, qs_gt)
                s_valid += sum(vl); s_len += sum(ln); n_all += len(rew); s_rews += rew
                Ri.append(sum(rew) / len(rew) - a.invalid_weight * b)
                continue
            # joint: G sampled answers per question for the student update, greedy answers for the set reward
            activate(["k", "s"])
            ys = sample(sp, a.samples, 1.0, 1.0, 0)
            qidx = [qi for qi in range(len(qs_gt)) for _ in range(a.samples)]
            rf = [reward(tok.decode(y, skip_special_tokens=True), qs_gt[qi]["answers"]) for y, qi in zip(ys, qidx)]
            rew = [x[0] for x in rf]
            s_valid += sum(x[1] for x in rf); s_len += sum(len(y) for y in ys); n_all += len(ys); s_rews += rew
            greedy_rew, _, _ = greedy_answers(["k", "s"], sp, qs_gt)
            Ri.append(sum(greedy_rew) / len(greedy_rew) - a.invalid_weight * b)
            A = [None] * len(ys)
            for qi in range(len(qs_gt)):
                rs = [r for r in range(len(ys)) if qidx[r] == qi]
                m = sum(rew[r] for r in rs) / len(rs)
                sd = (sum((rew[r] - m) ** 2 for r in rs) / len(rs)) ** 0.5
                s_groups += 1
                s_zero += int(sd < 1e-6)
                for r in rs:
                    A[r] = (rew[r] - m) / (sd + 1e-4)
            sp_rep = [sp[qi] for qi in qidx]
            activate(["k", "s"], train="s")
            for idx in chunks(sp_rep, ys, a.grad_budget):
                if all(abs(A[r]) < 1e-8 for r in idx):
                    continue
                with torch.enable_grad():
                    part = sum(-A[r] * ls.sum() / len(ys[r]) for r, ls in zip(idx, seq_logps(sp_rep, ys, idx))) / (a.answer_sets * len(ys))
                    part.backward()
            freeze()
        t_stu = time.time() - t0 - t_gen
        # 4. set-level GRPO on the teacher: one advantage per answer set, shared by all of its answers
        mR = sum(Ri) / len(Ri)
        sdR = (sum((x - mR) ** 2 for x in Ri) / len(Ri)) ** 0.5
        At = [(x - mR) / (sdR + 1e-4) for x in Ri]
        t_zero = int(sdR < 1e-6)
        opt_t.zero_grad(set_to_none=True)
        loss_t = 0.0
        if not t_zero:
            tp = [teacher_ids(u, q) for q in qs]
            pre, ys_t, wt = [], [], []
            for i in range(a.answer_sets):
                for j, y in enumerate(sets[i]):
                    pre.append(tp[j]); ys_t.append(y); wt.append(At[i] / (a.answer_sets * len(sets[i])))
            activate(["t"], train="t")
            gc_on = max(len(p) + len(y) for p, y in zip(pre, ys_t)) > a.ckpt_len
            if gc_on:
                model.train()
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            for idx in chunks(pre, ys_t, a.grad_budget):
                with torch.enable_grad():
                    part = sum(-wt[r] * lp.sum() / len(ys_t[r]) for r, lp in zip(idx, seq_logps(pre, ys_t, idx)))
                    part.backward()
                loss_t += part.item()
            if gc_on:
                model.gradient_checkpointing_disable()
                model.eval()
                model.config.use_cache = True
            freeze()
        t_tea = time.time() - t0 - t_gen - t_stu
        # 5. average the gradients over the processes and step
        for x in ("t", "s"):
            if x == "s" and FREEZE_S:
                continue
            for _, p in P[x]:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                dist.all_reduce(p.grad)
                p.grad /= W
        gn_t = torch.nn.utils.clip_grad_norm_([p for _, p in P["t"]], 1.0)
        opt_t.step()
        if FREEZE_S:
            gn_s = torch.tensor(0.0)
        else:
            gn_s = torch.nn.utils.clip_grad_norm_([p for _, p in P["s"]], 1.0)
            opt_s.step()
        step += 1
        ok = [c for c in ces if c == c]
        stat = torch.tensor([sum(bi) / len(bi), mR, sdR, float(t_zero), sum(s_rews) / len(s_rews), s_zero / max(1, s_groups),
                             s_valid / n_all, s_len / n_all, float(gn_t), float(gn_s), sum(ok) / max(1, len(ok)), loss_t,
                             time.time() - t0, t_gen, t_stu, t_tea, float(len(qs))], device=dev)
        dist.all_reduce(stat)
        stat /= W
        if R == 0:
            names = ["t_invalid", "R_mean", "R_std", "t_zero", "s_reward", "s_zero", "s_valid", "s_len", "gnorm_t", "gnorm_s",
                     "k_loss", "loss_t", "sec", "sec_gen", "sec_stu", "sec_tea", "n_syn"]
            rec = {"ep": ep, "step": step, "unit": u["id"], "set": u["set"], **{n: round(v, 4) for n, v in zip(names, stat.tolist())}}
            with open(os.path.join(a.out, "log.jsonl"), "a") as f:
                f.write(json.dumps(rec) + "\n")
            if step % 5 == 0 or step == 1:
                log(f"ep {ep} step {step}/{total} [{u['set']}] t_invalid {rec['t_invalid']:.2f} R {rec['R_mean']:.3f} (std {rec['R_std']:.3f}) "
                    f"t_zero {rec['t_zero']:.2f} | s_reward {rec['s_reward']:.3f} s_valid {rec['s_valid']:.2f} len {rec['s_len']:.0f} "
                    f"| k_loss {rec['k_loss']:.3f} | gn_t {rec['gnorm_t']:.2f} gn_s {rec['gnorm_s']:.2f} "
                    f"| {rec['sec']:.0f}s (gen {rec['sec_gen']:.0f} stu {rec['sec_stu']:.0f} tea {rec['sec_tea']:.0f})")
        if a.val_every and step % a.val_every == 0:
            validate(f"ep{ep} step{step}")
            save(f"step{step}")
        if a.max_steps and step >= a.max_steps:
            stop = True
            break
    save(f"epoch{ep}")
    validate(f"epoch{ep}")
    if stop:
        break
log("TEACHER UPDATE DONE")
dist.barrier()
dist.destroy_process_group()
