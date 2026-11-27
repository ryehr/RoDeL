"""Prompts, output parsing, metrics, model helpers and file layout shared by the RoDeL scripts."""
import json
import os
import re
import string
from collections import Counter

import torch
import torch.nn.functional as F

DEFAULT_MODEL = "Qwen/Qwen3.5-4B"
DEFAULT_MAX_NEW_TOKENS = 1024
INFER_SAMPLING = dict(temperature=0.6, top_p=0.95, top_k=20)   # decoding outside the RL updates
INVALID_REWARD = -2.0                                           # reward of a format-invalid student answer
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# ---------------------------------------------------------------- prompts
DOC_TMPL = "[Document {i}] {title}\n{text}\n"
CONTEXT_INTRO = {
    "hotpotqa": "Below is a set of documents.",
    "musique": "Below is a set of documents.",
    "2wiki": "Below is a set of documents.",
    "squad": "Below is a set of documents.",
    "nq": "Below is a document split into parts.",
    "triviaqa": "Below is a document split into parts.",
    "multifieldqa": "Below is a document split into parts.",
    "qasper": "Below is a scientific paper split into sections.",
    "narrativeqa": "Below is a story split into parts.",
    "locomo": "Below is a long-term conversation split into sessions.",
}
DEFAULT_INTRO = "Below is a set of documents."
QUESTION_TMPL = ("Answer the following question. Give a short factual answer (a few words). "
                 "For yes/no questions, answer exactly yes or no. If the question asks for a date "
                 "or a number, answer with the date or the number.\n\nQuestion: {q}")


def dataset_of(set_name):
    """Dataset behind a unit file: training files are named train_<dataset>."""
    return set_name[len("train_"):] if set_name.startswith("train_") else set_name


def render_docs(paragraphs):
    return "\n".join(DOC_TMPL.format(i=i + 1, title=p["title"], text=p["text"]) for i, p in enumerate(paragraphs))


def student_prompt(question):
    """User message of the student: the question only."""
    return QUESTION_TMPL.format(q=question)


def teacher_prompt(unit, question):
    """User message of the teacher: the document followed by the same question wrapper."""
    intro = CONTEXT_INTRO.get(dataset_of(unit["set"]), DEFAULT_INTRO)
    return intro + "\n\n" + render_docs(unit["paragraphs"]) + "\n\n" + QUESTION_TMPL.format(q=question)


def chat_frame(tok):
    """(head, tail) of the chat template around a user message, with thinking enabled."""
    mark = "@@S@@"
    text = tok.apply_chat_template([{"role": "user", "content": mark}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=True)
    head, tail = text.split(mark)
    return head, tail


# ---------------------------------------------------------------- output parsing
def parse_answer(text):
    """Model output -> (valid, reasoning trace, answer).

    The output is valid if the reasoning trace is closed ('</think>') and a non-empty line follows it;
    the answer is the first non-empty line after the trace."""
    if "</think>" not in text:
        return False, "", ""
    trace, body = text.split("</think>", 1)
    lines = [l.strip() for l in body.strip().splitlines() if l.strip()]
    return bool(lines), trace.replace("<think>", "", 1).strip(), (lines[0] if lines else "")


# ---------------------------------------------------------------- metrics
def normalize(s):
    s = "".join(ch for ch in s.lower() if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def token_f1(pred, gold):
    pt, gt = normalize(pred).split(), normalize(gold).split()
    common = Counter(pt) & Counter(gt)
    ns = sum(common.values())
    if ns == 0 or not pt or not gt:
        return 0.0
    p, r = ns / len(pt), ns / len(gt)
    return 2 * p * r / (p + r)


def best_f1(pred, answers):
    """Token F1 against the closest reference."""
    return max(token_f1(pred, g) for g in answers) if answers else 0.0


# ---------------------------------------------------------------- model helpers
def lora_targets(model):
    """LoRA modules: the attention and MLP projections, plus the projections of linear-attention layers
    when the model has them, i.e. every linear layer of the text model."""
    has_linear_attn = any(".linear_attn." in n for n, _ in model.named_modules())
    return LORA_TARGETS + (["in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"] if has_linear_attn else [])


def fix_linear_attn_cache(model):
    """Models with linear-attention layers: store the new recurrent state by assignment instead of an in-place
    copy into the cached tensor. The distillation forward encodes the prompt with the adapters off, keeps the
    cache and back-propagates through the target tokens; the in-place copy would overwrite a tensor that the
    backward pass still needs. No effect on models without linear attention."""
    if not any(".linear_attn." in n for n, _ in model.named_modules()):
        return
    from transformers import cache_utils as cu

    def update_recurrent_state(self, recurrent_states, state_idx=0, **kw):
        if not self.is_recurrent_states_initialized[state_idx]:
            self.lazy_initialization(recurrent_states=recurrent_states, state_idx=state_idx)
        self.recurrent_states[state_idx] = recurrent_states
        return recurrent_states

    cu.LinearAttentionLayer.update_recurrent_state = update_recurrent_state


def vllm_model_kwargs(model_path):
    """vLLM arguments for a vision-language checkpoint: load the language model only."""
    from transformers import AutoConfig
    return {"language_model_only": True} if hasattr(AutoConfig.from_pretrained(model_path), "vision_config") else {}


def left_pad(seqs, pad_id, device):
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, L - len(s):] = torch.tensor(s)
        mask[i, L - len(s):] = 1
    return ids.to(device), mask.to(device)


def right_pad(seqs, pad_id, device):
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
        mask[i, :len(s)] = 1
    return ids.to(device), mask.to(device)


def positions(mask):
    return (mask.cumsum(-1) - 1).clamp(min=0)


def distill_logits(model, prompts, targets, pad_id, device):
    """Logits of the target tokens for a distillation batch -> (logits, target ids, target mask).

    The prompt is encoded by the frozen base model (adapters off) into a key-value cache; the target tokens
    are then scored with the active adapter on."""
    p_ids, p_mask = left_pad(prompts, pad_id, device)
    pre_ids, pre_mask = p_ids[:, :-1], p_mask[:, :-1]
    with torch.no_grad(), model.disable_adapter():
        past = model(input_ids=pre_ids, attention_mask=pre_mask,
                     position_ids=positions(pre_mask), use_cache=True).past_key_values
    y_ids, y_mask = right_pad(targets, pad_id, device)
    resp = torch.cat([p_ids[:, -1:], y_ids], 1)                 # last prompt token + target
    resp_mask = torch.cat([torch.ones_like(p_ids[:, -1:]), y_mask], 1)
    fmask = torch.cat([pre_mask, resp_mask], 1)
    pos = positions(fmask)[:, pre_ids.shape[1]:]
    out = model(input_ids=resp, attention_mask=fmask, position_ids=pos,
                past_key_values=past, use_cache=False)
    return out.logits[:, :y_ids.shape[1]], y_ids, y_mask


def ce_loss(logits, y_ids, y_mask):
    """Token-mean cross-entropy over the target tokens."""
    B, L, V = logits.shape
    ce = F.cross_entropy(logits.float().reshape(-1, V), y_ids.reshape(-1), reduction="none").reshape(B, L)
    return (ce * y_mask).sum() / y_mask.sum().clamp(min=1)


# ---------------------------------------------------------------- data and file layout
def load_units(data_dir, name):
    """Units (documents with their dataset questions) of {data_dir}/{name}.jsonl."""
    units = []
    with open(os.path.join(data_dir, f"{name}.jsonl"), encoding="utf-8") as f:
        for line in f:
            u = json.loads(line)
            u["set"] = name
            units.append(u)
    return units


def load_questions(data_dir, name):
    """Synthetic questions of {data_dir}/{name}.questions.jsonl -> {unit id: [question, ...]}."""
    with open(os.path.join(data_dir, f"{name}.questions.jsonl"), encoding="utf-8") as f:
        return {r["id"]: [x["q"] for x in r["questions"]] for r in map(json.loads, f)}


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def answers_path(run_dir, tag, name, shard):
    """Teacher answer sets of one shard of a unit file."""
    return os.path.join(run_dir, "answers", tag, f"{name}.shard{shard}.jsonl")


def knowledge_dir(run_dir, tag, name, unit_id=None):
    """Knowledge adapters distilled from the answer sets `tag`: one sub-directory per unit."""
    d = os.path.join(run_dir, "knowledge", tag, name)
    return d if unit_id is None else os.path.join(d, unit_id)


def distill_log_path(run_dir, tag, name, shard):
    return os.path.join(run_dir, "knowledge", tag, f"{name}.shard{shard}.jsonl")


def preds_path(run_dir, tag, name, shard):
    """Student predictions of one shard of a unit file."""
    return os.path.join(run_dir, "preds", tag, f"{name}.shard{shard}.jsonl")
