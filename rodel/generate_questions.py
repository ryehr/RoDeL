"""Question generation: the synthetic questions of every document.

A language model reads the document, never the dataset questions, and writes questions about it. Each dataset
has a recipe (prompts in question_prompts.py): a few routes, each with one prompt and a number of questions.
  part   one prompt per part of the document
  docs   one prompt over all documents of the unit, with a number of questions per document
  unit   one prompt over the whole unit
A generated line is kept if it is a well-formed, self-contained question that is new for the unit. A prompt
that returned less than 80% of its target is sampled again in the next round with another seed.
Conversations (LoCoMo) have their own recipe: one prompt per session and three prompts over the whole
conversation, each sampled until its target is reached.

  CUDA_VISIBLE_DEVICES=0 python rodel/generate_questions.py --set hotpotqa --shard 0 --nshards 8

Writes {data-dir}/{set}.questions.shard{i}.jsonl, one row per unit: {"id", "questions": [{"q", "route"}]}.
The shards of a set are joined into {data-dir}/{set}.questions.jsonl (scripts/common.sh).
"""
import argparse
import collections
import json
import os
import re
import string
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (INFER_SAMPLING, dataset_of, exit_with_children, load_units, prepare_vllm, questions_path, token_f1,
                    vllm_model_kwargs)
from question_prompts import LOCOMO, PREVIOUS_PART, PROMPTS, STYLES

DEFAULT_GENERATOR = "Qwen/Qwen3-4B"                  # model that writes the questions
DEFAULT_QUESTION_TOKENS = 8192                       # length limit of one output: reasoning trace + questions
RECIPE_OF = {"triviaqa": "nq"}                       # datasets that use the recipe of another dataset
DEFAULT_TYPE = {"hotpotqa": "bridge", "2wiki": "compositional"}    # style of route R for units without a "type"
WINDOW = {"narrativeqa": 40960, "nq": 40960, "locomo": 32768}    # vLLM context window by recipe (default 24576)

# recipe -> route -> (kind, questions per part or per document, questions per unit)
ROUTES = {
    "musique":      {"S": ("docs", 2, 40), "B": ("unit", None, 10)},
    "2wiki":        {"S": ("docs", 3, 30), "R": ("unit", None, 10)},
    "qasper":       {"S": ("docs", 3, 40), "E": ("unit", None, 10), "Y": ("unit", None, 6), "A": ("unit", None, 4)},
    "hotpotqa":     {"S": ("docs", 3, None), "R": ("unit", None, 10)},
    "squad":        {"S": ("unit", None, 20)},
    "nq":           {"G": ("unit", None, 20), "Q": ("unit", None, 15), "L": ("part", 12, 96)},
    "narrativeqa":  {"L1": ("part", 15, None), "L2": ("part", 15, None), "G": ("unit", None, 40)},
    "multifieldqa": {"S": ("part", 5, 45), "E": ("unit", None, 10), "A": ("unit", None, 5)},
}
RESAMPLE = 0.8                                       # a prompt below this share of its target is sampled again
NEAR_DUPLICATE = 0.8                                 # token F1 above which a question repeats an earlier one
EXACT_ONLY = ("musique", "2wiki", "qasper")          # recipes that drop exact repetitions only
SEPARATE = {"nq": {"Q"}}                             # routes compared with their own questions only
PRONOUNS = ("he ", "she ", "it ", "they ", "his ", "her ", "their ", "this ", "these ", "that ")

# references that tie a question to the prompt
DOCUMENT_WORDS = re.compile(r"\b(documents?|passages?|paragraphs?|texts?|articles?)\b|\bD\d+\b|\bthe one from\b|\bthe one in\b", re.I)
FORBIDDEN = {
    "musique": DOCUMENT_WORDS, "2wiki": DOCUMENT_WORDS, "hotpotqa": DOCUMENT_WORDS, "squad": DOCUMENT_WORDS, "qasper": None,
    "narrativeqa": re.compile(r"\b(passage|excerpt|paragraph|this part|that part|the part|part \d+|the text|the document)\b", re.I),
    "multifieldqa": re.compile(r"\b(passage|excerpt|this part|that part|the part|part \d+)\b", re.I),
    "nq": re.compile(r"\b(passage|excerpt|paragraph|this part|that part|the part|part \d+|the text|the document|the article|"
                     r"this article|the page|this page|wikipedia)\b", re.I),
}


def norm(q):
    return " ".join("".join(ch for ch in q.lower() if ch not in set(string.punctuation)).split())


def render(paragraphs, label):
    return "\n".join(f"[{label} {i + 1}] {p['title']}\n{p['text']}\n" for i, p in enumerate(paragraphs))


def title_of(unit):
    """Title of a document whose parts are titled '<title> (part i)'."""
    return re.sub(r"\s*\(part \d+\)\s*$", "", unit["paragraphs"][0]["title"])


def build_prompt(recipe, route, unit, part):
    """User message of one route; `part` is the index of the part for 'part' routes, None otherwise."""
    paras = unit["paragraphs"]
    n, P = len(paras), PROMPTS[recipe][route]
    if recipe in ("musique", "2wiki", "hotpotqa"):
        if route == "R":
            return P.format(n=n, docs=render(paras, "Document"), style=STYLES[recipe][unit.get("type") or DEFAULT_TYPE[recipe]])
        return P.format(n=n, docs=render(paras, "Document"))
    if recipe == "qasper":
        return P.format(n=n, docs=render(paras, "Section"))
    if recipe == "squad":
        return P.format(title=paras[0]["title"], chunk=paras[0]["text"])
    if recipe == "narrativeqa":
        if route == "G":
            return P.format(title=title_of(unit), n=n, docs=render(paras, "Part"))
        prev = "" if part == 0 else PREVIOUS_PART.format(i=part, text=paras[part - 1]["text"])
        return P.format(title=title_of(unit), n=n, i=part + 1, prev=prev, chunk=paras[part]["text"])
    if recipe == "nq":
        if route == "L":
            return P.format(title=title_of(unit), n=n, i=part + 1, head=paras[0]["text"][:1200].strip(), chunk=paras[part]["text"])
        return P.format(title=title_of(unit), n=n, docs=render(paras, "Part"))
    if route == "S":                                 # multifieldqa
        return P.format(n=n, i=part + 1, head=paras[0]["text"][:1200].strip(), chunk=paras[part]["text"])
    return P.format(docs=render(paras, "Part"))


def line_pattern(recipe):
    """A question line: the prefix asked for by the prompt ('D3: ', 'D3+D11: ', 'S4: ', 'Q: ', ...), then the question."""
    if recipe == "nq":
        return re.compile(r"^\s*(?:[-*\d.)\s]*)?([GLQ])\s*[:：]\s*(.+?)\s*$")
    letters = "DSEYA" if recipe in EXACT_ONLY else "DSEYAQGL"
    return re.compile(r"^\s*(?:[-*\d.)\s]*)?([%s]\d*(?:\+[DS]\d+)*)\s*[:：]\s*(.+\?)\s*$" % letters)


def parse(text, recipe, route, n_docs, part, pattern, stats):
    """Questions of one output: the lines after the reasoning trace that pass the filters."""
    if "</think>" not in text:
        stats["unclosed"] += 1
        return []
    out = []
    for line in text.split("</think>", 1)[1].split("\n"):
        m = pattern.match(line)
        if not m:
            continue
        tag, q = m.group(1), m.group(2).strip()
        if recipe == "nq":
            q = q.strip('"“”\'')
            if route == "Q":                         # search queries: lowercase, no question mark
                q = q.rstrip("?").strip().lower()
            elif not q.endswith("?"):
                stats["no question mark"] += 1
                continue
        if not 4 <= len(q.split()) <= 60:
            stats["length"] += 1
            continue
        if q.lower().startswith(PRONOUNS):
            stats["pronoun"] += 1
            continue
        if FORBIDDEN[recipe] is not None and FORBIDDEN[recipe].search(q):
            stats["reference"] += 1
            continue
        if part is not None or recipe == "squad":
            docs = [1 if part is None else part + 1]
        else:
            docs = [int(x) for x in re.findall(r"\d+", tag)]
            if any(d < 1 or d > n_docs for d in docs):
                stats["document number"] += 1
                continue
        out.append({"q": q, "route": route, "docs": docs})
    return out


def generate(llm, chat, recipe, units, rounds, cap):
    """Questions of the units of a document recipe -> {unit id: [{"q", "route"}]}."""
    from vllm import SamplingParams
    R, separate, pattern = ROUTES[recipe], SEPARATE.get(recipe, set()), line_pattern(recipe)
    got = {u["id"]: {rt: [] for rt in R} for u in units}
    seen = {u["id"]: set() for u in units}
    stats = collections.Counter()

    def count(uid, rt, d=None):
        return sum(1 for x in got[uid][rt] if d is None or x["docs"][:1] == [d])

    def near_duplicate(uid, it):
        if it["route"] in separate:
            pool = got[uid][it["route"]]
        else:                                        # same part or document, or either question is about the whole unit
            pool = [x for rt in R if rt not in separate for x in got[uid][rt]
                    if R[rt][0] == "unit" or R[it["route"]][0] == "unit" or x["docs"][:1] == it["docs"][:1]]
        return any(token_f1(it["q"], x["q"]) > NEAR_DUPLICATE for x in pool)

    for rnd in range(rounds):
        jobs = []
        for u in units:
            n = len(u["paragraphs"])
            for rt, (kind, per, total) in R.items():
                if kind == "part":
                    jobs += [(u, rt, i) for i in range(n) if rnd == 0 or count(u["id"], rt, i + 1) < RESAMPLE * per]
                elif rnd == 0 or count(u["id"], rt) < (RESAMPLE * total if total else RESAMPLE * per * n):
                    jobs.append((u, rt, None))
        if not jobs:
            break
        sp = SamplingParams(n=1, max_tokens=cap, seed=100 + rnd, **INFER_SAMPLING)
        outs = llm.generate([chat(build_prompt(recipe, rt, u, i)) for u, rt, i in jobs], sp)
        for (u, rt, i), o in zip(jobs, outs):
            uid, (kind, per, total) = u["id"], R[rt]
            for it in parse(o.outputs[0].text, recipe, rt, len(u["paragraphs"]), i, pattern, stats):
                key = norm(it["q"]) + (f"|{rt}" if rt in separate else "")
                if key in seen[uid]:
                    stats["repeated"] += 1
                    continue
                if kind != "unit" and it["docs"] and count(uid, rt, it["docs"][0]) >= per:
                    stats["over the count"] += 1
                    continue
                if recipe not in EXACT_ONLY and near_duplicate(uid, it):
                    stats["near-duplicate"] += 1
                    continue
                seen[uid].add(key)
                got[uid][rt].append(it)
        print(f"round {rnd}: {len(jobs)} prompts | questions per unit " +
              " ".join(f"{rt}={sum(count(u['id'], rt) for u in units) / len(units):.1f}" for rt in R) +
              f" | dropped {dict(stats)}", flush=True)

    result = {}
    for u in units:
        qs = []
        for rt, (kind, per, total) in R.items():
            items = got[u["id"]][rt]
            if kind == "part":                       # the parts take turns up to the number per unit
                rank, keyed = collections.Counter(), []
                for x in items:
                    d = x["docs"][0]
                    keyed.append((rank[d], d, x))
                    rank[d] += 1
                items = [x for _, _, x in sorted(keyed, key=lambda z: (z[0], z[1]))]
                if total:
                    items = items[:total]
                items.sort(key=lambda x: x["docs"][0])
            elif total:
                items = items[:total]
            qs += items
        result[u["id"]] = [{"q": x["q"], "route": x["route"]} for x in qs]
    return result


# ---------------------------------------------------------------- conversations
SPEAKER_PRONOUNS = ("he ", "she ", "they ", "his ", "her ", "their ")
PER_SESSION = 30                                     # single-hop questions per session
CONVERSATION_ROUTES = (("multi_hop", 120, 2000), ("temporal", 120, 3000), ("open_domain", 80, 4000))    # route, questions, seed
SESSION_TOKENS, CONVERSATION_TOKENS = 3072, 2560     # length limits of a session prompt and of a conversation prompt
DATED = (r"january|february|march|april|may|june|july|august|september|october|november|december|\d{4}")
RELATIVE_TIME = ["last week", "last weekend", "yesterday", "today", "recently", "last month", "next week"]


def split_session(text):
    """'DATE: <date>\\nCONVERSATION:\\n<turns>' -> (date, turns)."""
    date, _, turns = re.sub(r"^DATE: ", "", text.strip()).partition("\n")
    return date.strip(), turns.replace("CONVERSATION:\n", "", 1).strip()


def question_lines(text):
    if "</think>" not in text:
        return []
    lines = [l.strip().lstrip("-*0123456789. ").strip() for l in text.split("</think>", 1)[1].split("\n")]
    return [l for l in lines if l.endswith("?") and 6 <= len(l.split()) <= 60]


def anchor_date(q, date):
    """Pin a question without a date to the date of its session."""
    if re.search(DATED, q.lower()):
        return q
    day = re.sub(r"^\d{1,2}:\d{2} (am|pm) on ", "", date)
    q = re.sub(r"\b(the|this) (session|conversation)\b", f"the conversation on {day}", q, flags=re.I)
    if any(t in q.lower() for t in RELATIVE_TIME) and q.endswith("?"):
        q = q[:-1] + f" (as of {day})?"
    return q


def generate_conversation(llm, chat, unit, cap):
    """Questions of one conversation: per session (single-hop), then multi-hop, temporal and open-domain."""
    from vllm import SamplingParams
    A, B = unit["speakers"]
    paras = unit["paragraphs"][::-1]                 # sessions from the latest to the earliest
    sessions = [split_session(p["text"]) for p in paras]
    conv = "".join(p["text"] for p in paras).strip()

    def usable(q):                                   # names a speaker and does not open with a pronoun
        low = q.lower()
        return not low.startswith(SPEAKER_PRONOUNS) and any(name.lower() in low for name in (A, B))

    single, seen = [[] for _ in sessions], [set() for _ in sessions]
    for rnd in range(6):
        todo = [i for i in range(len(sessions)) if len(single[i]) < PER_SESSION]
        if not todo:
            break
        prompts = [chat(LOCOMO["single_hop"].format(A=A, B=B, date=sessions[i][0], session=sessions[i][1], N=20)) for i in todo]
        sp = SamplingParams(n=4, max_tokens=min(cap, SESSION_TOKENS), seed=1000 + rnd, **INFER_SAMPLING)
        for i, g in zip(todo, llm.generate(prompts, sp)):
            for o in g.outputs:
                for q in question_lines(o.text):
                    if usable(q) and norm(q) not in seen[i]:
                        seen[i].add(norm(q))
                        single[i].append(anchor_date(q, sessions[i][0]))
        print(f"{unit['id']} single_hop round {rnd}: {sum(len(s) for s in single)}", flush=True)
    qs = [{"q": q, "route": "single_hop"} for s in single for q in s[:PER_SESSION]]

    for route, target, seed in CONVERSATION_ROUTES:
        prompt = chat(LOCOMO[route].format(A=A, B=B, conv=conv, N=20))
        got, seen_route = [], set()
        for rnd in range(12):
            if len(got) >= target:
                break
            sp = SamplingParams(n=6, max_tokens=min(cap, CONVERSATION_TOKENS), seed=seed + rnd, **INFER_SAMPLING)
            for o in llm.generate([prompt], sp)[0].outputs:
                for q in question_lines(o.text):
                    if usable(q) and norm(q) not in seen_route:
                        seen_route.add(norm(q))
                        got.append(q)
            print(f"{unit['id']} {route} round {rnd}: {len(got)}", flush=True)
        qs += [{"q": q, "route": route} for q in got[:target]]
    return qs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True, help="unit file {data-dir}/{set}.jsonl")
    ap.add_argument("--recipe", default="", choices=[""] + sorted(ROUTES) + ["locomo"], help="recipe (default: by dataset name)")
    ap.add_argument("--model", default=DEFAULT_GENERATOR, help="model that writes the questions")
    ap.add_argument("--max-new-tokens", type=int, default=DEFAULT_QUESTION_TOKENS, help="length limit of one output")
    ap.add_argument("--rounds", type=int, default=3, help="sampling rounds of the document recipes")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="only the first N units of the shard")
    ap.add_argument("--max-model-len", type=int, default=0, help="vLLM context window (0 = by recipe)")
    ap.add_argument("--gpu-mem", type=float, default=0.90, help="vLLM gpu_memory_utilization")
    a = ap.parse_args()

    dataset = dataset_of(a.set)
    recipe = a.recipe or RECIPE_OF.get(dataset, dataset)
    if recipe != "locomo" and recipe not in ROUTES:
        sys.exit(f"no question recipe for '{dataset}': choose one with --recipe")
    units = load_units(a.data_dir, a.set)[a.shard::a.nshards]
    if a.limit:
        units = units[:a.limit]
    out = questions_path(a.data_dir, a.set, a.shard)
    if os.path.exists(out) and sum(1 for _ in open(out, encoding="utf-8")) >= len(units):
        print("already complete", out)
        return
    if not units:                                    # more shards than units
        open(out, "w").close()
        return
    prepare_vllm()

    from transformers import AutoTokenizer
    from vllm import LLM
    tok = AutoTokenizer.from_pretrained(a.model)
    max_len = a.max_model_len or WINDOW.get(recipe, 24576)
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=max_len, gpu_memory_utilization=a.gpu_mem,
              enable_prefix_caching=True, **vllm_model_kwargs(a.model))

    def chat(user):
        return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=True)

    print(f"{a.set} shard {a.shard}/{a.nshards}: {len(units)} units, recipe {recipe}", flush=True)
    if recipe == "locomo":
        questions = {u["id"]: generate_conversation(llm, chat, u, a.max_new_tokens) for u in units}
    else:
        questions = generate(llm, chat, recipe, units, a.rounds, a.max_new_tokens)
    with open(out, "w", encoding="utf-8") as f:
        for u in units:
            f.write(json.dumps({"id": u["id"], "questions": questions[u["id"]]}, ensure_ascii=False) + "\n")
    n = [len(questions[u["id"]]) for u in units]
    print(f"saved {sum(n)} questions, per unit mean {sum(n) / len(n):.1f} min {min(n)} max {max(n)} -> {out}", flush=True)
    exit_with_children(0)


if __name__ == "__main__":
    main()
