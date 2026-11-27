"""Synthesis: the teacher reads each document and answers every synthetic question of that document once.

The teacher is served by vLLM. It is either the base model (untrained teacher) or a directory in which the
teacher adapter has been merged into the base model (merge_lora.py). An answer is kept if its reasoning
trace is closed within the length limit and followed by a non-empty answer line; the kept answers of a
document form its answer set.

  CUDA_VISIBLE_DEVICES=0 python rodel/synthesize.py --set hotpotqa --tag t0 --run-dir runs/seq --shard 0 --nshards 8

Writes {run-dir}/answers/{tag}/{set}.shard{i}.jsonl, one row per document: {"id", "pairs": [{"q", "full", "ans"}]}
("full" = reasoning trace + answer, the distillation target; "ans" = the answer line).
"""
import argparse
import json
import os
import signal
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (DEFAULT_MAX_NEW_TOKENS, DEFAULT_MODEL, INFER_SAMPLING, answers_path, dataset_of, load_questions,
                    load_units, teacher_prompt, vllm_model_kwargs)

LONG_DOCUMENT_SETS = ("narrativeqa", "nq", "locomo")     # served with a 40960-token context window, the others with 20480


def exit_with_children(code=0):
    """Terminate the vLLM engine processes started by this process, then exit without interpreter teardown."""
    sys.stdout.flush(); sys.stderr.flush()
    me, kids = os.getpid(), []
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                with open(f"/proc/{d}/stat") as f:
                    if int(f.read().split(")")[-1].split()[1]) == me:
                        kids.append(int(d))
            except Exception:
                pass
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for k in kids:
            try:
                os.kill(k, sig)
            except Exception:
                pass
        if sig == signal.SIGTERM:
            time.sleep(3)
    os._exit(code)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True, help="unit file {data-dir}/{set}.jsonl with questions {set}.questions.jsonl")
    ap.add_argument("--tag", required=True, help="name of the answer sets (one per teacher)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="base model, or a directory with the teacher adapter merged in")
    ap.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS, help="length limit of an answer")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="only the first N units of the shard")
    ap.add_argument("--max-model-len", type=int, default=0, help="vLLM context window (0 = by dataset)")
    ap.add_argument("--gpu-mem", type=float, default=0.90, help="vLLM gpu_memory_utilization")
    a = ap.parse_args()

    units = load_units(a.data_dir, a.set)[a.shard::a.nshards]
    if a.limit:
        units = units[:a.limit]
    questions = load_questions(a.data_dir, a.set)
    out = answers_path(a.run_dir, a.tag, a.set, a.shard)
    if os.path.exists(out) and sum(1 for _ in open(out, encoding="utf-8")) >= len(units):
        print("already complete", out)
        return
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if not units:                          # more shards than units
        open(out, "w").close()
        return
    if "VLLM_PORT" not in os.environ:      # a separate port range per GPU, so that engines started together do not collide
        os.environ["VLLM_PORT"] = str(42000 + 50 * int((os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",")[0]))

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    tok = AutoTokenizer.from_pretrained(a.model)
    max_len = a.max_model_len or (40960 if dataset_of(a.set) in LONG_DOCUMENT_SETS else 20480)
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=max_len, gpu_memory_utilization=a.gpu_mem,
              enable_prefix_caching=True, **vllm_model_kwargs(a.model))
    sp = SamplingParams(n=1, max_tokens=a.max_new_tokens, seed=42, **INFER_SAMPLING)

    jobs, prompts = [], []
    for u in units:
        for q in questions.get(u["id"], []):
            prompts.append(tok.apply_chat_template([{"role": "user", "content": teacher_prompt(u, q)}], tokenize=False,
                                                   add_generation_prompt=True, enable_thinking=True))
            jobs.append((u["id"], q))
    print(f"{a.set} shard {a.shard}/{a.nshards}: {len(units)} units, {len(prompts)} prompts", flush=True)
    outs = llm.generate(prompts, sp)

    pairs = {u["id"]: [] for u in units}
    dropped = 0
    for (uid, q), o in zip(jobs, outs):
        text = o.outputs[0].text
        if "</think>" not in text:
            dropped += 1
            continue
        ans = text.split("</think>", 1)[1].strip()
        ans = ans.splitlines()[0].strip() if ans else ""
        if not ans:
            dropped += 1
            continue
        pairs[uid].append({"q": q, "full": text, "ans": ans})
    with open(out, "w", encoding="utf-8") as f:
        for u in units:
            f.write(json.dumps({"id": u["id"], "pairs": pairs[u["id"]]}, ensure_ascii=False) + "\n")
    print(f"saved {sum(len(v) for v in pairs.values())} pairs (dropped {dropped}) -> {out}", flush=True)
    exit_with_children(0)


if __name__ == "__main__":
    main()
