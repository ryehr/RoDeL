# RoDeL: Role-Decoupled Learning for Context Distillation

RoDeL is a training method for closed-book question answering over documents. A document is stored in its own
LoRA by context distillation, and the two roles involved in that process are trained separately:

- The **teacher** reads the document and answers a fixed set of synthetic questions about it (**synthesis**).
- The teacher's answers are fitted into a document-specific **knowledge adapter** (LoRA_k) with the document
  removed from the prompt (**distillation**).
- The **student** answers questions about the document with LoRA_k attached and without the document.

One frozen base model carries three LoRA adapters:

| Adapter | Role | Training |
|---|---|---|
| LoRA_t | teacher: writes the answer sets that are distilled | set-level GRPO; the reward of an answer set is the student's closed-book score after the set has been distilled |
| LoRA_k | knowledge of one document, a new adapter per document | cross-entropy on the teacher's answers (reasoning trace + answer) |
| LoRA_s | student: answers with LoRA_k attached | GRPO; the reward is the token F1 of its closed-book answer |

Two training schedules are provided:

- **RoDeL-Seq** (`scripts/run_seq.sh`): student update and teacher update in turns, `s1 -> t1 -> ... -> sK -> tK`;
  one adapter is trained per update and the other is frozen.
- **RoDeL-Sim** (`scripts/run_sim.sh`): the teacher and the student are updated in the same step.

At test time both schedules run the same pipeline: the trained teacher synthesizes an answer set for each new
document, the set is distilled into LoRA_k, and the trained student answers the test questions.

## Repository layout

```
rodel/common.py          prompts, output parsing, metrics, model helpers, file layout
rodel/synthesize.py      synthesis: teacher answers for the synthetic questions (vLLM)
rodel/distill.py         distillation: one LoRA_k per document
rodel/train_student.py   student update (RoDeL-Seq)
rodel/train_teacher.py   teacher update (RoDeL-Seq); with --joint, teacher and student together (RoDeL-Sim)
rodel/merge_lora.py      merge LoRA_t into the base model so that vLLM can serve the trained teacher
rodel/evaluate.py        closed-book evaluation
rodel/score.py           F1 tables with bootstrap confidence intervals
scripts/common.sh        settings and pipeline stages
scripts/run_seq.sh       RoDeL-Seq, training and test
scripts/run_sim.sh       RoDeL-Sim, training and test
```

## Installation

Two Python 3.12 environments are used: one for training, distillation and evaluation, and one with vLLM for
synthesis.

```bash
conda create -y -n rodel python=3.12
conda run -n rodel pip install torch==2.9.1 transformers==5.19.0 peft==0.21.2 nltk \
    flash-linear-attention==0.5.2 tilelang==0.1.15
conda run -n rodel pip install --no-build-isolation causal-conv1d==1.7.0

conda create -y -n rodel-vllm python=3.12
conda run -n rodel-vllm pip install vllm==0.31.0
```

Download the base model before the first run, since every stage starts one process per GPU:

```bash
hf download Qwen/Qwen3.5-4B
```

## Data

The repository contains no data. Every dataset is a pair of JSON-lines files in `data/`.

`data/<set>.jsonl` holds one document (unit) per line:

```json
{"id": "doc-1",
 "paragraphs": [{"title": "...", "text": "..."}],
 "official": [{"id": "q-1", "question": "...", "answers": ["...", "..."]}],
 "split": "train"}
```

- `paragraphs`: the document, as one or more titled parts.
- `official`: the dataset's own questions about the document with their reference answers. They are answered
  by the student only; the teacher never sees them.
- `split`: training files only. `val` units are held out and used for the validation readings in the training logs.

`data/<set>.questions.jsonl` holds the synthetic questions of each document:

```json
{"id": "doc-1", "questions": [{"q": "..."}, {"q": "..."}]}
```

Synthetic questions are generated from the document alone, before training, and stay fixed. They are the only
questions used for synthesis and distillation.

Training files are named `train_<dataset>`. The dataset name selects the sentence that introduces the
document in the teacher's prompt (`CONTEXT_INTRO` in `rodel/common.py`). LoCoMo questions carry their category
in `"extra": {"category": n}`, which the scorer uses.

Default file names (`scripts/common.sh`):

| Role | Sets |
|---|---|
| training | `train_hotpotqa`, `train_musique` |
| test, training datasets | `hotpotqa`, `musique` |
| test, other datasets | `squad`, `2wiki`, `nq`, `locomo`, `qasper`, `narrativeqa`, `multifieldqa`, `triviaqa` |

## Running

```bash
export PY=/path/to/envs/rodel/bin/python
export PY_VLLM=/path/to/envs/rodel-vllm/bin/python

bash scripts/run_seq.sh        # RoDeL-Seq
bash scripts/run_sim.sh        # RoDeL-Sim
```

Both scripts skip the stages that are already finished when they are started again. Settings are environment
variables (defaults in `scripts/common.sh`):

| Variable | Default | Meaning |
|---|---|---|
| `MODEL` | `Qwen/Qwen3.5-4B` | base model (for example `Qwen/Qwen3.5-2B`, `Qwen/Qwen3.5-0.8B`) |
| `CAP` | `1024` | length limit, in tokens, of every generation: teacher answers and student answers, in training and at test time |
| `ROUNDS` | `3` | training rounds `K` |
| `GPUS` | `0,1,2,3,4,5,6,7` | GPUs to use; one process per GPU |
| `DATA` | `data` | data directory |
| `RUN` | `runs/seq`, `runs/sim` | output directory |
| `TRAIN_SETS`, `ID_SETS`, `OOD_SETS` | see above | training and test files |
| `TEST_EVERY_ROUND` | `0` | `1` runs the test pipeline after every round instead of after the last one only |
| `EVAL_CONFIGS` | `ks` | `ks`: trained student; `k,ks` also evaluates the untrained student with the same LoRA_k |
| `DISTILL_JOBS`, `EVAL_JOBS` | `1`, `3` | files distilled / evaluated at the same time; each one adds a process on every GPU, so raise them only if the GPU memory allows |

A training step takes one document per process, so the number of GPUs is the number of documents per step.

```bash
MODEL=Qwen/Qwen3.5-2B CAP=2048 GPUS=0,1,2,3 RUN=runs/seq_2b bash scripts/run_seq.sh
```

### Stages

Each stage can also be run on its own; `--help` lists the options of a script.

| Stage | Script | What it does |
|---|---|---|
| synthesis | `synthesize.py` | The teacher (base model, or base model with LoRA_t merged in) answers every synthetic question of a document once. Answers whose reasoning trace is not closed within the length limit, or that have no answer line after it, are dropped. |
| distillation | `distill.py` | A new LoRA_k per document is trained on the kept answers. The prompt contains the question only; the target is the teacher's full output. |
| student update | `train_student.py` | With the LoRA_k of the training documents fixed, the student samples `G` answers per dataset question; the advantage is normalised within the `G` answers of a question. |
| teacher update | `train_teacher.py` | The teacher samples `M` answer sets per document; each is distilled into a fresh LoRA_k and scored by the student's greedy closed-book answers minus a penalty on invalid answers; the advantage is normalised over the `M` sets and shared by all answers of a set. With `--joint` the student is updated in the same step. |
| merge | `merge_lora.py` | Merges LoRA_t into the base model for synthesis with vLLM. |
| evaluation | `evaluate.py` | Attaches the LoRA_k of each test document and answers its questions without the document. |
| scoring | `score.py` | Token F1 per test set, the average over sets and a bootstrap confidence interval. |

### Outputs

```
runs/<run>/answers/<tag>/<set>.shard<i>.jsonl      teacher answer sets (<tag>: t0 = untrained teacher, t1, t2, ... / j1, j2, ...)
runs/<run>/knowledge/<tag>/<set>/<unit id>/        knowledge adapters distilled from those answer sets
runs/<run>/ckpt/<update>/epoch1/{t,s}              trained teacher / student adapters (s1, t1, ... / j1, ...)
runs/<run>/ckpt/<update>/merged                    base model with LoRA_t merged in
runs/<run>/preds/<tag>/<set>.shard<i>.jsonl        student predictions
runs/<run>/results_<tag>.md                        F1 tables
runs/<run>/logs/                                   one log per process
```

## Default settings

| Item | Value |
|---|---|
| Base model | Qwen3.5-4B, frozen, text model only, native thinking mode (reasoning trace, then answer) |
| LoRA modules | every linear layer of the text model: attention and MLP projections, projections of the linear-attention layers |
| LoRA_t, LoRA_s | rank 64, alpha 128, no dropout |
| LoRA_k | rank 16, alpha 32, dropout 0.05; AdamW, learning rate 2e-4, weight decay 0.01, batch size 4, 5 epochs, gradient clipping 1.0 |
| Length limit | 1024 tokens for every generation |
| Valid answer | the reasoning trace is closed within the length limit and a non-empty line follows; the answer is that line |
| Answer sets per document `M` / student samples per question `G` | 4 / 4 |
| Student reward | token F1 against the closest reference; -2 for an invalid answer |
| Teacher reward | mean student reward (greedy answers) minus 2 x the share of invalid answers in the set |
| LoRA_t, LoRA_s optimisation | AdamW, learning rate 1e-5, no weight decay, gradient clipping 1.0; one pass over the training documents per update |
| Decoding in the RL updates | sampling at temperature 1.0 without truncation |
| Decoding elsewhere | temperature 0.6, top-p 0.95, top-k 20 |
| Rounds `K` | 3 |
