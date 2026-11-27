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
data/<set>.jsonl                documents and dataset questions of the training and test sets
data/<set>.questions.jsonl      synthetic questions of these documents
rodel/common.py                 prompts, output parsing, metrics, model helpers, file layout
rodel/question_prompts.py       prompts of the question-generation recipes
rodel/generate_questions.py     question generation: the synthetic questions of every document (vLLM)
rodel/synthesize.py             synthesis: teacher answers for the synthetic questions (vLLM)
rodel/distill.py                distillation: one LoRA_k per document
rodel/train_student.py          student update (RoDeL-Seq)
rodel/train_teacher.py          teacher update (RoDeL-Seq); with --joint, teacher and student together (RoDeL-Sim)
rodel/merge_lora.py             merge LoRA_t into the base model so that vLLM can serve the trained teacher
rodel/evaluate.py               closed-book evaluation
rodel/score.py                  F1 tables with bootstrap confidence intervals
scripts/common.sh               settings and pipeline stages
scripts/generate_questions.sh   question generation for the files without synthetic questions
scripts/run_seq.sh              RoDeL-Seq, training and test
scripts/run_sim.sh              RoDeL-Sim, training and test
```

## Installation

Two Python 3.12 environments are used: one for training, distillation and evaluation, and one with vLLM for
question generation and synthesis.

```bash
conda create -y -n rodel python=3.12
conda run -n rodel pip install torch==2.9.1 transformers==5.19.0 peft==0.21.2 nltk

conda create -y -n rodel-vllm python=3.12
conda run -n rodel-vllm pip install vllm==0.31.0 transformers==5.17.0
```

Optional, in the training environment: kernels for the linear-attention layers of Qwen3.5. Without them
`transformers` uses its PyTorch implementation of these layers. `causal-conv1d` is compiled during
installation and needs the CUDA toolkit (`nvcc`; set `CUDA_HOME` if it is not on `PATH`).

```bash
conda run -n rodel pip install flash-linear-attention==0.5.2 tilelang==0.1.15
conda run -n rodel pip install --no-build-isolation causal-conv1d==1.7.0
```

The vLLM 0.31.0 wheel on PyPI is built for CUDA 13 and needs an NVIDIA driver that supports it (580 or newer).
With an older driver, install the CUDA 12.9 build instead of the `pip install` line above:

```bash
conda run -n rodel-vllm pip install transformers==5.17.0 --extra-index-url https://download.pytorch.org/whl/cu129 \
    "https://github.com/vllm-project/vllm/releases/download/v0.31.0/vllm-0.31.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl"
conda run -n rodel-vllm pip install --no-deps "torchcodec==0.16.0+cu129" --index-url https://download.pytorch.org/whl/cu129
conda run -n rodel-vllm pip install "flashinfer-jit-cache==0.7.0.post1+cu129" --index-url https://flashinfer.ai/whl/cu129
```

Download the models before the first run, since every stage starts one process per GPU:

```bash
conda run -n rodel hf download Qwen/Qwen3.5-4B     # base model
conda run -n rodel hf download Qwen/Qwen3-4B       # question generator, only needed to generate synthetic questions
```

## Data

`data/` holds the documents, the dataset questions and the synthetic questions of the two training sets and
the ten test sets. A **unit** is what one knowledge adapter stores: a document, or the set of documents of a
multi-hop question.

| File | Source | Unit | Units | Dataset questions | Synthetic questions |
|---|---|---|---|---|---|
| `train_hotpotqa` | HotpotQA (distractor setting), training split | the paragraphs of one question (10) | 200, 20 of them for validation | 200 | 7,530 |
| `train_musique` | MuSiQue-Ans, training split | the paragraphs of one question (20) | 100, 10 of them for validation | 100 | 4,811 |
| `hotpotqa` | HotpotQA (distractor setting), validation split | the paragraphs of one question (10) | 200 | 200 | 7,499 |
| `musique` | MuSiQue-Ans, validation split | the paragraphs of one question (20) | 200 | 200 | 9,684 |
| `squad` | SQuAD v1.1, validation split | one paragraph | 200 | 1,118 | 3,725 |
| `2wiki` | 2WikiMultihopQA, validation split | the paragraphs of one question (10) | 200 | 200 | 7,538 |
| `nq` | Natural Questions, validation split | one Wikipedia page | 200 | 218 | 16,968 |
| `locomo` | LoCoMo | one conversation, its sessions as parts | 10 | 1,540 | 11,360 |
| `qasper` | QASPER, test split | one paper, its sections as parts | 200 | 549 | 10,653 |
| `narrativeqa` | NarrativeQA, test split | one story | 52 | 1,568 | 18,047 |
| `multifieldqa` | MultiFieldQA-en (LongBench) | one document | 80 | 105 | 3,174 |
| `triviaqa` | TriviaQA (Wikipedia), training split | one Wikipedia page | 200 | 643 | 16,689 |

The units are samples of these sources. Only answerable questions with a reference answer shorter than 20
tokens are kept (LoCoMo: its four answerable categories), and documents are shorter than 20k tokens
(NarrativeQA: 28k; LoCoMo conversations are kept whole). Pages, stories and long documents are split into
parts of 1,500 tokens. The files are derived from the public releases of the datasets and remain under the
licenses of their sources.

### Format

`data/<set>.jsonl` holds one unit per line:

```json
{"id": "doc-1",
 "split": "train",
 "paragraphs": [{"title": "...", "text": "..."}],
 "official": [{"id": "q-1", "question": "...", "answers": ["...", "..."]}]}
```

- `paragraphs`: the document, as one or more titled parts.
- `official`: the dataset questions about the document with their reference answers. They are answered by
  the student only; the teacher never sees them.
- `split`: training files only. `val` units are held out and used for the validation readings in the training
  logs.
- `type` (HotpotQA, 2WikiMultihopQA): the type of the dataset question; it selects the style of the
  multi-document synthetic questions. `speakers` (LoCoMo): the two speakers of the conversation.
- LoCoMo questions carry their category in `"extra": {"category": n}`, which the scorer uses.

Training files are named `train_<dataset>`. The dataset name selects the sentence that introduces the
document in the teacher's prompt (`CONTEXT_INTRO` in `rodel/common.py`) and the question-generation recipe.

### Synthetic questions

`data/<set>.questions.jsonl` holds the synthetic questions of each unit:

```json
{"id": "doc-1", "questions": [{"q": "...", "route": "S"}, {"q": "...", "route": "R"}]}
```

They are written by a language model that reads the document only, before training, and stay fixed: all
runs and both schedules use the same files. They are the only questions used for synthesis and distillation.
The files in `data/` were written by Qwen3-4B in thinking mode. `scripts/generate_questions.sh` writes the
files that are missing in the data directory with the same procedure, for example for other documents; the
reasoning of the generator (`QGEN_MODEL`) has to finish within `QGEN_CAP` tokens. Each dataset has a recipe
(`rodel/generate_questions.py`, prompts in `rodel/question_prompts.py`):

| Recipe | Files | Questions per unit |
|---|---|---|
| `hotpotqa` | `train_hotpotqa`, `hotpotqa` | 3 per paragraph; 10 multi-document questions (bridge or comparison, by `type`) |
| `musique` | `train_musique`, `musique` | 2 per paragraph; 10 multi-hop questions |
| `2wiki` | `2wiki` | 3 per paragraph; 10 multi-document questions of the unit's `type` |
| `squad` | `squad` | 20 |
| `nq` | `nq`, `triviaqa` | 20 page-level questions; 15 search queries; 12 per part, at most 96 |
| `qasper` | `qasper` | up to 3 per section, at most 40; 10 cross-section, 6 yes/no and 4 abstractive questions |
| `narrativeqa` | `narrativeqa` | 15 event questions and 15 character and detail questions per part; 40 whole-story questions |
| `multifieldqa` | `multifieldqa` | 5 per part, at most 45; 10 document-level and 5 explanatory questions |
| `locomo` | `locomo` | 30 per session; 120 multi-hop, 120 temporal and 80 open-domain questions |

A file of another dataset needs a recipe: `python rodel/generate_questions.py --set <set> --recipe <recipe>`.

## Running

```bash
export PY=/path/to/envs/rodel/bin/python
export PY_VLLM=/path/to/envs/rodel-vllm/bin/python

bash scripts/run_seq.sh                # RoDeL-Seq: training, then test -> runs/seq/results_seq_r3.md
bash scripts/run_sim.sh                # RoDeL-Sim: training, then test -> runs/sim/results_sim_r3.md
```

Both scripts skip what is already finished when they are started again. They first generate the synthetic
questions that are missing in the data directory (none for the files in `data/`);
`bash scripts/generate_questions.sh` runs this step alone. Settings are environment variables (defaults in
`scripts/common.sh`):

| Variable | Default | Meaning |
|---|---|---|
| `MODEL` | `Qwen/Qwen3.5-4B` | base model (for example `Qwen/Qwen3.5-2B`, `Qwen/Qwen3.5-0.8B`) |
| `CAP` | `1024` | length limit, in tokens, of every generation of the base model: teacher answers and student answers, in training and at test time |
| `QGEN_MODEL` | `Qwen/Qwen3-4B` | model that writes synthetic questions that are missing |
| `QGEN_CAP` | `8192` | length limit, in tokens, of one question-generation output (reasoning trace and list of questions) |
| `ROUNDS` | `3` | training rounds `K` |
| `GPUS` | `0,1,2,3,4,5,6,7` | GPUs to use; one process per GPU |
| `DATA` | `data` | data directory |
| `RUN` | `runs/seq`, `runs/sim` | output directory |
| `TRAIN_SETS` | `train_hotpotqa train_musique` | training files |
| `ID_SETS` | `hotpotqa musique` | test files from the training datasets |
| `OOD_SETS` | `squad 2wiki nq locomo qasper narrativeqa multifieldqa triviaqa` | test files from other datasets |
| `TEST_EVERY_ROUND` | `0` | `1` runs the test pipeline after every round instead of after the last one only |
| `EVAL_CONFIGS` | `ks` | `ks`: trained student; `k,ks` also evaluates the untrained student with the same LoRA_k |
| `DISTILL_JOBS`, `EVAL_JOBS` | `1`, `3` | files distilled / evaluated at the same time; each one adds a process on every GPU, so raise them only if the GPU memory allows |
| `PORT` | `29677` | port of the multi-GPU training processes; change it when two runs share a machine |

A training step takes one document per process, so the number of GPUs is the number of documents per step.

```bash
MODEL=Qwen/Qwen3.5-2B CAP=2048 GPUS=0,1,2,3 RUN=runs/seq_2b bash scripts/run_seq.sh
```

### Quick check

A run on a few units with a few synthetic questions each exercises every stage of training and test, and is
a way to test an installation before a full run:

```bash
$PY - <<'EOF'
import json, os
os.makedirs("data_small", exist_ok=True)
for s in ("train_hotpotqa", "train_musique", "hotpotqa", "squad"):      # 11 units per file, 8 synthetic questions per unit
    for suffix, keep in ((".jsonl", None), (".questions.jsonl", 8)):
        with open(f"data/{s}{suffix}", encoding="utf-8") as f, open(f"data_small/{s}{suffix}", "w", encoding="utf-8") as g:
            for line in list(f)[:11]:
                row = json.loads(line)
                if keep:
                    row["questions"] = row["questions"][:keep]
                g.write(json.dumps(row, ensure_ascii=False) + "\n")
EOF
DATA=data_small ID_SETS=hotpotqa OOD_SETS=squad ROUNDS=1 RUN=runs/check_seq bash scripts/run_seq.sh
DATA=data_small ID_SETS=hotpotqa OOD_SETS=squad ROUNDS=1 RUN=runs/check_sim bash scripts/run_sim.sh
```

Question generation can be tested on its own, for example on eight units of one file:

```bash
mkdir -p data_questions && head -n 8 data/squad.jsonl > data_questions/squad.jsonl
CUDA_VISIBLE_DEVICES=0 $PY_VLLM rodel/generate_questions.py --set squad --data-dir data_questions
```

### Stages

Each stage can also be run on its own; `--help` lists the options of a script.

| Stage | Script | What it does |
|---|---|---|
| question generation | `generate_questions.py` | A language model reads each document and writes the synthetic questions of its recipe. Lines that are not well-formed, self-contained questions and repetitions are dropped; prompts that fall short of their number of questions are sampled again. |
| synthesis | `synthesize.py` | The teacher (base model, or base model with LoRA_t merged in) answers every synthetic question of a document once. Answers whose reasoning trace is not closed within the length limit, or that have no answer line after it, are dropped. |
| distillation | `distill.py` | A new LoRA_k per document is trained on the kept answers. The prompt contains the question only; the target is the teacher's full output. |
| student update | `train_student.py` | With the LoRA_k of the training documents fixed, the student samples `G` answers per dataset question; the advantage is normalised within the `G` answers of a question. |
| teacher update | `train_teacher.py` | The teacher samples `M` answer sets per document; each is distilled into a fresh LoRA_k and scored by the student's greedy closed-book answers minus a penalty on invalid answers; the advantage is normalised over the `M` sets and shared by all answers of a set. With `--joint` the student is updated in the same step. |
| merge | `merge_lora.py` | Merges LoRA_t into the base model for synthesis with vLLM. |
| evaluation | `evaluate.py` | Attaches the LoRA_k of each test document and answers its questions without the document. |
| scoring | `score.py` | Token F1 per test set, the average over sets and a bootstrap confidence interval. |

### Outputs

```
data/<set>.questions.jsonl                         synthetic questions
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
| Length limit | 1024 tokens for every generation of the base model |
| Valid answer | the reasoning trace is closed within the length limit and a non-empty line follows; the answer is that line |
| Answer sets per document `M` / student samples per question `G` | 4 / 4 |
| Student reward | token F1 against the closest reference; -2 for an invalid answer |
| Teacher reward | mean student reward (greedy answers) minus 2 x the share of invalid answers in the set |
| LoRA_t, LoRA_s optimisation | AdamW, learning rate 1e-5, no weight decay, gradient clipping 1.0; one pass over the training documents per update |
| Decoding in the RL updates | sampling at temperature 1.0 without truncation |
| Decoding elsewhere | temperature 0.6, top-p 0.95, top-k 20 |
| Rounds `K` | 3 |
| Question generation | Qwen3-4B in thinking mode, temperature 0.6, top-p 0.95, top-k 20; at most 8192 tokens per output (LoCoMo: 3072 per session prompt, 2560 per conversation prompt); up to 3 sampling rounds per prompt |
