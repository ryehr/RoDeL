#!/bin/bash
# Settings and pipeline stages shared by generate_questions.sh, run_seq.sh and run_sim.sh (sourced by them).
# Every setting can be overridden from the environment, e.g.  MODEL=Qwen/Qwen3.5-2B CAP=2048 bash scripts/run_seq.sh

MODEL=${MODEL:-Qwen/Qwen3.5-4B}          # base model
CAP=${CAP:-1024}                         # length limit, in tokens, of every generation (teacher and student)
QGEN_MODEL=${QGEN_MODEL:-Qwen/Qwen3-4B}    # model that writes the synthetic questions
QGEN_CAP=${QGEN_CAP:-8192}               # length limit, in tokens, of one question-generation output
ROUNDS=${ROUNDS:-3}                      # training rounds
GPUS=${GPUS:-0,1,2,3,4,5,6,7}            # one process per GPU; a training step takes one document per process
DATA=${DATA:-data}                       # unit files; the synthetic questions are written next to them
PY=${PY:-python}                         # interpreter of the training environment
PY_VLLM=${PY_VLLM:-python}               # interpreter of the vLLM environment (question generation, synthesis)
TRAIN_SETS=${TRAIN_SETS:-"train_hotpotqa train_musique"}
ID_SETS=${ID_SETS:-"hotpotqa musique"}   # test sets from the training datasets
OOD_SETS=${OOD_SETS:-"squad 2wiki nq locomo qasper narrativeqa multifieldqa triviaqa"}
TEST_EVERY_ROUND=${TEST_EVERY_ROUND:-0}  # 1 = run the test pipeline after every round, 0 = after the last round only
EVAL_CONFIGS=${EVAL_CONFIGS:-ks}         # ks = trained student; k,ks also evaluates the untrained student
DISTILL_JOBS=${DISTILL_JOBS:-1}          # unit files distilled at the same time (each one adds a process on every GPU)
EVAL_JOBS=${EVAL_JOBS:-3}                # unit files evaluated at the same time (each one adds a process on every GPU)
PORT=${PORT:-29677}                      # port of torch.distributed.run

TEST_SETS="$ID_SETS $OOD_SETS"
IFS=, read -ra GPU <<< "$GPUS"; NGPU=${#GPU[@]}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
mkdir -p $RUN/logs

log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
nunits(){ grep -c "" $DATA/$1.jsonl; }
nrows(){ cat "$@" 2>/dev/null | wc -l; }
wait_all(){ local fail=0 p; for p in "${pids[@]}"; do wait $p || fail=1; done
  if [ $fail = 1 ]; then log "$1 FAILED (see $RUN/logs)"; exit 1; fi; }

questions(){ # $1 unit files: synthetic questions that are not in $DATA yet -> $DATA/<set>.questions.jsonl
  local S i
  for S in $1; do
    if [ "$(nrows $DATA/$S.questions.jsonl)" -ge "$(nunits $S)" ]; then continue; fi
    log "questions $S"; pids=()
    for i in "${!GPU[@]}"; do
      CUDA_VISIBLE_DEVICES=${GPU[$i]} $PY_VLLM rodel/generate_questions.py --set $S --model $QGEN_MODEL --max-new-tokens $QGEN_CAP \
        --data-dir $DATA --shard $i --nshards $NGPU > $RUN/logs/questions_$S.shard$i.log 2>&1 & pids+=($!)
    done
    wait_all "questions $S"
    for i in "${!GPU[@]}"; do cat $DATA/$S.questions.shard$i.jsonl; done > $DATA/$S.questions.tmp
    mv $DATA/$S.questions.tmp $DATA/$S.questions.jsonl; rm -f $DATA/$S.questions.shard*.jsonl
  done
}

synthesize(){ # $1 answers tag, $2 teacher (base model, or directory with the teacher adapter merged in), $3 unit files
  local S i
  for S in $3; do
    if [ "$(nrows $RUN/answers/$1/$S.shard*.jsonl)" -ge "$(nunits $S)" ]; then continue; fi
    log "synthesize [$1] $S"; pids=()
    for i in "${!GPU[@]}"; do
      CUDA_VISIBLE_DEVICES=${GPU[$i]} $PY_VLLM rodel/synthesize.py --set $S --tag $1 --model $2 --max-new-tokens $CAP \
        --data-dir $DATA --run-dir $RUN --shard $i --nshards $NGPU > $RUN/logs/synthesize_$1_$S.shard$i.log 2>&1 & pids+=($!)
    done
    wait_all "synthesize [$1] $S"
  done
}

distill(){ # $1 answers tag (also names the knowledge adapters), $2 unit files
  local todo=() S i n
  for S in $2; do
    if [ "$(nrows $RUN/knowledge/$1/$S.shard*.jsonl)" -lt "$(nunits $S)" ]; then todo+=($S); fi
  done
  for ((n = 0; n < ${#todo[@]}; n += DISTILL_JOBS)); do
    log "distill [$1] ${todo[*]:n:DISTILL_JOBS}"; pids=()
    for S in "${todo[@]:n:DISTILL_JOBS}"; do
      for i in "${!GPU[@]}"; do
        CUDA_VISIBLE_DEVICES=${GPU[$i]} $PY rodel/distill.py --set $S --tag $1 --model $MODEL --max-new-tokens $CAP \
          --data-dir $DATA --run-dir $RUN --shard $i --nshards $NGPU > $RUN/logs/distill_$1_$S.shard$i.log 2>&1 & pids+=($!)
      done
    done
    wait_all "distill [$1] ${todo[*]:n:DISTILL_JOBS}"
  done
}

evaluate(){ # $1 knowledge tag, $2 predictions tag, $3 student adapter (LoRA_s directory), $4 unit files
  local todo=() S i n
  for S in $4; do
    if [ "$(nrows $RUN/preds/$2/$S.shard*.jsonl)" -lt "$(nunits $S)" ]; then todo+=($S); fi
  done
  for ((n = 0; n < ${#todo[@]}; n += EVAL_JOBS)); do
    log "evaluate [$2] ${todo[*]:n:EVAL_JOBS}"; pids=()
    for S in "${todo[@]:n:EVAL_JOBS}"; do
      for i in "${!GPU[@]}"; do
        CUDA_VISIBLE_DEVICES=${GPU[$i]} $PY rodel/evaluate.py --set $S --knowledge-tag $1 --tag $2 ${3:+--student $3} --configs $EVAL_CONFIGS \
          --model $MODEL --max-new-tokens $CAP --data-dir $DATA --run-dir $RUN --shard $i --nshards $NGPU \
          > $RUN/logs/evaluate_$2_$S.shard$i.log 2>&1 & pids+=($!)
      done
    done
    wait_all "evaluate [$2] ${todo[*]:n:EVAL_JOBS}"
  done
  for S in $4; do
    if [ "$(nrows $RUN/preds/$2/$S.shard*.jsonl)" -lt "$(nunits $S)" ]; then log "evaluate [$2] $S incomplete"; exit 1; fi
  done
}

score(){ # $1 predictions tag: F1 tables -> $RUN/results_$1.md
  local points=() c
  for c in ${EVAL_CONFIGS//,/ }; do points+=($1.$c); done
  { [ -z "$ID_SETS" ] || $PY rodel/score.py --run-dir $RUN --points "${points[@]}" --sets $ID_SETS --label in-domain
    echo
    [ -z "$OOD_SETS" ] || $PY rodel/score.py --run-dir $RUN --points "${points[@]}" --sets $OOD_SETS --label out-of-domain
  } > $RUN/results_$1.md
  cat $RUN/results_$1.md
}

merge(){ # $1 teacher adapter (LoRA_t directory), $2 output directory
  if [ -f $2/tokenizer_config.json ]; then return; fi
  log "merge $1 -> $2"
  $PY rodel/merge_lora.py --model $MODEL --adapter $1 --out $2 > $RUN/logs/merge_$(basename $(dirname $2)).log 2>&1 \
    || { log "merge FAILED (see $RUN/logs)"; exit 1; }
}

test_point(){ # $1 answers tag, $2 teacher (base model or merged directory), $3 student adapter, $4 predictions tag
  synthesize $1 $2 "$TEST_SETS"; distill $1 "$TEST_SETS"; evaluate $1 $4 $3 "$TEST_SETS"; score $4
}

train(){ # $1 name, $2 script, $3 adapter that marks the update as finished, $4.. arguments
  local NAME=$1 SCRIPT=$2 OUT=$RUN/ckpt/$1 DONE=$3; shift 3
  if [ -f $OUT/epoch1/$DONE/adapter_model.safetensors ]; then log "$NAME exists, skip"; return; fi
  if [ -d $OUT ]; then mv $OUT ${OUT}_partial_$(date +%m%d%H%M%S); fi
  log "$NAME: $SCRIPT $*"
  CUDA_VISIBLE_DEVICES=$GPUS $PY -m torch.distributed.run --nproc_per_node $NGPU --master_port $PORT rodel/$SCRIPT \
    --model $MODEL --max-new-tokens $CAP --sets $TRAIN_SETS --data-dir $DATA --out $OUT "$@" > $RUN/logs/train_$NAME.log 2>&1 \
    || { log "$NAME FAILED (see $RUN/logs/train_$NAME.log)"; exit 1; }
  log "$NAME done: $(grep -h 'VAL epoch1' $RUN/logs/train_$NAME.log | head -n 2 | tail -n 1 | cut -c1-160)"
}

student_update(){ # $1 name, $2 knowledge tag of the training documents, $3 LoRA_s to continue from ("" = new adapter)
  train $1 train_student.py s --run-dir $RUN --knowledge-tag $2 ${3:+--init-student $3}
}
teacher_update(){ # $1 name, $2 LoRA_t to continue from ("" = new adapter), $3 frozen student (LoRA_s directory)
  train $1 train_teacher.py t --init-student $3 ${2:+--init-teacher $2}
}
joint_update(){ # $1 name, $2 LoRA_t to continue from, $3 LoRA_s to continue from ("" = new adapters)
  train $1 train_teacher.py s --joint ${2:+--init-teacher $2} ${3:+--init-student $3}
}
