#!/bin/bash
# RoDeL-Seq: the student and the teacher are updated in turns, s1 -> t1 -> ... -> sK -> tK.
#   round r:  the current teacher synthesizes answer sets for the training documents and they are distilled
#             into knowledge adapters  ->  student update s_r  ->  teacher update t_r (student frozen)
#   test:     the trained teacher synthesizes for the test documents, the answer sets are distilled, and the
#             student answers the test questions without the documents.
# Synthetic questions that are not in the data directory yet are generated first (scripts/generate_questions.sh).
# Finished stages are skipped when the script is started again.
#   bash scripts/run_seq.sh                 (settings: scripts/common.sh)
set -e
cd "$(dirname "$0")/.."
RUN=${RUN:-runs/seq}
source scripts/common.sh

questions "$TRAIN_SETS $TEST_SETS"
T=""; S=""; TEACHER=$MODEL; TAG=t0
for r in $(seq 1 $ROUNDS); do
  synthesize $TAG $TEACHER "$TRAIN_SETS"; distill $TAG "$TRAIN_SETS"
  student_update s$r $TAG "$S";  S=$RUN/ckpt/s$r/epoch1/s
  teacher_update t$r "$T" $S;    T=$RUN/ckpt/t$r/epoch1/t
  TAG=t$r; TEACHER=$RUN/ckpt/t$r/merged
  merge $T $TEACHER
  if [ "$TEST_EVERY_ROUND" = 1 ] || [ $r = $ROUNDS ]; then test_point $TAG $TEACHER $S seq_r$r; fi
done
log "RoDeL-Seq done"
