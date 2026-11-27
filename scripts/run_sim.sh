#!/bin/bash
# RoDeL-Sim: the teacher and the student are updated in the same step.
#   round r:  one pass over the training documents; in every step the teacher synthesizes M answer sets, each
#             is distilled into a fresh knowledge adapter, and both adapters receive their GRPO update
#   test:     the trained teacher synthesizes for the test documents, the answer sets are distilled, and the
#             student answers the test questions without the documents.
# Finished stages are skipped when the script is started again.
#   bash scripts/run_sim.sh                 (settings: scripts/common.sh)
set -e
cd "$(dirname "$0")/.."
RUN=${RUN:-runs/sim}
source scripts/common.sh

T=""; S=""
for r in $(seq 1 $ROUNDS); do
  joint_update j$r "$T" "$S"
  T=$RUN/ckpt/j$r/epoch1/t; S=$RUN/ckpt/j$r/epoch1/s
  if [ "$TEST_EVERY_ROUND" = 1 ] || [ $r = $ROUNDS ]; then
    merge $T $RUN/ckpt/j$r/merged
    test_point j$r $RUN/ckpt/j$r/merged $S sim_r$r
  fi
done
log "RoDeL-Sim done"
