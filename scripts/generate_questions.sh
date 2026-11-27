#!/bin/bash
# Synthetic questions of every training and test document, written once into the data directory and used by
# every run (run_seq.sh and run_sim.sh generate the missing ones themselves).
#   bash scripts/generate_questions.sh      (settings: scripts/common.sh)
set -e
cd "$(dirname "$0")/.."
RUN=${RUN:-runs/questions}
source scripts/common.sh

questions "$TRAIN_SETS $TEST_SETS"
log "questions done"
