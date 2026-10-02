#!/bin/zsh
# Refresh everything after new captures: crop → teacher-label (only new crops) → train → export.
# Each step is resumable; re-running only does what is missing.
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python
$PY teacher/prepare.py --train 100000 --val 20000   # all of FairFace
(cd teacher && caffeinate -i ../$PY label.py)
(cd train && $PY train.py --epochs ${EPOCHS:-12} && $PY export.py)
