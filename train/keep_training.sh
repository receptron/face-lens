#!/bin/zsh
# Two rounds while the Bonsai teacher is still labeling:
#   1. all of FairFace → strong gender/age/race (plus whatever teacher labels exist), export
#   2. once label.py exits → fine-tune round 1 with every teacher label, export
# Each export lands in public/models/; reload the page to pick it up.
set -e
cd "$(dirname "$0")/.."
PY=$PWD/.venv/bin/python
echo "== $(date) prepare all of FairFace"
$PY teacher/prepare.py --train 100000 --val 20000 2>&1 | grep -v "^W0000\|^I0000\|INFO\|WARNING" | tail -1
cd train
echo "== $(date) round 1"
caffeinate -i $PY train.py --epochs ${ROUND1_EPOCHS:-6}
cp runs/best.pt runs/round1.pt; cp runs/best.json runs/round1.json
$PY export.py
echo "== $(date) round 1 exported; waiting for the teacher"
while pgrep -f "python label.py" > /dev/null; do sleep 120; done
echo "== $(date) round 2"
caffeinate -i $PY train.py --init runs/round1.pt --epochs ${ROUND2_EPOCHS:-5} --lr 4e-4
cp runs/best.pt runs/round2.pt; cp runs/best.json runs/round2.json
$PY export.py
echo "== $(date) done"
