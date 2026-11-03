#!/bin/bash
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
#
# SLURM template for scripts/tempo/run_matrix.py. Everything is parameterized through environment variables, so the
# same file serves single jobs and job arrays. GPU count must match on both sides: --gres sets the allocation, GPUS
# builds the device string (GPUS>1 trains with DDP).
#
#   # whole matrix on one 4-GPU node
#   sbatch --gres=gpu:4 --export=ALL,GPUS=4,DATASETS="data/stvd_tempo.yaml data/mot_tempo.yaml" scripts/tempo/slurm_matrix.sbatch
#
#   # one array task per config (7 configs -> --array=0-6), 1 GPU each, all appending to the same locked CSV
#   sbatch --array=0-6 --gres=gpu:1 --export=ALL,GPUS=1,DATASETS="data/visdrone_vid_tempo.yaml" scripts/tempo/slurm_matrix.sbatch
#
#   # subset of models/configs, shorter schedule
#   sbatch --gres=gpu:1 --export=ALL,MODELS=yolo26n,CONFIGS="n1 n3s1",EPOCHS=30 scripts/tempo/slurm_matrix.sbatch
#
# Variables (defaults in brackets): DATASETS [data/stvd_tempo.yaml], MODELS [yolo26n yolo26s], CONFIGS [all 7],
# GPUS [1], IMGSZ [640], EPOCHS [100], BATCH [16], SEED [0], WORKERS [8], PROJECT [runs/tempo],
# OUT [results/matrix.csv], EXTRA [extra run_matrix.py arguments, e.g. --scratch], VENV [virtualenv to activate].
# With a job array, task i runs only the i-th entry of CONFIGS.

#SBATCH --job-name=tempo-matrix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail

DATASETS=${DATASETS:-data/stvd_tempo.yaml}
MODELS=${MODELS:-yolo26n yolo26s}
CONFIGS=${CONFIGS:-n1 n2s1 n2s1d n3s1 n3s2 n3s3 n5s2}
GPUS=${GPUS:-1}
IMGSZ=${IMGSZ:-640}
EPOCHS=${EPOCHS:-100}
BATCH=${BATCH:-16}
SEED=${SEED:-0}
WORKERS=${WORKERS:-8}
PROJECT=${PROJECT:-runs/tempo}
OUT=${OUT:-results/matrix.csv}
EXTRA=${EXTRA:-}

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs "$(dirname "$OUT")"
if [[ -n "${VENV:-}" ]]; then
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
fi

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    read -r -a ALL_CONFIGS <<< "$CONFIGS"
    if (( SLURM_ARRAY_TASK_ID >= ${#ALL_CONFIGS[@]} )); then
        echo "Array task $SLURM_ARRAY_TASK_ID has no config (CONFIGS has ${#ALL_CONFIGS[@]} entries)"
        exit 0
    fi
    CONFIGS=${ALL_CONFIGS[$SLURM_ARRAY_TASK_ID]}
fi

DEVICE=$(seq -s, 0 $((GPUS - 1)))
echo "host=$(hostname) device=$DEVICE datasets=[$DATASETS] models=[$MODELS] configs=[$CONFIGS]"

# shellcheck disable=SC2086
python scripts/tempo/run_matrix.py \
    --data $DATASETS \
    --models $MODELS \
    --configs $CONFIGS \
    --imgsz "$IMGSZ" --epochs "$EPOCHS" --batch "$BATCH" --seed "$SEED" \
    --device "$DEVICE" --workers "$WORKERS" \
    --project "$PROJECT" --out "$OUT" $EXTRA
