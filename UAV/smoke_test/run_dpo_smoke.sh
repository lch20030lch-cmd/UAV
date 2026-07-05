#!/usr/bin/env bash
set -euo pipefail

# Run from the UAV project root:
#   bash smoke_test/run_dpo_smoke.sh
#
# Optional overrides:
#   SFT_CKPT=/path/to/stage1_sft_final bash smoke_test/run_dpo_smoke.sh
#   SOURCE_DATA_DIR=/path/to/full5000 bash smoke_test/run_dpo_smoke.sh
#   SMOKE_PAIRS=320 bash smoke_test/run_dpo_smoke.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

CONFIG="${CONFIG:-smoke_test/dpo_smoke_5090.yaml}"
SFT_CKPT="${SFT_CKPT:-/root/autodl-tmp/outputs/stage1_sft_final}"
SOURCE_DATA_DIR="${SOURCE_DATA_DIR:-/root/autodl-tmp/data/full5000}"
SMOKE_DATA_DIR="${SMOKE_DATA_DIR:-/root/autodl-tmp/data/dpo_smoke}"
SMOKE_PAIRS="${SMOKE_PAIRS:-160}"
LOG_FILE="${LOG_FILE:-dpo_smoke.log}"

echo "== DPO smoke test =="
echo "Project root : ${PROJECT_ROOT}"
echo "Config       : ${CONFIG}"
echo "SFT ckpt     : ${SFT_CKPT}"
echo "Source data  : ${SOURCE_DATA_DIR}"
echo "Smoke data   : ${SMOKE_DATA_DIR}"
echo "Smoke pairs  : ${SMOKE_PAIRS}"
echo "Log file     : ${LOG_FILE}"

if [[ ! -d "${SFT_CKPT}" ]]; then
  echo "ERROR: Stage I checkpoint not found: ${SFT_CKPT}" >&2
  exit 1
fi

if [[ ! -f "${SOURCE_DATA_DIR}/dpo_dataset.jsonl" ]]; then
  echo "ERROR: DPO dataset not found: ${SOURCE_DATA_DIR}/dpo_dataset.jsonl" >&2
  exit 1
fi

if [[ ! -f "${SOURCE_DATA_DIR}/sft_dataset.jsonl" ]]; then
  echo "ERROR: SFT dataset not found: ${SOURCE_DATA_DIR}/sft_dataset.jsonl" >&2
  exit 1
fi

mkdir -p "${SMOKE_DATA_DIR}"
head -n "${SMOKE_PAIRS}" "${SOURCE_DATA_DIR}/dpo_dataset.jsonl" > "${SMOKE_DATA_DIR}/dpo_dataset.jsonl"
cp "${SOURCE_DATA_DIR}/sft_dataset.jsonl" "${SMOKE_DATA_DIR}/sft_dataset.jsonl"

mkdir -p /root/autodl-tmp/outputs/dpo_smoke /root/autodl-tmp/checkpoints/dpo_smoke /root/autodl-tmp/logs/dpo_smoke

echo
echo "Prepared smoke dataset:"
wc -l "${SMOKE_DATA_DIR}/dpo_dataset.jsonl"
ls -lh "${SMOKE_DATA_DIR}/dpo_dataset.jsonl"

echo
echo "Starting DPO smoke test in background..."
nohup python src/training/train_dpo.py \
  --config "${CONFIG}" \
  --stage1_ckpt "${SFT_CKPT}" \
  --data_dir "${SMOKE_DATA_DIR}" \
  > "${LOG_FILE}" 2>&1 &

PID="$!"
echo "PID: ${PID}"
echo
echo "Monitor:"
echo "  tail -f ${LOG_FILE}"
echo "  nvidia-smi"
echo
echo "Expected final output:"
echo "  /root/autodl-tmp/outputs/dpo_smoke/stage2_dpo_final"
