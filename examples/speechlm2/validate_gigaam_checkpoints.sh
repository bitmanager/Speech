#!/usr/bin/env bash
# Run the upstream validator on a dedicated GPU after each completed checkpoint.
set -euo pipefail

run_dir=${1:?Usage: validate_gigaam_checkpoints.sh RUN_DIR [MAX_STEPS]}
max_steps=${2:-3000}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}

for step in $(seq 200 200 "$max_steps"); do
    checkpoint="$run_dir/checkpoints/step-$step-last.ckpt"
    unfinished="$run_dir/checkpoints/step-$step-last-unfinished"
    echo "Waiting for completed checkpoint: $checkpoint"
    timeout 3600 bash -c 'until [[ -s "$1" && ! -e "$2" ]]; do sleep 10; done' \
        _ "$checkpoint" "$unfinished"
    export ASR_OUTPUT_DIR="$run_dir/independent-validation/step-$step"
    mkdir -p "$ASR_OUTPUT_DIR"
    torchrun --standalone --nproc_per_node=1 examples/speechlm2/stt_duplex_infer.py \
        --config-name gigaam_qwen_asr \
        "+model.init_from_checkpoint=$checkpoint" \
        trainer.devices=1 trainer.limit_val_batches=1.0 \
        data.validation_ds.batch_size=8 '~data.train_ds' \
        exp_manager.create_checkpoint_callback=false \
        > "$ASR_OUTPUT_DIR/console.log" 2>&1
    echo "Validation complete: $ASR_OUTPUT_DIR"
done
