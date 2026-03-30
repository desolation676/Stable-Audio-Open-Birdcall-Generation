#!/bin/bash
echo "Starting Stable Audio Open Fine-Tune on A40 Linux Pod..."

pip install wandb prefigure
pip install -e ./stable-audio-tools

python stable-audio-tools/train.py \
  --config-file stable-audio-tools/defaults.ini \
  --dataset-config dataset.json \
  --model-config model_config_2304.json \
  --name xeno_canto_finetune_2304_v1 \
  --save-dir checkpoints \
  --pretrained-ckpt-path  checkpoints/model_2304.safetensors \
  --batch-size 16 \
  --num-workers 12 \
  --checkpoint-every 1000 \
  --precision 16-mixed