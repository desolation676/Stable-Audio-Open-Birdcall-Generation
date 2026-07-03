@echo off
echo Starting Stable Audio Open Fine-Tune...

set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python stable-audio-tools/train.py ^
  --config-file stable-audio-tools/defaults.ini ^
  --dataset-config dataset.json ^
  --model-config checkpoints/model_config_CA_general.json ^
  --name xeno_canto_finetune ^
  --save-dir checkpoints ^
  --pretrained-ckpt-path checkpoints/model.safetensors ^
  --batch-size 1 ^
  --num-workers 0 ^
  --checkpoint-every 1000 ^
  --precision 16-mixed

pause