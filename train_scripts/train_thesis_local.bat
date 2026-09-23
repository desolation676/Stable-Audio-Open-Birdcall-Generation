@echo off
cd /d "%~dp0.."
 
py -3.10 src/thesis/train_birds.py ^
  --config-file stable-audio-tools/defaults.ini ^
  --model-config checkpoints/model_config_CA_general.json ^
  --parquet-path data_thesis/manifests/all.parquet ^
  --class-mapping-path class_mappings/class_mapping_runpod.json ^
  --sampler-alpha 0.5 ^
  --name test_01 ^
  --save-dir checkpoints ^
  --batch-size 1 ^
  --num-workers 1 ^
  --checkpoint-every 1000 ^
  --precision 16-mixed