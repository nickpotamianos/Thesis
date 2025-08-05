#!/usr/bin/env bash
set -e
EXP="cir_3_random3_0"
python sota/data_prep/uwb_extract_dataset.py   $EXP --exp_root ./data/three_robots
python sota/uwb/train_nlos_classifier.py  $EXP
python sota/uwb/train_bias_regressor.py   $EXP

python sota/data_prep/imu_extract_dataset.py   $EXP
python sota/imu/repiln/train_repiln.py         --dataset $EXP

echo "✨  All front‑ends trained.  Ready for factor‑graph."
