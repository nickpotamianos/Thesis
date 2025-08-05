# SOTA Pipeline - MILUV Thesis

State-of-the-Art pipeline implementation for multi-robot localization with UWB, IMU, and vision.

## Quick Start

1. Create environment:
```bash
conda env create -f env/environment_sota.yml
conda activate miluv-sota
```

2. Train all models:
```bash
bash scripts/run_training_all.sh
```

3. Run online demo:
```bash
bash scripts/run_online_demo.sh
```

## Structure

- `data_prep/` - Dataset extraction and preprocessing
- `uwb/` - UWB machine learning models and training
- `imu/` - Deep IMU odometry (RepILN)
- `factor_graph/` - Factor graph core with custom factors
- `scripts/` - Training and demo scripts
