# RoadHeightFormer Config System - Quick Start

## What I've Created

I've set up a complete configuration system for your RoadHeightFormer experiments:

### Files Created:
1. **`config.yaml`** - Main configuration file with all training parameters
2. **`utils/config.py`** - Configuration loading utilities
3. **`config_l1_baseline.yaml`** - Example config for L1 loss baseline
4. **`CONFIG_README.md`** - Detailed documentation
5. **Updated `run_experiments.sh`** - Now supports config files

### Dependencies Installed:
- **PyYAML** - For parsing YAML configuration files

## How to Use

### 1. Basic Training (uses config.yaml)
```bash
python train.py
```

### 2. Use Specific Config
```bash
python train.py --config config_l1_baseline.yaml
```

### 3. Override Config Values
```bash
python train.py --config config.yaml --epochs 100 --batch_size 4
```

### 4. Use the Experiment Script
```bash
./run_experiments.sh config.yaml
./run_experiments.sh config_l1_baseline.yaml --name_run "my_experiment"
```

## Key Benefits

1. **Organized**: All parameters in one place
2. **Versionable**: Config files can be tracked in git
3. **Reusable**: Easy to create experiment variants
4. **Documented**: Self-documenting with comments
5. **Compatible**: Backward compatible with existing command-line args

## Next Steps

1. **Modify train.py** (optional): To use the config system, replace:
   ```python
   args = parser.parse_args()
   ```
   with:
   ```python
   from utils.config import parse_args_with_config
   args = parse_args_with_config()
   ```

2. **Customize configs**: Copy `config.yaml` and modify for your experiments

3. **Experiment tracking**: Use different config files for different ablation studies

## Example Workflow

```bash
# Create experiment configs
cp config.yaml experiment_1.yaml
cp config.yaml experiment_2.yaml

# Edit experiment_1.yaml to use different loss
# Edit experiment_2.yaml to use different backbone

# Run experiments
./run_experiments.sh experiment_1.yaml
./run_experiments.sh experiment_2.yaml
```

The config system is now ready to use! Check `CONFIG_README.md` for detailed documentation.