# RoadHeightFormer Configuration Guide

This guide explains how to use the configuration system for training RoadHeightFormer models.

## Configuration File Structure

The main configuration file is `config.yaml`. It contains all the parameters that were previously passed as command-line arguments to `train.py`.

### Key Configuration Sections:

1. **Dataset Configuration**
   - `name`: Dataset to use (RSRD, CARDSet, CARDSetSmall, CARDSetV2Small)
   - `stereo`: Whether to use stereo mode
   - `preprocessed`: Use preprocessed data if available

2. **Model Configuration**
   - `backbone`: Feature extractor (efficientnet, dino)
   - `regression`: True for regression, False for classification
   - `normalize`: Normalize heights to [-1, 1]
   - `pred_head_dim`: Bottleneck dimension

3. **Training Configuration**
   - `batch_size`: Training batch size
   - `epochs`: Number of training epochs
   - `seed`: Random seed for reproducibility

4. **Loss Configuration**
   - `type`: Loss function (L1, MSE, composite, etc.)
   - `composite_weights`: Weights for composite loss components

5. **Optimizer & Scheduler**
   - Learning rates, weight decay, scheduler type

6. **Logging & WandB**
   - Logging directory, WandB project settings

## Usage

### Basic Training
```bash
# Use default config
python train.py

# Use specific config file
python train.py --config my_config.yaml

# Use the experiment script
./run_experiments.sh my_config.yaml
```

### Overriding Config Values
You can override any config value using command-line arguments:
```bash
python train.py --config config.yaml --epochs 100 --batch_size 16 --name_run "custom_run"
```

### Creating Custom Configs
1. Copy `config.yaml` to a new file
2. Modify the parameters as needed
3. Run with the new config file

## Example Configurations

### For CARDSet Dataset with Composite Loss
```yaml
dataset:
  name: "CARDSet"
  stereo: false
  preprocessed: true

loss:
  type: "composite"
  composite_weights:
    pixel: 0.3
    gradient: 1.0
    structure: 0.5
    normal: 0.1
    smoothness: 0.1
```

### For RSRD Dataset with L1 Loss
```yaml
dataset:
  name: "RSRD"
  stereo: true

loss:
  type: "L1"
```

## Integration with train.py

To use the config system in `train.py`, you need to:

1. Import the config utilities:
```python
from utils.config import parse_args_with_config
```

2. Replace the argument parsing:
```python
# Instead of: args = parser.parse_args()
args = parse_args_with_config()
```

The config system maintains backward compatibility - all existing command-line arguments still work.

## Config File Validation

The config file uses YAML format. Make sure:
- Indentation is consistent (use spaces, not tabs)
- String values are quoted if they contain special characters
- Boolean values are `true`/`false` (lowercase)
- Numeric values don't need quotes

## Best Practices

1. **Version Control**: Keep your config files in version control
2. **Naming**: Use descriptive names for config files (e.g., `experiment_1.yaml`, `baseline_config.yaml`)
3. **Comments**: Add comments in your config files to explain non-obvious parameters
4. **Backup**: Save working configs for reproducibility
5. **Modular**: Create base configs and extend them for different experiments