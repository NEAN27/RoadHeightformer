# Debug Setup: Identical Train/Test Loaders

## Changes Made

### 1. **Dataset Fix** ([utils/dataset.py](utils/dataset.py#L24-L30))
Both train and test loaders now load **identical data**:
```python
if self.training:
    self.load_dataset_names('./filenames/train/')
    self.preprocessed_path = os.path.join(preprocessed_path, 'train')
else:
    # IDENTICAL TO TRAIN FOR DEBUG: loads same training data
    self.load_dataset_names('./filenames/train/')
    self.preprocessed_path = os.path.join(preprocessed_path, 'train')
```

### 2. **DataLoader Configuration** ([train.py](train.py#L957-L961))
Both loaders now use **identical configurations**:
```python
# IDENTICAL LOADERS FOR DEBUG: both use same data, batch size, and workers
train_loader = DataLoader(train_set, args.batch_size, shuffle=False, num_workers=8, drop_last=True, pin_memory=True)
test_loader = DataLoader(test_set, args.batch_size, shuffle=False, num_workers=8, drop_last=True, pin_memory=True)
```

**Before:**
- train_loader: batch_size=args.batch_size, num_workers=8, drop_last=True
- test_loader: batch_size=1, num_workers=4, drop_last=False ❌

**After:**
- train_loader: batch_size=args.batch_size, num_workers=8, drop_last=True ✅
- test_loader: batch_size=args.batch_size, num_workers=8, drop_last=True ✅

---

## Loss vs Abs_Err Relationship

### Expected Behavior
If using **L1Loss**, they should be **identical**:

| Component | Formula | Location |
|-----------|---------|----------|
| **Loss (L1)** | `mean(\|ele_pred[mask] - ele_gt[mask]\|)` | [models/loss.py](models/loss.py#L45) - `LossReg` |
| **Metric abs_err** | `mean(\|ele_gt[mask] - ele_pred[mask]\|)` | [utils/metric.py](utils/metric.py#L88) - `compute_values_rhf` |

**Note:** They are mathematically the same! The order doesn't matter due to absolute value.

### Different Behavior If Using MSE
If using **MSE loss**, they will **NOT match**:
- Loss (MSE): `mean((ele_pred[mask] - ele_gt[mask])^2)`
- Metric abs_err: `mean(\|ele_pred[mask] - ele_gt[mask]\|)`

---

## Debug Script

Created: `debug_loss_vs_metric.py`

### Purpose
Compares:
1. ✅ Are train and test loaders loading identical data?
2. ✅ Do predictions match between loaders on same data?
3. ✅ Does loss output match abs_err from metrics?

### Usage

```bash
# Basic usage (RSRD dataset, default model)
python debug_loss_vs_metric.py --loadckpt ./path/to/checkpoint.ckpt

# With specific dataset
python debug_loss_vs_metric.py --dataset CARDSet --loadckpt ./checkpoint.ckpt

# With custom settings
python debug_loss_vs_metric.py \
  --dataset RSRD \
  --loadckpt ./checkpoint.ckpt \
  --batch_size 4 \
  --loss L1 \
  --backbone efficientnet \
  --regression

# For CARDSetV2Small
python debug_loss_vs_metric.py \
  --dataset CARDSetV2Small \
  --loadckpt ./checkpoint.ckpt \
  --batch_size 4
```

### Available Arguments
```
--loadckpt          Path to checkpoint file
--dataset           Dataset: RSRD, CARDSet, CARDSetSmall, CARDSetV2Small
--batch_size        Batch size (default: 4)
--stereo            Use stereo mode
--down_scale        Down scale factor (default: 4)
--regression        Use regression loss
--backbone          Model backbone: efficientnet, dino
--normalize         Normalize heights to [-1, 1]
--pred_head_dim     Prediction head dimension (default: 128)
--preprocessed      Use preprocessed data
--ele_range         Elevation range in meters (default: 0.2)
--loss              Loss type: L1, MSE (default: L1)
```

### Expected Output

For each batch, you'll see:

```
================================================================================
COMPARING LOSS vs ABS_ERR (Metric)
================================================================================

================================================================================
Batch 1/5
================================================================================

>>> LOADER COMPARISON:
Train batch shape: torch.Size([4, 3, 512, 512])
Test batch shape:  torch.Size([4, 3, 512, 512])
Train images equal to test images: True ✅
Train GT equal to test GT: True ✅
Train mask equal to test mask: True ✅

--- Sample 1 in Batch ---
Loss from loss_func: 0.123456
Loss from test:      0.123456
Losses equal: True ✅

Abs_err from metric: 0.123456
Manual abs_err calc: 0.123456

>>> COMPARISON: Loss (0.123456) vs Abs_Err (0.123456)
Difference: 0.000000
Match (diff < 1e-5): True ✅
```

---

## What to Verify

When you run the debug script, check:

1. **Loader Identity** ✅
   - "Train images equal to test images: True"
   - "Train GT equal to test GT: True"
   - "Train mask equal to test mask: True"

2. **Prediction Consistency** ✅
   - "Losses equal: True" (same predictions for same input)

3. **Loss-Metric Agreement** ✅
   - "Match (diff < 1e-5): True" (loss matches abs_err)

If any check shows **False** or large differences:
- Different training than test data configuration
- Model predictions are non-deterministic
- Loss function differs from metric calculation

---

## Reverting Changes

To revert to original separate train/test splits:

### [utils/dataset.py](utils/dataset.py#L24-L30)
```python
else:
    self.load_dataset_names('./filenames/test/')  # Changed from 'train/'
    self.preprocessed_path = os.path.join(preprocessed_path, 'test')  # Changed from 'train'
```

### [train.py](train.py#L957-L961)
```python
train_loader = DataLoader(train_set, args.batch_size, shuffle=False, num_workers=8, drop_last=True, pin_memory=True)
test_loader = DataLoader(test_set, 1, shuffle=False, num_workers=4, drop_last=False, pin_memory=True)  # Restored original
```
