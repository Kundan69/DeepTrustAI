# Phase 1: Image Deepfake Detection

This phase is considered complete only when the image model performs well on images that were never part of the original dataset source.

## Dataset Structure

Training expects:

```text
datasets/image/
  train/
    fake/
    real/
  val/
    fake/
    real/
  test/
    fake/
    real/
```

Create an external real-world test set:

```text
datasets/image_external/
  fake/
  real/
```

Use images from outside the training dataset: screenshots, WhatsApp-compressed images, social-media images, web images, AI-generated faces, and camera photos.

## Train

Run training from `backend/training/image`:

```bash
python train.py
```

The script saves:

```text
backend/models/image/best_model_v2.pth
backend/models/image/best_model_v2.metadata.json
```

The metadata file stores the actual class mapping used during training.

## Evaluate External Images

Run:

```bash
python evaluate_external.py ../../../datasets/image_external
```

Do not trust only internal test accuracy. Check:

- fake precision
- fake recall
- real precision
- real recall
- confusion matrix

## Completion Target

Before moving to video detection, target at least:

- 90%+ internal test accuracy
- 80%+ external/wild test accuracy
- balanced fake and real recall
- no obvious label inversion

If external accuracy is poor, improve the dataset split and retrain with more outside-source images.
