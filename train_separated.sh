#!/bin/bash

SUBJECT=3
ROIS="VC"
EPOCHS=150

echo "=== STEP 1: TRAINING DEPTH MODEL ==="
python scripts/train_separated.py \
    --subjects $SUBJECT --rois "$ROIS" \
    --task "depth" \
    --epochs $EPOCHS \
    --lambda_pixel 10.0 --lambda_grad 5.0

echo "=== STEP 2: TRAINING SEGMENTATION MODEL ==="
python scripts/train_separated.py \
    --subjects $SUBJECT --rois "$ROIS" \
    --task "seg" \
    --epochs $EPOCHS \
    --lambda_pixel 10.0 --lambda_tv 0.5

echo "=== STEP 3: TRAINING ALIGNER (AlexNet) ==="
python scripts/train_separated.py \
    --subjects $SUBJECT --rois "$ROIS" \
    --task "aligner" \
    --epochs $EPOCHS

echo "TUTTI I TRAINING COMPLETATI."