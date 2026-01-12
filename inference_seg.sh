#!/bin/bash

# Prova con pochi segmenti per forzare forme pulite
SEGMENTS=30
SCALE=0.5

python scripts/inference_seg_quantization.py \
    --subject 3 \
    --rois "VC" \
    --scale $SCALE \
    --segments $SEGMENTS \
    --output_dir "results_seg_slic_S${SCALE}_N${SEGMENTS}"