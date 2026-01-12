#!/bin/bash

# --- CONFIGURAZIONE ---
SUBJECT=3
ROIS="VC"
# SCALE: Quanto ControlNet deve obbedire alla mappa fMRI.
# 1.0 = Obbedienza totale (Bordi storti rimangono storti)
# 0.5 = Suggerimento vago (Stable Diffusion raddrizza i bordi)
# 0.0 = Ignora la mappa (Solo prompt)
SCALE=1

echo "Running K-Means Planes Analysis con Scale=${SCALE}..."

python scripts/inference_segmentation_combined.py \
    --subject $SUBJECT \
    --rois "$ROIS" \
    --scale $SCALE \
    --output_dir "results_planes_weak_S${SCALE}"