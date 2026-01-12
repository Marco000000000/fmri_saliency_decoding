#!/bin/bash

# Genera analisi completa V2 (7 colonne)
python scripts/inference_multimodal.py \
    --subject 3 \
    --rois "VC" \
    --num_samples 50 \
    --output_dir "predictions/complete_analysis_v2_7cols"