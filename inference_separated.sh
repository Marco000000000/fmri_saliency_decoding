#!/bin/bash
SUBJECT=3
python scripts/inference_separated.py \
    --subject $SUBJECT \
    --output_dir "results_comparison_S${SUBJECT}" \
    --scale_depth 0.6 --scale_seg 0.6