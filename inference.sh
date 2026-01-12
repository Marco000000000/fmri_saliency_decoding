#!/bin/bash

# --- PARAMETRI ---
SUBJECT=3
# Scale: 0.5 è un buon punto di partenza per fMRI
SCALE_D=0.5
SCALE_S=0.5
SEGMENTS=30

echo "Running Multi-ControlNet FULL Analysis..."
echo "Depth Scale: $SCALE_D | Seg Scale: $SCALE_S"

python scripts/inference_final_comparison.py \
    --subject $SUBJECT \
    --scale_depth $SCALE_D \
    --scale_seg $SCALE_S \
    --segments $SEGMENTS \
    --output_dir "results_multicontrolnet_full"