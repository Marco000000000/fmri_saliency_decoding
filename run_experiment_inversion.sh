#!/bin/bash

# ==============================================================================
# Script Esperimento: Blind Reconstruction (fMRI -> Optimized Latent -> SD)
# Nessuna immagine ground truth viene usata in input, solo fMRI.
# ==============================================================================

SUBJECT=3
ROIS="VC"
DATASET_ROOT="data/GOD_Dataset"

# Training Aligner (fMRI -> CLIP)
# Serve solo se non l'hai già fatto nel passo precedente
TRAIN_EPOCHS=100
BATCH_SIZE=64
SAVE_DIR="trained_fmri_decoders/clip_aligner"

# Inference Optimization
OUTPUT_DIR="results_reconstruction_blind_or_S${SUBJECT}"
STRENGTH=0.65       # Bilanciamento: 0.65 corregge gli artefatti ma tiene la forma
TV_WEIGHT=0.05      # Regola quanto è "pulita/sfocata" l'immagine ottimizzata
OPT_STEPS=250       # Step di gradient descent per creare l'immagine

echo "--- STARTING BLIND RECONSTRUCTION EXPERIMENT ---"

# 1. TRAINING (Se necessario)
ALIGNER_PATH="${SAVE_DIR}/S${SUBJECT}_${ROIS}/best_clip_aligner.pth"
if [ ! -f "$ALIGNER_PATH" ]; then
    echo "[1/2] Training fMRI -> CLIP Aligner..."
    python scripts/train_clip_aligner.py \
        --subject $SUBJECT \
        --rois "$ROIS" \
        --batch_size $BATCH_SIZE \
        --epochs $TRAIN_EPOCHS \
        --dataset_root "$DATASET_ROOT" \
        --save_dir "$SAVE_DIR"
else
    echo "[1/2] Aligner trovato. Skipping training."
fi

# 2. INFERENCE
echo ""
echo "[2/2] Inference: Optimizing Inputs from fMRI embeddings..."
python scripts/inference_reconstruction.py \
    --subject $SUBJECT \
    --rois "$ROIS" \
    --dataset_root "$DATASET_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --strength $STRENGTH \
    --tv_weight $TV_WEIGHT \
    --opt_steps $OPT_STEPS

echo "Done. Results in $OUTPUT_DIR"