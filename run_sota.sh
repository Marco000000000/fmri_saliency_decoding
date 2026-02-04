#!/bin/bash

# Interrompe lo script immediatamente se un comando fallisce
set -e

# --- CONFIGURAZIONE ---
SUBJECT=3
ROIS="VC"
DATASET_ROOT="data/GOD_Dataset"

# Parametri Training (SOTA Alignment)
# 50 epoche sono spesso sufficienti per l'allineamento delle feature
EPOCHS=50           
BATCH_SIZE=2
LR=0.0003
TRAIN_OUT_DIR="trained_sota_models"

# Parametri Inference
# IMPORTANTE: Qui deve esserci il tuo 'best_aligner.pth' (AlexNet) per il retrieval della classe
ALIGNER_DIR="trained_fmri_decoders/multimodal" 
INF_OUT_DIR="results_sota_full_pipeline_S${SUBJECT}"

echo "============================================================"
echo "AVVIO PIPELINE COMPLETA: SOTA ALIGNMENT (S${SUBJECT})"
echo "Architettura: fMRI -> Linear -> Inverse Adapter -> Frozen SOTA"
echo "============================================================"

# ---------------------------------------------------------
# FASE 1: TRAINING DEPTH (DPT WRAPPER)
# ---------------------------------------------------------
echo ""
echo "[1/3] Training Adapter for DPT (Depth)..."
python scripts/train_sota_alignment.py \
    --subject $SUBJECT \
    --task "depth" \
    --epochs $EPOCHS \
    --lr $LR \
    --dataset_root "$DATASET_ROOT" \
    --output_dir "$TRAIN_OUT_DIR"

echo "-> Depth Model salvato in $TRAIN_OUT_DIR"

# ---------------------------------------------------------
# FASE 2: TRAINING SEGMENTATION (SEGFORMER WRAPPER)
# ---------------------------------------------------------
echo ""
echo "[2/3] Training Adapter for SEGFORMER (Segmentation)..."
python scripts/train_sota_alignment.py \
    --subject $SUBJECT \
    --task "seg" \
    --epochs $EPOCHS \
    --lr $LR \
    --dataset_root "$DATASET_ROOT" \
    --output_dir "$TRAIN_OUT_DIR"

echo "-> Segmentation Model salvato in $TRAIN_OUT_DIR"

# ---------------------------------------------------------
# FASE 3: FULL INFERENCE (Retrieval + Reconstruction + ControlNet)
# ---------------------------------------------------------
echo ""
echo "[3/3] Running Inference..."

# Controllo esistenza Aligner (opzionale ma consigliato per risultati sensati)
ALIGNER_CHECK="$ALIGNER_DIR/S$SUBJECT/$ROIS/best_aligner.pth"
if [ ! -f "$ALIGNER_CHECK" ]; then
    echo "ATTENZIONE: Aligner semantico non trovato in $ALIGNER_CHECK"
    echo "L'inferenza userà una classificazione casuale per il prompt."
fi

python scripts/inference_sota_controlnet.py \
    --subject $SUBJECT \
    --rois "$ROIS" \
    --dataset_root "$DATASET_ROOT" \
    --models_dir "$TRAIN_OUT_DIR" \
    --aligner_dir "$ALIGNER_DIR" \
    --output_dir "$INF_OUT_DIR" \
    --scale_depth 0.7 \
    --scale_seg 0.6

echo ""
echo "============================================================"
echo "PIPELINE COMPLETATA CON SUCCESSO!"
echo "Risultati salvati in: $INF_OUT_DIR"
echo "============================================================"