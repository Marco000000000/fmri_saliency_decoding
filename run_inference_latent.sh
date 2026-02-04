#!/bin/bash

# Interrompe lo script se un comando fallisce
set -e

# --- CONFIGURAZIONE ---
SUBJECT=3
ROIS="VC"
DATASET_ROOT="data/GOD_Dataset"

# Parametri Training (Latent Projectors)
EPOCHS=100           # 100 epoche sono solitamente sufficienti per convergere su 64x64
BATCH_SIZE=32
LR=0.0001
TRAIN_OUT_DIR="trained_latent_models"

# Parametri Inference
# IMPORTANTE: Indica qui dove si trova il tuo modello 'best_aligner.pth' (AlexNet)
# Se hai usato lo script 'hybrid', sarà in trained_fmri_decoders/multimodal/S3/VC
ALIGNER_DIR="trained_fmri_decoders/multimodal" 
INF_OUT_DIR="results_latent_full_pipeline_S${SUBJECT}"

echo "============================================================"
echo "AVVIO PIPELINE COMPLETA: LATENT INJECTION (S${SUBJECT})"
echo "============================================================"

# ---------------------------------------------------------
# FASE 1: TRAINING DEPTH PROJECTOR
# ---------------------------------------------------------
# echo ""
# echo "[1/3] Training Linear Projector for DEPTH..."
# python scripts/train_latent_injection.py \
#     --subject $SUBJECT \
#     --task "depth" \
#     --epochs $EPOCHS \
#     --lr $LR \
#     --dataset_root "$DATASET_ROOT" \
#     --output_dir "$TRAIN_OUT_DIR"

# echo "-> Depth Projector salvato in $TRAIN_OUT_DIR/best_latent_depth.pth"

# # ---------------------------------------------------------
# # FASE 2: TRAINING SEG PROJECTOR
# # ---------------------------------------------------------
# echo ""
# echo "[2/3] Training Linear Projector for SEGMENTATION..."
# python scripts/train_latent_injection.py \
#     --subject $SUBJECT \
#     --task "seg" \
#     --epochs $EPOCHS \
#     --lr $LR \
#     --dataset_root "$DATASET_ROOT" \
#     --output_dir "$TRAIN_OUT_DIR"

# echo "-> Seg Projector salvato in $TRAIN_OUT_DIR/best_latent_seg.pth"

# ---------------------------------------------------------
# FASE 3: FULL INFERENCE (Retrieval + Injection + ControlNet)
# ---------------------------------------------------------
echo ""
echo "[3/3] Running Inference..."

# Verifica se l'aligner esiste, altrimenti avvisa
ALIGNER_CHECK="$ALIGNER_DIR/S$SUBJECT/$ROIS/best_aligner.pth"
if [ ! -f "$ALIGNER_CHECK" ]; then
    echo "ATTENZIONE: Aligner non trovato in $ALIGNER_CHECK"
    echo "L'inferenza userà una classificazione casuale o cercherà un fallback."
fi

python scripts/inference_latent_controlnet.py \
    --subject $SUBJECT \
    --rois "$ROIS" \
    --dataset_root "$DATASET_ROOT" \
    --models_dir "$TRAIN_OUT_DIR" \
    --aligner_dir "$ALIGNER_DIR" \
    --output_dir "$INF_OUT_DIR" \
    --scale_depth 0.7 \
    --scale_seg 0.5

echo ""
echo "============================================================"
echo "PIPELINE COMPLETATA CON SUCCESSO!"
echo "Risultati salvati in: $INF_OUT_DIR"
echo "============================================================"