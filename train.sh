#!/bin/bash

# ==============================================================================
# Script di Training Multimodale (fMRI -> Depth + Segmentation + AlexNet Features)
# ==============================================================================

# Configurazione Training
SUBJECTS="3"              # ID soggetto (es. "3" o "1,2,3,4,5")
ROIS="VC"                 # Regione di interesse (Visual Cortex)
BATCH_SIZE=64
EPOCHS=200                # Numero massimo epoche
PATIENCE=3               # Early stopping patience
LR=0.0001                 # Learning rate

# Pesi delle Loss (Bilanciati per ridurre sfocatura)
LAMBDA_PIXEL=1.0         # Peso L1 Loss (Ricostruzione pixel-wise)
LAMBDA_GRAD=1.0           # Peso Gradient Loss (Nitidezza bordi Depth)
LAMBDA_TV=0.5             # Peso Total Variation (Uniformità colori Seg)
LAMBDA_ALEX=0.5           # Peso Allineamento Semantico (AlexNet)

# Percorsi
DATASET_ROOT="data/GOD_Dataset"
CHECKPOINTS_DIR="trained_fmri_decoders"

echo "Avvio Training Multimodale per Soggetto(i): $SUBJECTS..."
echo "Configurazione Loss: Pixel=$LAMBDA_PIXEL, Grad=$LAMBDA_GRAD, TV=$LAMBDA_TV, Alex=$LAMBDA_ALEX"

python scripts/train_multimodal.py \
    --subjects "$SUBJECTS" \
    --rois "$ROIS" \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --patience $PATIENCE \
    --lr $LR \
    --lambda_pixel $LAMBDA_PIXEL \
    --lambda_grad $LAMBDA_GRAD \
    --lambda_tv $LAMBDA_TV \
    --lambda_alex $LAMBDA_ALEX \
    --dataset_root "$DATASET_ROOT" \
    --checkpoints_dir "$CHECKPOINTS_DIR"

echo "Training completato."