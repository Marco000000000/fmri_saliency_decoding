#!/bin/bash

# ==========================================
# CONFIGURAZIONE PERCORSI BASE
# ==========================================
BASE_DIR="/home/mfinocchiaro/Kamitani_fMRI/Dataset_Kamitani/preprocessed/fmri_saliency_decoding"

echo "=========================================================="
echo "🚀 AVVIO TEST DI CONTROLLO: LATE FUSION (SOLO MODELLO OLISTICO)"
echo "=========================================================="

# ==========================================
# 1. VALUTAZIONE GOD (Soggetti 1-5)
# ==========================================
DATASET="god"
ROI_GOD="VC"

for s in {1..5}; do
    echo ""
    echo "----------------------------------------------------------"
    echo "🧠 Valutazione $DATASET (Soggetto $s - ROI: $ROI_GOD)"
    echo "----------------------------------------------------------"
    
    DIR_FULL="${BASE_DIR}/results_final/sdxl_inpaint/S${s}_${ROI_GOD}_box/single"

    if [ -d "$DIR_FULL" ]; then
        echo "✅ Trovata cartella FULL per S${s}. Avvio Test Olistico..."
        python scripts/eval_tris.py \
            --dataset "$DATASET" \
            --gen_dir "$DIR_FULL"
    else
        echo "⚠️ ATTENZIONE: Manca cartella FULL per GOD S${s}."
        echo "   Cercavo: $DIR_FULL"
    fi
done

# ==========================================
# 2. VALUTAZIONE SHEN (Soggetti 1-3)
# ==========================================
DATASET="shen"

for s in {1..3}; do
    echo ""
    echo "----------------------------------------------------------"
    echo "🧠 Valutazione $DATASET (Soggetto $s)"
    echo "----------------------------------------------------------"
    
    # Percorso esatto per Shen estratto dal tuo file
    DIR_FULL="${BASE_DIR}/reconstructions/S${s}_natural_inpaint_mask/single"

    if [ -d "$DIR_FULL" ]; then
        echo "✅ Trovata cartella FULL per S${s}. Avvio Test Olistico..."
        python scripts/eval_tris.py \
            --dataset "$DATASET" \
            --gen_dir "$DIR_FULL"
    else
        echo "⚠️ ATTENZIONE: Manca cartella FULL per SHEN S${s}."
        echo "   Cercavo: $DIR_FULL"
    fi
done

echo ""
echo "🎉 TEST OLISTICO DI CONTROLLO COMPLETATO!"