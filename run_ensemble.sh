#!/bin/bash

# ==========================================
# CONFIGURAZIONE PERCORSI BASE
# ==========================================
BASE_DIR="/home/mfinocchiaro/Kamitani_fMRI/Dataset_Kamitani/preprocessed/fmri_saliency_decoding"

echo "=========================================================="
echo "🚀 AVVIO VALUTAZIONE ENSEMBLE (LATE FUSION) SOLO SU VC"
echo "=========================================================="

# ==========================================
# 1. VALUTAZIONE GOD (Soggetti 1-5, ROI: VC)
# # ==========================================
# DATASET="god"
# ROI_GOD="VC"

# for s in {1..5}; do
#     echo ""
#     echo "----------------------------------------------------------"
#     echo "🧠 Valutazione $DATASET (Soggetto $s - ROI: $ROI_GOD)"
#     echo "----------------------------------------------------------"
    
#     # Percorsi esatti estratti dai tuoi esempi
#     DIR_FULL="${BASE_DIR}/results_final/sdxl_inpaint/S${s}_${ROI_GOD}_box/single"
#     DIR_FG="${BASE_DIR}/results_sdxl_split_all/S${s}_${ROI_GOD}/fg_only"
#     DIR_BG="${BASE_DIR}/results_sdxl_split_all/S${s}_${ROI_GOD}/bg_only"

#     if [ -d "$DIR_FULL" ] && [ -d "$DIR_FG" ] && [ -d "$DIR_BG" ]; then
#         echo "✅ Trovate le 3 cartelle per S${s}. Avvio Late Fusion..."
#         python scripts/eval_ensemble.py \
#             --dataset "$DATASET" \
#             --gen_full "$DIR_FULL" \
#             --gen_fg "$DIR_FG" \
#             --gen_bg "$DIR_BG"
#     else
#         echo "⚠️ ATTENZIONE: Mancano cartelle per GOD S${s}_${ROI_GOD}."
#         echo "   Cercavo:"
#         echo "   - $DIR_FULL"
#         echo "   - $DIR_FG"
#         echo "   - $DIR_BG"
#     fi
# done

# ==========================================
# 2. VALUTAZIONE SHEN (Soggetti 1-3, ROI: ROI_VC)
# ==========================================
DATASET="shen"
ROI_SHEN="ROI_VC"
for s in {1..3}; do
    echo ""
    echo "----------------------------------------------------------"
    echo "🧠 Valutazione $DATASET (Soggetto $s - ROI: $ROI_SHEN)"
    echo "----------------------------------------------------------"
    # Adatto i percorsi per Shen (usando la logica dei tuoi vecchi script)
    DIR_FULL="${BASE_DIR}/reconstructions/S${s}_natural_inpaint_mask/single"
    DIR_FG="${BASE_DIR}/reconstructions/S${s}_natural_ALL_VERSIONS/fg_only"
    DIR_BG="${BASE_DIR}/reconstructions/S${s}_natural_ALL_VERSIONS/bg_only"

    if [ -d "$DIR_FULL" ] && [ -d "$DIR_FG" ] && [ -d "$DIR_BG" ]; then
        echo "✅ Trovate le 3 cartelle per S${s}. Avvio Late Fusion..."
        python scripts/eval_ensemble.py \
            --dataset "$DATASET" \
            --gen_full "$DIR_FULL" \
            --gen_fg "$DIR_FG" \
            --gen_bg "$DIR_BG"
    else
        echo "⚠️ ATTENZIONE: Mancano cartelle per SHEN S${s}_${ROI_SHEN}."
        echo "   Cercavo:"
        echo "   - $DIR_FULL"
        echo "   - $DIR_FG"
        echo "   - $DIR_BG"
    fi
done

echo ""
echo "🎉 VALUTAZIONE ENSEMBLE COMPLETATA!"