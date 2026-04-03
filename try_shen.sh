#!/bin/bash

# Attivazione ambiente
source $(conda info --base)/etc/profile.d/conda.sh
conda activate labram

SUBJECTS=(1 2 3)
ROI="ROI_VC"  # Fissato solo su Visual Cortex per Shen
TEST_TYPE="natural"

SPATIAL_MODES=("none" "inpaint_mask" "inpaint_box")
EVAL_MODES=("full" "foreground" "background")

LOG_DIR="evaluation_logs_ablation"
mkdir -p "$LOG_DIR"

echo "🚀 Inizio Studio di Ablazione Shen 2019 (Locale) | ROI: $ROI"
echo "Soggetti: ${SUBJECTS[*]}"
echo "=============================================================================="

EVAL_SCRIPT="scripts/evaluate_try.py"

for subj in "${SUBJECTS[@]}"; do
    echo -e "\n\n======================================================="
    echo "▶ SOGGETTO $subj"
    echo "======================================================="
    
    for spatial in "${SPATIAL_MODES[@]}"; do
        for eval_mode in "${EVAL_MODES[@]}"; do
            
            # --- A. SDXL ---
            GEN_DIR_SDXL="reconstructions/S${subj}_${TEST_TYPE}_${spatial}/single"
            LOG_FILE_SDXL="${LOG_DIR}/eval_SDXL_S${subj}_spatial-${spatial}_eval-${eval_mode}.txt"
            
            if [ -d "$GEN_DIR_SDXL" ]; then
                echo "📊 Valutazione SDXL S$subj | Generato: $spatial | Metrica: $eval_mode..."
                python $EVAL_SCRIPT --dataset shen --subject "$subj" --test_type "$TEST_TYPE" \
                    --eval_mode "$eval_mode" --gen_dir "$GEN_DIR_SDXL" > "$LOG_FILE_SDXL" 2>&1
            else
                echo "⚠️ Cartella mancante: $GEN_DIR_SDXL"
            fi

            
            
        done
    done
done

echo -e "\n🎉 ✅ TUTTI GLI ESPERIMENTI SHEN SONO TERMINATI!"