#!/bin/bash

# Attivazione ambiente (modifica il percorso se necessario)
source $(conda info --base)/etc/profile.d/conda.sh
conda activate labram

SUBJECTS=(1 2 3 4 5)
ROI="VC"  # Fissato solo su Visual Cortex

echo "======================================================================="
echo "📊 RIVALUTAZIONE VECCHI MODELLI (GOD) | ROI: ${ROI}"
echo "======================================================================="

EVAL_SCRIPT="scripts/evaluate_try.py"
BASE_OUT="/home/mfinocchiaro/Kamitani_fMRI/Dataset_Kamitani/preprocessed/fmri_saliency_decoding/results_final"
EVAL_DIR="eval_reports"
mkdir -p "$EVAL_DIR"

SPATIALS=("box")
EVAL_MODES=("full" "foreground" "background")

for SUBJ in "${SUBJECTS[@]}"; do
    echo -e "\n======================================================================="
    echo "▶ ELABORAZIONE SOGGETTO ${SUBJ}..."
    echo "======================================================================="
    
    REPORT_FILE="$EVAL_DIR/S${SUBJ}_${ROI}_ABLATION_REPORT.txt"
    echo "REPORT STUDIO DI ABLAZIONE (VECCHI MODELLI): SOGGETTO $SUBJ - ROI $ROI" > $REPORT_FILE
    echo "=======================================================================" >> $REPORT_FILE

    for SPATIAL in "${SPATIALS[@]}"; do
        echo -e "\n\n#######################################################################" >> $REPORT_FILE
        echo "                      VARIANTE SPAZIALE GENERAZIONE: $SPATIAL" >> $REPORT_FILE
        echo "#######################################################################" >> $REPORT_FILE

        for MODE in "${EVAL_MODES[@]}"; do
            echo -e "\n---> Valutazione Metrica: ${MODE^^} <---" >> $REPORT_FILE
            
          

            # --- B. SDXL INPAINT ---
            echo ">> SDXL 2-STEP INPAINT ($MODE)" >> $REPORT_FILE
            if [ -d "$BASE_OUT/sdxl_inpaint/S${SUBJ}_${ROI}_${SPATIAL}/single" ]; then
                python $EVAL_SCRIPT --dataset god --subject $SUBJ \
                    --eval_mode $MODE --gen_dir "$BASE_OUT/sdxl_inpaint/S${SUBJ}_${ROI}_${SPATIAL}/single" >> $REPORT_FILE 2>&1
            fi

            
        done
    done
    echo "✅ Completato S${SUBJ} - ${ROI}."
done

echo "🎉 TUTTI I SOGGETTI GOD COMPLETATI!"