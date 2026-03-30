#!/bin/bash
#SBATCH --job-name=Kam_Baseline
#SBATCH --output=logs/kam_%A_%a.out
#SBATCH --error=logs/kam_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=train
#SBATCH --array=0-39

source $(conda info --base)/etc/profile.d/conda.sh
conda activate labram

SUBJECTS=(1 2 3 4 5)
ROIS=("V1" "V2" "V3" "V4" "LOC" "FFA" "PPA" "VC")

SUBJ_IDX=$(( SLURM_ARRAY_TASK_ID / 8 ))
ROI_IDX=$(( SLURM_ARRAY_TASK_ID % 8 ))
SUBJ=${SUBJECTS[$SUBJ_IDX]}
ROI=${ROIS[$ROI_IDX]}

EVAL_SCRIPT="scripts/evaluate_images.py"
BASE_OUT="results_final/kamitani_baseline"
REPORT_FILE="eval_reports/KAMITANI_S${SUBJ}_${ROI}_REPORT.txt"

echo "REPORT VALUTAZIONE: KAMITANI BASELINE S$SUBJ - ROI $ROI" > $REPORT_FILE
echo "==================================================" >> $REPORT_FILE

SPATIALS=("none" "mask" "box")

for SPATIAL in "${SPATIALS[@]}"; do
    echo "🧠 Generazione Kamitani S${SUBJ} ROI ${ROI} [${SPATIAL}]..."
    
    # 1. Genera Immagini
    python scripts/inference_kamitani_confronto.py --subject $SUBJ --rois $ROI --spatial $SPATIAL
    
    # 2. Valuta Immagini
    echo -e "\n\n#######################################################################" >> $REPORT_FILE
    echo "                      VARIANTE SPAZIALE: $SPATIAL" >> $REPORT_FILE
    echo "#######################################################################" >> $REPORT_FILE
    
    # Titolo formattato per farsi leggere dal nostro estrattore Python!
    echo -e "\n--- MODELLO: KAMITANI 2019 BASELINE | SPAZIO: $SPATIAL ---" >> $REPORT_FILE
    
    python $EVAL_SCRIPT --subject $SUBJ --rois $ROI \
        --gen_dir "$BASE_OUT/S${SUBJ}_${ROI}_${SPATIAL}/single" >> $REPORT_FILE 2>&1
done

echo "✅ Kamitani Baseline completata per S${SUBJ} - ${ROI}."