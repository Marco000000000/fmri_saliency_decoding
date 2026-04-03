#!/bin/bash
#SBATCH --job-name=SplitSemantics
#SBATCH --output=logs/split_%A_%a.out
#SBATCH --error=logs/split_%A_%a.err
#SBATCH --time=24:00:00               # Tempo stimato (modifica se serve)
#SBATCH --partition=gpu
#SBATCH --qos=train
#SBATCH --gres=gpu:1
#SBATCH --array=0-7                   # 8 Task totali (0-4 per GOD, 5-7 per Shen)

# Assicurati di aver attivato il tuo ambiente conda prima!
# source /path/to/miniconda3/bin/activate labram (Scommenta e adatta se necessario)

# Colori per i log
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m'

# ==========================================
# MAPPATURA ARRAY ID -> DATASET & SOGGETTO
# ==========================================
TASK_ID=$SLURM_ARRAY_TASK_ID

if [ "$TASK_ID" -le 4 ]; then
    # Indici 0-4 mappano ai soggetti GOD 1-5
    DATASET="god"
    SUBJECT=$((TASK_ID + 1))
    ROI="VC"
elif [ "$TASK_ID" -le 7 ]; then
    # Indici 5-7 mappano ai soggetti Shen 1-3
    DATASET="shen"
    SUBJECT=$((TASK_ID - 4))
    ROI="ROI_VC"
else
    echo "Task ID non valido!"
    exit 1
fi
source $(conda info --base)/etc/profile.d/conda.sh

# 2. Attiva l'ambiente (Questa è la tua "concatenazione")
conda activate labram
echo -e "${GREEN}====================================================${NC}"
echo -e "${GREEN}🚀 INIZIO JOB [ID: ${TASK_ID}] -> DATASET: ${DATASET^^} | SOGGETTO: ${SUBJECT} ${NC}"
echo -e "${GREEN}====================================================${NC}"

# ==========================================
# ESECUZIONE SPECIFICA PER DATASET
# ==========================================

if [ "$DATASET" == "god" ]; then

    echo -e "\n${CYAN}--- FASE 1: TRAINING GOD (Sub ${SUBJECT}) ---${NC}"
    echo -e "${YELLOW}>> Train Foreground...${NC}"
    python scripts/train_split_god.py --subject ${SUBJECT} --rois ${ROI} --mask_mode foreground
    echo -e "${YELLOW}>> Train Background...${NC}"
    python scripts/train_split_god.py --subject ${SUBJECT} --rois ${ROI} --mask_mode background

    echo -e "\n${CYAN}--- FASE 2: INFERENZA GOD (Sub ${SUBJECT}) ---${NC}"
    python scripts/inference_split_all_god.py --subject ${SUBJECT} --rois ${ROI}

    echo -e "\n${CYAN}--- FASE 3: EVALUATION GOD (Sub ${SUBJECT}) ---${NC}"
    MODES=("full" "foreground" "background")
    for mode in "${MODES[@]}"; do
        echo -e "${YELLOW}>> Valutazione MODO: ${mode^^}${NC}"
        python scripts/evaluate_all_modes.py --dataset god --subject ${SUBJECT} --roi ${ROI} --eval_mode $mode
    done

elif [ "$DATASET" == "shen" ]; then

    echo -e "\n${CYAN}--- FASE 1: TRAINING SHEN (Sub ${SUBJECT}) ---${NC}"
    echo -e "${YELLOW}>> Train Foreground...${NC}"
    python scripts/train_split.py --subject ${SUBJECT} --roi ${ROI} --mask_mode foreground
    echo -e "${YELLOW}>> Train Background...${NC}"
    python scripts/train_split.py --subject ${SUBJECT} --roi ${ROI} --mask_mode background

    echo -e "\n${CYAN}--- FASE 2: INFERENZA SHEN (Sub ${SUBJECT}) ---${NC}"
    echo -e "${YELLOW}>> Natural Images...${NC}"
    python scripts/inference_split_all.py --subject ${SUBJECT} --roi ${ROI} --test_type natural
    echo -e "${YELLOW}>> Imagery (Sogni)...${NC}"
    python scripts/inference_split_all.py --subject ${SUBJECT} --roi ${ROI} --test_type imagery

    echo -e "\n${CYAN}--- FASE 3: EVALUATION SHEN (Sub ${SUBJECT}) ---${NC}"
    MODES=("full" "foreground" "background")
    for mode in "${MODES[@]}"; do
        echo -e "${YELLOW}>> Valutazione NATURAL - MODO: ${mode^^}${NC}"
        python scripts/evaluate_all_modes.py --dataset shen --subject ${SUBJECT} --roi ${ROI} --test_type natural --eval_mode $mode
    done
    
    echo -e "${YELLOW}>> Valutazione IMAGERY - MODO: FULL${NC}"
    python scripts/evaluate_all_modes.py --dataset shen --subject ${SUBJECT} --roi ${ROI} --test_type imagery --eval_mode full

fi

echo -e "\n${GREEN}✅ JOB [ID: ${TASK_ID}] COMPLETATO CON SUCCESSO!${NC}"