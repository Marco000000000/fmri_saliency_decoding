import os
import pickle
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

# ================= CONFIGURAZIONE =================
# Percorso del file PKL generato dal metodo Kamitani (con i CLIP embedding)
PKL_PATH = "/home/mfinocchiaro/Kamitani_fMRI/Dataset_Kamitani/preprocessed/fmri_saliency_decoding/kamitani_method/results/GenericObjectDecoding.pkl"

SUBJECTS = [1, 2, 3, 4, 5]
# Elenca tutte le ROI che hai calcolato
ROIS = ["VC", "V1", "V2", "V3", "V4", "LOC", "FFA", "PPA"] 
# ==================================================

def calculate_n_way_accuracy(pred_features, true_features):
    """
    Calcola l'accuratezza Top-1 e Top-5 usando la Cosine Similarity.
    pred_features: numpy array (N, 1280)
    true_features: numpy array (N, 1280)
    """
    preds = torch.tensor(pred_features, dtype=torch.float32)
    gts = torch.tensor(true_features, dtype=torch.float32)
    
    # Normalizzazione L2 per Cosine Similarity
    preds_norm = F.normalize(preds, p=2, dim=1)
    gts_norm = F.normalize(gts, p=2, dim=1)
    
    # Calcolo matrice di similarità (N x N)
    # sim_matrix[i, j] = similarità tra la predizione i e la ground truth j
    sim_matrix = torch.mm(preds_norm, gts_norm.t())
    
    num_samples = preds.shape[0]
    top1_correct = 0
    top5_correct = 0
    
    for i in range(num_samples):
        # Ordiniamo gli indici dal più simile al meno simile
        ranked_indices = torch.argsort(sim_matrix[i], descending=True)
        
        # Troviamo in che posizione si trova la Ground Truth corretta (che è l'indice 'i')
        rank = (ranked_indices == i).nonzero(as_tuple=True)[0].item()
        
        if rank == 0:
            top1_correct += 1
        if rank < 5:
            top5_correct += 1
            
    top1_acc = (top1_correct / num_samples) * 100
    top5_acc = (top5_correct / num_samples) * 100
    
    return top1_acc, top5_acc

def main():
    print("🧠 CALCOLO PURE SEMANTIC (METODO KAMITANI) 🧠")
    
    if not os.path.exists(PKL_PATH):
        print(f"❌ File non trovato: {PKL_PATH}")
        return
        
    print(f"📂 Caricamento file: {os.path.basename(PKL_PATH)}...")
    with open(PKL_PATH, 'rb') as f:
        results_df = pickle.load(f)
        
    results_list = []

    for roi in ROIS:
        for sub in SUBJECTS:
            sbj_str = f"Subject{sub}"
            
            # Filtriamo il dataframe per Soggetto e ROI
            filtered_df = results_df[(results_df['subject'] == sbj_str) & (results_df['roi'] == roi)]
            
            if filtered_df.empty:
                # Se non c'è questo soggetto/roi, saltiamo
                continue
                
            row = filtered_df.iloc[0]
            
            # Estraiamo le feature Predette e le True (Ground Truth)
            # NOTA: I nomi esatti dipendono da come bdpy ha salvato il file. Di default sono questi:
            try:
                preds = row['predicted_feature_averaged_percept']
                gts = row['true_feature_averaged_percept']
                
                # Calcolo metriche
                top1, top5 = calculate_n_way_accuracy(preds, gts)
                
                results_list.append({
                    "ROI": roi,
                    "Subject": sbj_str,
                    "Top-1 (%)": top1,
                    "Top-5 (%)": top5
                })
            except KeyError as e:
                print(f"⚠️ Errore per {sbj_str} {roi}: Colonna {e} non trovata nel DataFrame.")
                continue

    if not results_list:
        print("❌ Nessun dato elaborato. Controlla i nomi delle ROI o del DataFrame.")
        return

    # Trasformiamo in DataFrame per visualizzarlo e salvarlo comodamente
    df_results = pd.DataFrame(results_list)
    
    print("\n" + "="*50)
    print(f"{'ROI':<10} | {'SOGGETTO':<12} | {'TOP-1':<10} | {'TOP-5':<10}")
    print("-" * 50)
    
    for _, r in df_results.iterrows():
        print(f"{r['ROI']:<10} | {r['Subject']:<12} | {r['Top-1 (%)']:05.2f}%    | {r['Top-5 (%)']:05.2f}%")
        
    print("="*50)

    # Calcoliamo anche la media per ogni ROI
    print("\n📊 MEDIA PER ROI:")
    df_mean = df_results.groupby('ROI')[['Top-1 (%)', 'Top-5 (%)']].mean().reset_index()
    for _, r in df_mean.iterrows():
        print(f"ROI {r['ROI']:<6}: Top-1 = {r['Top-1 (%)']:05.2f}% | Top-5 = {r['Top-5 (%)']:05.2f}%")

    # Salva in CSV
    out_csv = "kamitani_pure_semantic_all_rois.csv"
    df_results.to_csv(out_csv, index=False)
    print(f"\n✅ Risultati completi salvati in '{out_csv}'")

if __name__ == "__main__":
    main()