import os
import glob
import re
import pandas as pd
import numpy as np

def clean_metric_name(raw_name):
    """Pulisce i nomi delle metriche rimuovendo frecce e spiegazioni tra parentesi quadre."""
    name = re.sub(r'[↓↑]', '', raw_name)
    name = re.sub(r'\[.*?\]', '', name)
    name = name.replace('(fMRI 1280D vs Real 1280D)', '')
    name = name.replace('(IoU su Maschera Binaria)', 'IoU')
    name = name.replace('(Cosine Sim su Mappa Continua)', 'Cosine')
    name = name.replace('(MSE su Mappa Continua)', 'MSE')
    return name.strip()

def parse_file(filepath, is_shen=False):
    """Parsa il file e restituisce una lista di dizionari con i risultati."""
    results = []
    filename = os.path.basename(filepath)
    
    current_model = None
    current_spatial = None
    subject_id = None
    
    # Se è SHEN, estraiamo modello e spazio dal nome del file
    if is_shen:
        # Es: eval_SDXL_S1_inpaint_mask.txt
        name_no_ext = filename.replace('.txt', '')
        parts = name_no_ext.split('_')
        
        raw_model = parts[1] # SDXL o KAND
        subj_str = parts[2]  # S1
        subject_id = f"Shen_{subj_str}"
        
        raw_mode = "_".join(parts[3:]) # none, inpaint_mask, inpaint_box
        
        # Mappiamo i nomi per allinearli perfettamente con GOD
        current_model = "SDXL 2-STEP INPAINT" if raw_model == "SDXL" else "KANDINSKY 2-STEP"
        current_spatial = "mask" if raw_mode == "inpaint_mask" else ("box" if raw_mode == "inpaint_box" else "none")

    else:
        # Se è GOD, estraiamo il soggetto dal nome (Es: S1_VC_EVAL_ONLY_REPORT.txt)
        try:
            subject_id = f"GOD_S{int(filename.split('_')[0][1:])}"
        except ValueError:
            subject_id = f"GOD_{filename.split('_')[0]}"

    with open(filepath, 'r') as f:
        lines = f.readlines()

    header_pattern = re.compile(r"--- MODELLO: (.*?) \| SPAZIO: (.*?) ---")
    global_pattern = re.compile(r"Medie Globali Spaziali:\s*IoU:\s*([0-9.]+)\s*\|\s*Cosine:\s*([0-9.]+)\s*\|\s*MSE:\s*([0-9.]+)")

    current_metrics = {}

    for line in lines:
        line = line.strip()
        
        # Se è GOD, dobbiamo trovare le intestazioni per capire a che modello siamo
        if not is_shen:
            h_match = header_pattern.search(line)
            if h_match:
                # Salva il blocco precedente se esiste
                if current_model is not None and current_metrics:
                    for m_key, m_val in current_metrics.items():
                        results.append({'Subject': subject_id, 'Model': current_model, 'Spatial Variant': current_spatial, 'Metric': m_key, 'Value': m_val})
                current_model = h_match.group(1).strip()
                current_spatial = h_match.group(2).strip()
                current_metrics = {}
                continue
        
        if current_model is None:
            continue

        # Cattura le "Medie Globali Spaziali"
        if "Medie Globali Spaziali" in line:
            g_match = global_pattern.search(line)
            if g_match:
                current_metrics["Global IoU"] = float(g_match.group(1))
                current_metrics["Global Cosine"] = float(g_match.group(2))
                current_metrics["Global MSE"] = float(g_match.group(3))
            continue

        # Cattura i dati dalle Tabelle 1 e 2
        if '|' in line and 'Metrica' not in line and '---' not in line and '===' not in line:
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 3:
                raw_metric = parts[0]
                metric_name = clean_metric_name(raw_metric)
                
                # Top-1
                val1 = parts[1].replace('%', '')
                if val1.replace('.', '', 1).isdigit() or (val1.startswith('-') and val1[1:].replace('.', '', 1).isdigit()):
                    current_metrics[f"{metric_name} (Top-1)"] = float(val1)
                
                # Top-5
                val2 = parts[2].replace('%', '')
                if val2.replace('.', '', 1).isdigit() or (val2.startswith('-') and val2[1:].replace('.', '', 1).isdigit()):
                    current_metrics[f"{metric_name} (Top-5)"] = float(val2)

    # Salva l'ultimo blocco letto
    if current_model is not None and current_metrics:
        for m_key, m_val in current_metrics.items():
            results.append({'Subject': subject_id, 'Model': current_model, 'Spatial Variant': current_spatial, 'Metric': m_key, 'Value': m_val})

    return results

def main():
    god_dir = "eval_reports"
    shen_dir = "evaluation_logs"
    output_csv = "final_combined_results_pivot.csv"

    god_files = glob.glob(os.path.join(god_dir, "S*_VC_EVAL_ONLY_REPORT.txt"))
    shen_files = glob.glob(os.path.join(shen_dir, "eval_*_S*.txt"))
    
    print(f"📂 Trovati {len(god_files)} file GOD e {len(shen_files)} file SHEN. Inizio il parsing combinato...")

    all_results = []

    # Parsa i file GOD
    for f in god_files:
        all_results.extend(parse_file(f, is_shen=False))

    # Parsa i file SHEN
    for f in shen_files:
        all_results.extend(parse_file(f, is_shen=True))

    if not all_results:
        print("⚠️ Nessun dato trovato nei file!")
        return

    # Creazione DataFrame piatto
    df = pd.DataFrame(all_results)

    # Creazione Tabella PIVOT
    pivot_df = df.pivot_table(
        index=['Model', 'Spatial Variant', 'Metric'], 
        columns='Subject', 
        values='Value'
    )

    # Calcolo Media e Standard Deviation considerando tutti i soggetti trovati
    pivot_df['Mean (All)'] = pivot_df.mean(axis=1)
    pivot_df['Std (All)'] = pivot_df.std(axis=1)

    pivot_df['Mean (All)'] = pivot_df['Mean (All)'].round(2)
    pivot_df['Std (All)'] = pivot_df['Std (All)'].round(2)

    # Reimposta l'indice per mostrare bene le colonne testuali a sinistra
    pivot_df = pivot_df.reset_index()

    # Ordina: prima Modello, poi Spazio (none, mask, box), poi Metrica
    spatial_order = {'none': 0, 'mask': 1, 'box': 2}
    pivot_df['Spatial_Rank'] = pivot_df['Spatial Variant'].map(spatial_order)
    pivot_df = pivot_df.sort_values(by=['Model', 'Spatial_Rank', 'Metric']).drop('Spatial_Rank', axis=1)

    print("\n" + "="*140)
    print("📊 TABELLA PIVOT UNIFICATA: GOD E SHEN (Tutti i Soggetti, Media e Std)")
    print("="*140)
    print(pivot_df.to_string(index=False))
    print("="*140)

    pivot_df.to_csv(output_csv, index=False)
    print(f"\n✅ Tabella esportata con successo in: {output_csv}")

if __name__ == "__main__":
    main()