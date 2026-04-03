import os
import re
import glob
import pandas as pd
import numpy as np

def parse_logs(log_dir="logs"):
    parsed_data = {}
    subjects_god = []
    subjects_shen = []

    # Cerca tutti i file di log nella directory specificata
    log_files = glob.glob(f"{log_dir}/split_*.out")
    if not log_files:
        print(f"❌ Nessun file trovato in {log_dir}/")
        return

    for file in log_files:
        with open(file, 'r', encoding='utf-8') as f:
            content = f.read()

        # 1. Trova Dataset e Soggetto dal banner iniziale
        m = re.search(r'DATASET:\s*([A-Z]+)\s*\|\s*SOGGETTO:\s*(\d+)', content)
        if not m: continue
        
        dataset = m.group(1).upper()
        subject = m.group(2)
        
        # 2. Nuova Regex "flessibile": 
        # Cattura sia ">> Valutazione MODO: [X]" che ">> Valutazione [TEST_TYPE] - MODO: [X]"
        matches = list(re.finditer(r'>> Valutazione(?:\s+([A-Z]+)\s+-\s+)?\s*MODO:\s+([A-Z]+)', content))
        
        for i, match in enumerate(matches):
            test_type = match.group(1) if match.group(1) else "NATURAL"
            mode_name = match.group(2)
            
            # Isola solo il blocco di testo di questa specifica valutazione
            start_idx = match.end()
            end_idx = matches[i+1].start() if i+1 < len(matches) else len(content)
            chunk = content[start_idx:end_idx]
            
            # Gestione colonne Shen Imagery vs Natural
            col_name = f"{dataset}_S{subject}"
            if test_type == "IMAGERY":
                col_name += "_IMAGERY"
                
            # Aggiorna la lista delle colonne disponibili
            if dataset == "GOD" and col_name not in subjects_god: subjects_god.append(col_name)
            if dataset == "SHEN" and col_name not in subjects_shen: subjects_shen.append(col_name)

            for line in chunk.split('\n'):
                # Parsing TABELLA 1
                if '|' in line and '%' in line and '/' in line:
                    parts = [p.strip() for p in line.split('|')]
                    if len(parts) >= 6 and parts[0] != "Metodo":
                        method = parts[0]
                        lpips_val = float(parts[1])
                        
                        lpips_name = f"[{mode_name}] Tab1 - {method} (LPIPS)"
                        if lpips_name not in parsed_data: parsed_data[lpips_name] = {}
                        parsed_data[lpips_name][col_name] = lpips_val

                        metrics = ['CLIP-B', 'CLIP-XL', 'Alex2', 'Alex5', 'Alex7']
                        for j, m_name in enumerate(metrics):
                            t1, t5 = map(float, parts[j+2].replace('%', '').split('/'))
                            row_name = f"[{mode_name}] Tab1 - {method} ({m_name})"
                            if row_name not in parsed_data: parsed_data[row_name] = {}
                            parsed_data[row_name][col_name] = (t1, t5)

                # Parsing TABELLA 2
                elif '|' in line and '%' in line and '/' not in line:
                    parts = [p.strip() for p in line.split('|')]
                    if len(parts) == 3 and not parts[0].startswith('Metrica'):
                        metric = parts[0]
                        t1 = float(parts[1].replace('%', ''))
                        t5 = float(parts[2].replace('%', ''))
                        row_name = f"[{mode_name}] Tab2 - {metric}"
                        if row_name not in parsed_data: parsed_data[row_name] = {}
                        parsed_data[row_name][col_name] = (t1, t5)

    # Ordinamento e composizione colonne
    subjects_god.sort()
    subjects_shen.sort()
    all_subjects = subjects_god + subjects_shen
    
    rows = []
    for metric, values in parsed_data.items():
        row = {'Metrica': metric}
        god_t1, god_t5, shen_t1, shen_t5 = [], [], [], []
        
        for sub in all_subjects:
            val = values.get(sub, None)
            if val is None:
                row[sub] = "-"
            elif isinstance(val, tuple):
                row[sub] = f"{val[0]:.2f}% / {val[1]:.2f}%"
                if sub in subjects_god:
                    god_t1.append(val[0]); god_t5.append(val[1])
                elif sub in subjects_shen and not sub.endswith("_IMAGERY"): 
                    # Nelle medie SHEN consideriamo solo i test naturali
                    shen_t1.append(val[0]); shen_t5.append(val[1])
            else:
                row[sub] = f"{val:.4f}"
                if sub in subjects_god: god_t1.append(val)
                elif sub in subjects_shen and not sub.endswith("_IMAGERY"): shen_t1.append(val)

        # Helper per calcolare le medie globali
        def calc_avg(t1_list, t5_list):
            if not t1_list: return "-"
            if not t5_list: return f"{np.mean(t1_list):.4f}" # Per LPIPS
            return f"{np.mean(t1_list):.2f}% / {np.mean(t5_list):.2f}%"

        row['AVG_GOD'] = calc_avg(god_t1, god_t5)
        row['AVG_SHEN'] = calc_avg(shen_t1, shen_t5)
        row['AVG_TOTALE'] = calc_avg(god_t1 + shen_t1, god_t5 + shen_t5)

        rows.append(row)

    df = pd.DataFrame(rows)
    cols = ['Metrica'] + subjects_god + ['AVG_GOD'] + subjects_shen + ['AVG_SHEN', 'AVG_TOTALE']
    df = df[[c for c in cols if c in df.columns]]

    df.to_csv("Mega_Tabella_Risultati.csv", index=False)
    print("✅ Mega Tabella salvata con successo in 'Mega_Tabella_Risultati.csv'!")

if __name__ == "__main__":
    parse_logs("logs")