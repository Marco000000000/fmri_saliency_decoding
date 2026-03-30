import bdpy
import numpy as np

def main():
    h5_path = "/home/mfinocchiaro/Kamitani_fMRI/Dataset_Kamitani/preprocessed/fmri_saliency_decoding/data/Shen2019/fmri/sub-01_Imagery_fmriprep_volume_native.h5"
    bdata = bdpy.BData(h5_path)

    # 1. Estraiamo gli ID numerici (i numeri da 1 a 26)
    ids = bdata.select('imagery_image_index')[:, 0]
    
    # 2. Estraiamo le etichette testuali (i nomi veri in inglese!)
    # bdpy.get_label() traduce la colonna nel suo formato testuale originale
    names = bdata.get_label('imagery_name')
    
    print("🧠 DIZIONARIO DELLE CATEGORIE IMMAGINATE (Shen 2019)\n" + "="*50)
    
    # Troviamo gli ID unici (ignorando gli zeri del riposo)
    unique_ids = np.unique(ids[ids > 0])
    
    for uid in sorted(unique_ids):
        # Troviamo la prima riga in cui il soggetto ha immaginato questo ID
        idx = np.where(ids == uid)[0][0]
        
        # Stampiamo l'associazione
        print(f"ID {int(uid):<3} ➡️  {names[idx]}")
        
    print("="*50)
    print("Adesso sai esattamente cosa sta cercando di disegnare SDXL!")

if __name__ == "__main__":
    main()