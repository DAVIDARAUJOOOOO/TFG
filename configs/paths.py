# Paths configured for local execution.
# The original notebook was developed and executed in Google Colab,
# For local execution, the Google Drive paths have been replaced

input_audio_folder = r"C:\Users\david\UNI\TFG\FILES\audios_castella" # Aquesta carpeta conté en local els audios originals, no unificats.
output_path = r"C:\Users\david\UNI\TFG\FILES\audios_castella_unificats\audios_castella_unificats" # Aquesta carpeta conté en local el resultat de la unificació dels audios després d'haver executat el codi
ground_truth_path = r"C:\Users\david\UNI\TFG\FILES\Los_okapis.txt" # Aquest fitxer conté en local el GT utilitzat per construir el dataset de Whisper. 
storage_path = r"C:\Users\david\UNI\TFG\FILES\transcripcions.json" # Per al dataset amb les característiques extretes de Whisper obtingudes després d'executar el codi (text, duració parla, etc.)
storage_path_2 = r"C:\Users\david\UNI\TFG\FILES\transcripcions_2.json" # Per al dataset amb les característiques extretes de Whisper i els embeddings de tot l'audio obtinguts després d'executar el codi
storage_path_3 = r"C:\Users\david\UNI\TFG\FILES\embeddings.json" # Per al dataset amb les característiques extretes de Whisper i els embeddings de fragments de 2 segons obtinguts després d'executar el codi

#Resultat de l'entrenament amb fragments de 2 segons després d'executar el codi dels diferents models
storage_xgb = r"C:\Users\david\UNI\TFG\FILES\modelo_xgb.pkl"
storage_rf = r"C:\Users\david\UNI\TFG\FILES\modelo_rf.pkl"
storage_svm = r"C:\Users\david\UNI\TFG\FILES\modelo_svm.pkl"
storage_knn = r"C:\Users\david\UNI\TFG\FILES\modelo_knn.pkl"
storage_lm = r"C:\Users\david\UNI\TFG\FILES\modelo_lr.pkl"
storage_lgbm = r"C:\Users\david\UNI\TFG\FILES\modelo_lgbm.pkl"