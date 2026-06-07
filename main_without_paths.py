import argparse
import os
import pandas as pd

from src.data.built_plot_df import build_plot_df
from src.data.dataset_1.dataset_builder import build_training_dataset
from src.data.dataset_2.dataset_builder import build_trainer_dataset_2
from src.data.embeddings.embeddings_pipeline import build_embeddings

from src.utils.files_utils import unificar_audios

from src.visualization.language_plot import (
    plot_balance_global,
    plot_distribution_dislexia,
    plot_language_violin,
    set_plot_style
)

from src.preprocessing.feature_engineering import build_features, run_correlation_analysis
from src.training.training_1.training_1_piepline import run_training1   
from src.training.training_mlp.pipeline_mlp import run_training_mlp
from src.training.training_2.training_2_pipeline import run_training2
from src.preprocessing.feature_engineering import expand_embeddings
from src.visualization.plot_balance_chunks import plot_balance_chunks
from src.training.training_2b.training_2b_pipeline import run_training2b
from src.training.training_3.training_3_pipeline import run_training3

DEFAULT_INPUT_FOLDER = r"Your\Path\To\Input\Audio\Folder" # Aquesta carpeta hauria de contenir els audios originals, no unificats. El codi s'encarregarà d'unificar-los i guardar-los a la carpeta de sortida.
DEFAULT_OUTPUT_FOLDER = r"Your\Path\To\Output\Audio\Folder" # Aquesta carpeta serà on es guardaran els audios unificats. El codi buscarà els audios en aquesta carpeta per construir el dataset i generar les visualitzacions. Assegura't que aquesta carpeta existeix abans d'executar el codi, ja que el codi no crearà la carpeta si no existeix.
DEFAULT_GT_PATH = r"Your\Path\To\GT.txt" # Ruta al fitxer de GT que s'utilitzarà per construir el dataset de Whisper. Aquest fitxer hauria de seguir el format especificat al projecte (amb les transcripcions i les etiquetes associades als audios). Assegura't que aquesta ruta és correcta i que el fitxer existeix abans d'executar el codi.
DEFAULT_STORAGE_WHISPER = r"storage_whisper.json" # Per al dataset amb les característiques extretes de Whisper (text, duració parla, etc.)
DEFAULT_STORAGE_EMB = r"storage_embeddings.json" # Per als embeddings de tot l'audio amb les característiques de Whisper
DEFAULT_STORAGE_EMB_2 = r"storage_embeddings_2.json" # Per als embeddings de fragments de 2 segons


def main():
    print("Començant el procés desde 0...")
    parser = argparse.ArgumentParser(description="Executant pipeline completa des de 0")
    parser.add_argument("--input-folder", default=DEFAULT_INPUT_FOLDER, help="Carpeta d'origen dels audios")
    parser.add_argument("--output-folder", default=DEFAULT_OUTPUT_FOLDER, help="Carpeta de sortida dels audios unificats")
    parser.add_argument("--gt-path", default=DEFAULT_GT_PATH, help="Ruta al fitxer de GT")
    parser.add_argument("--storage-whisper", default=DEFAULT_STORAGE_WHISPER, help="Ruta per guardar el dataset de característiques NLP")
    parser.add_argument("--storage-emb", default=DEFAULT_STORAGE_EMB, help="Ruta per guardar el dataset amb embeddings de tot l'audio juntament amb les característiques de Whisper")
    parser.add_argument("--storage-emb-2", default=DEFAULT_STORAGE_EMB_2, help="Ruta per guardar el dataset amb embeddings de fragments de 2 segons")
    args = parser.parse_args()

    if not os.path.exists(args.input_folder):
        raise FileNotFoundError(f"No s'ha trobat la carpeta d'entrada: {args.input_folder}")


    unificar_audios(args.input_folder, args.output_folder)
    set_plot_style()
    df_plot, audio_no_trobat = build_plot_df(args.output_folder)

    print(f"Audios processats: {len(df_plot)}")
    if audio_no_trobat:
        print(f"Audios no trobats al regex: {len(audio_no_trobat)}")

    plot_language_violin(df_plot)
    plot_balance_global(df_plot)
    plot_distribution_dislexia(df_plot)

    print("Construint dataset Whisper...")

    df_whisper = build_training_dataset(
        df=df_plot,
        gt_path=args.gt_path,
        storage_path=args.storage_whisper
    )

    print(f"Audios processats (Whisper): {len(df_whisper)}")


    print("Generant embeddings...")

    results, embeddings = build_embeddings(
        results=df_whisper.to_dict(orient="records"),
        storage_path_2=args.storage_emb,
        run_pca=True
    )
    features=["arxiu","confianca_mitjana",
          "confianca_std","ritme_std","silencis",
          "cer","wer","velocitat","ratio","curs","lm","embeddings","dislexia"]

    df=pd.DataFrame(results)
    df=df[features]

    df = build_features(df)
    run_correlation_analysis(df)

    #####FIRST TRAINING PIPELINE#####
    run_training1(df)

    #####MLP PIPELINE#####
    run_training_mlp(df)

    #####SECOND TRAINING PIPELINE#####
    run_training2(df)

    print(f"Generant df amb fragments de 2 segons...")
    df_2=build_trainer_dataset_2(storage_path=args.storage_whisper, storage_path_3=args.storage_emb_2)

    #####SECOND B TRAINING PIPELINE#####
    df_2=expand_embeddings(df_2)
    plot_balance_chunks(df_2)
    run_training2b(df_2)

    #####THIRD TRAINING PIPELINE#####
    run_training3(df)

if __name__ == "__main__":
    main()
