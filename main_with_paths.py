import os
import pandas as pd

from configs import paths as cfg_paths
from src.data.built_plot_df import build_plot_df
from src.visualization.language_plot import (
    plot_balance_global,
    plot_distribution_dislexia,
    plot_language_violin,
    set_plot_style
)
from src.utils.json_io import load_json
from src.preprocessing.feature_engineering import build_features, run_correlation_analysis
from src.training.training_1.training_1_piepline import run_training1
from src.training.training_mlp.pipeline_mlp import run_training_mlp
from src.training.training_2.training_2_pipeline import run_training2
from src.training.training_2b.training_2b_pipeline import run_training2b
from src.training.training_3.training_3_pipeline import run_training3
from src.visualization import plot_balance_chunks

def main():
    print("Començant el procés amb paths cnofigurats y datasets creats...")

    if not os.path.exists(cfg_paths.output_path):
        raise FileNotFoundError(f"No s'ha trobat la carpeta de sortida: {cfg_paths.output_path}")

    set_plot_style()
    df_plot, audio_no_trobat = build_plot_df(cfg_paths.output_path)

    print(f"Audios processats: {len(df_plot)}")
    if audio_no_trobat:
        print(f"Audios no trobats al regex: {len(audio_no_trobat)}")

    plot_language_violin(df_plot)
    plot_balance_global(df_plot)
    plot_distribution_dislexia(df_plot)

    results=load_json(cfg_paths.storage_path_2)
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


    #####SECOND B TRAINING PIPELINE#####

    print(f"Processant dades de fragmnts de 2 segons...")
    results=load_json(cfg_paths.storage_path_3)
    df_2=pd.DataFrame(results)
    plot_balance_chunks(df_2)
    run_training2b(df_2)

    #####THIRD TRAINING PIPELINE#####
    run_training3(df)

if __name__ == "__main__":
    main()
