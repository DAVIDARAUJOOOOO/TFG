import json
import pandas as pd

from src.data.dataset_1.whispers_features import extract_whisper_features
from src.data.dataset_1.text_metrics import compute_metrics
from src.utils.json_io import save_json


def build_training_dataset(model, df, gt, storage_path):

    results = df.to_dict(orient="records")

    for audio in results:
        arxiu = audio["arxiu"]
        print(f"Processant: {arxiu}")

        feats = extract_whisper_features(model, arxiu, audio)
        audio.update(feats)

        metrics = compute_metrics(gt, feats["text"], feats["duracio_parla"])
        audio.update(metrics)

    save_json(results, storage_path)

    return pd.DataFrame(results)
