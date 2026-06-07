import json
import numpy as np
import whisper

from src.data.embeddings.whisper_embeddings import complete_embedding
from src.data.embeddings.pca_analysis import run_pca_analysis
from src.utils.json_io import save_json


def build_embeddings(results, storage_path_2, run_pca=True):

    model = whisper.load_model("base")

    embeddings_audio = []

    for result in results:
        print(f"Processant audio {result['arxiu']}")
        emb = complete_embedding(result["arxiu"], model)
        embeddings_audio.append(emb)

    embeddings_audio = np.array(embeddings_audio)

    if run_pca:
        run_pca_analysis(embeddings_audio)

    for i, result in enumerate(results):
        result["embeddings"] = embeddings_audio[i].tolist()

    save_json(storage_path_2, results)

    return results, embeddings_audio