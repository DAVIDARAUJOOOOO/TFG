from src.data.dataset_2.chunks import chunk_audio
from src.data.dataset_2.silence import is_silent
from src.data.dataset_2.get_embedding import get_embedding
from src.utils.json_io import load_json, save_json
import whisper
import pandas as pd


def build_trainer_dataset_2(storage_path, storage_path_3):
  
    results = load_json(storage_path)

    df_2=[]
    model=whisper.load_model("base")
    for result in results:
        arxiu=result["arxiu"]
        audio=whisper.load_audio(arxiu)
        chunks=chunk_audio(audio)
        for i,chunk in enumerate(chunks):
            if is_silent(chunk):
                continue

            emb=get_embedding(model,chunk)
            df_2.append({
            "arxiu":arxiu,
            "embeddings":emb.tolist(),
            "label":result["dislexia"],
            "curs":result["curs"],
            "id_audio":i})

    save_json(storage_path_3, df_2)
    df_2=pd.DataFrame(df_2)
    return df_2