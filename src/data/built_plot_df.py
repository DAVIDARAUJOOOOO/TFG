import os
import re
import warnings

import librosa
import pandas as pd

from src.utils.language_utils import obtenir_llengua


def build_plot_df(output_path, l1_regex= r"llm_+(.+?)(?:_\d+)?_escola"):
    warnings.filterwarnings("ignore")
    
    map_llengua={
    "castellano": "castella",
    "espanol": "castella",
    "espanyol": "castella",
    "refeno_arab": "rifeny"
}
    results = []
    audio_no_trobat = []

    for audio in os.listdir(output_path):
        exemple = os.path.join(output_path, audio)

        parts = audio.split("_")
        classe_audio = parts[-2]
        curs = classe_audio[0]

        y, sr = librosa.load(exemple, sr=16000)
        duration = librosa.get_duration(y=y, sr=sr)

        l1_match = re.search(l1_regex, audio)

        if not l1_match:
            audio_no_trobat.append(audio)
            continue

        l1 = obtenir_llengua(l1_match, map_llengua)

        dislexia = 1 if parts[0] == "d" else 0

        results.append({
            "arxiu": exemple,
            "duracio": duration,
            "lm": l1,
            "curs": curs,
            "dislexia": dislexia
        })

    df_plot = pd.DataFrame(results)

    return df_plot, audio_no_trobat