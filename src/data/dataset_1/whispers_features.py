import numpy as np
import librosa


def extract_whisper_features(model, audio_path, output):
    y, sr = librosa.load(audio_path, sr=16000)

    output = model.transcribe(y, task="transcribe", fp16=False)

    logprobs = [seg["avg_logprob"] for seg in output["segments"]]
    confianca_mitjana = np.mean(logprobs)
    confianca_std = np.std(logprobs)

    ritmes_segment = [
        seg["end"] - seg["start"]
        for seg in output["segments"]
    ]

    inici = output["segments"][0]["start"]
    final = output["segments"][-1]["end"]
    duracio = final - inici

    ritme_std = np.std(ritmes_segment)
    temps_parla = np.sum(ritmes_segment)
    silencis = (duracio - temps_parla) / duracio

    return {
        "text": output["text"].strip(),
        "confianca_mitjana": confianca_mitjana,
        "confianca_std": confianca_std,
        "duracio_parla": duracio,
        "ritme_std": ritme_std,
        "silencis": silencis,
    }
