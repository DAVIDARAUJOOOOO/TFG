import whisper
import torch
import numpy as np

def get_embedding(model, audio_chunk):
    audio_chunk = whisper.pad_or_trim(audio_chunk)

    mel = whisper.log_mel_spectrogram(audio_chunk).to(model.device).unsqueeze(0)

    with torch.no_grad():
        emb = model.encoder(mel)

    emb = emb.squeeze(0).cpu().numpy()
    mean=emb.mean(axis=0)
    std=emb.std(axis=0)

    res=np.concatenate([mean,std])

    # 1024 dim
    return res