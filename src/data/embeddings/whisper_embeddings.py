import numpy as np
import whisper
import torch

def complete_embedding(ruta_audio, model):
  audio = whisper.load_audio(ruta_audio)

  all_embeddings = []
  all_weights = []

  chunk = 16000 * 15  #  chunks de 15s

  for i in range(0, len(audio), chunk):

    chunk_audio = audio[i:i+chunk]
    real_length = len(chunk_audio)

    chunk_audio = whisper.pad_or_trim(chunk_audio)

    mel = whisper.log_mel_spectrogram(chunk_audio).to(model.device).unsqueeze(0)

    with torch.no_grad():
        emb = model.encoder(mel)

    emb = emb.squeeze(0).cpu().numpy()

    mean = emb.mean(axis=0)
    std = emb.std(axis=0)  # variació de l'embedding

    #1024 dimensions
    chunk_embedding = np.concatenate([mean,std])

    all_embeddings.append(chunk_embedding)
    all_weights.append(real_length)

  final_embedding = np.average(all_embeddings, axis=0, weights=all_weights)

  return final_embedding