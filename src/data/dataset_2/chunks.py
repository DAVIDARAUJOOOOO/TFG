def chunk_audio(audio):
    SR = 16000
    CHUNK = 2 * SR
    chunks = []
    for i in range(0, len(audio), CHUNK):
        frag = audio[i:i+CHUNK]
        if len(frag) < 0.25 * CHUNK:
          continue
        chunks.append(frag)

    return chunks