import pandas as pd


def expand_embeddings(df):
    emb_df = pd.DataFrame(df["embeddings"].tolist(), index=df.index)
    emb_df.columns = [f"emb_{i}" for i in range(emb_df.shape[1])]
    df = df.drop(columns=["embeddings"])
    df = pd.concat([df, emb_df], axis=1)
    return df
