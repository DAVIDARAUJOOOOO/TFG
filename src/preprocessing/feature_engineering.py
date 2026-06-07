from src.preprocessing.language_encoding import encode_language
from src.preprocessing.embedding_expansion import expand_embeddings
from src.visualization.correlation import correlation_matrix


def build_features(df):
    df = encode_language(df)
    df = expand_embeddings(df)

    return df


def run_correlation_analysis(df):
    features_corr = [
        "confianca_mitjana", "confianca_std", "ritme_std", "silencis",
        "cer", "wer", "velocitat", "ratio", "curs", "lm_", "dislexia"
    ]

    correlation_matrix(df, features_corr)

    features_emb = [col for col in df.columns if "emb_" in col] + ["dislexia"]

    # correlation_matrix(df, features_emb)
