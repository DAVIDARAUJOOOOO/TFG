from sklearn.metrics import classification_report

def evaluate_by_audio(df_2, X_test_emb, y_pred_emb, y_test_emb, y_test_emb_probs):
    df_pred = df_2.loc[X_test_emb.index]
    df_pred["pred"] = y_pred_emb
    df_pred["test"] = y_test_emb
    df_pred["probs"] = y_test_emb_probs

    y_pred_audio = (
        df_pred
        .groupby("arxiu")["pred"]
        .agg(lambda x: x.value_counts().idxmax())
    )

    y_test_audio = (
        df_pred
        .groupby("arxiu")["test"]
        .agg(lambda x: x.value_counts().idxmax())
    )

    y_probs_audio = (
        df_pred
        .groupby("arxiu")["probs"]
        .mean()
    )

    y_pred_audio_default = y_pred_audio.sort_index()
    y_test_audio = y_test_audio.sort_index()
    y_probs_audio = y_probs_audio.sort_index()
    y_pred_audio = (y_probs_audio >= 0.51).astype(int)

    print(sum(x for x in y_pred_audio if x == 1))
    print(classification_report(y_test_audio, y_pred_audio))

    return y_pred_audio, y_test_audio, y_probs_audio, y_pred_audio_default