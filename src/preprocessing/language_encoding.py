def determinar_llengua(llengua):
    if "_" in llengua:
        if "latino" in llengua:
            return "castella"
        return "altres"
    else:
        if any(x in llengua for x in ["castella", "argenti", "dominica"]):
            return "castella"
        elif "catala" in llengua:
            return "catala"
        else:
            return "altres"


def encode_language(df):
    df = df.copy()
    df["lm_grup"] = df["lm"].apply(determinar_llengua)
    df["lm_"] = df["lm_grup"].map({
        "castella": 0,
        "catala": 1,
        "altres": 2
    })
    df = df.drop(columns=["lm_grup", "lm"])
    return df
