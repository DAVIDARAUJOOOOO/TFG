from src.training.training_mlp.mlp_architecture import crear_mlp_robusta
from src.training.training_mlp.mlp_tunning import tune_mlp
from src.training.training_mlp.mlp_preprocessing import prepare_mlp_data
from src.training.training_mlp.mlp_train_final import train_best_mlp
from src.training.training_mlp.mlp_plots import plot_mlp_results
from src.training.split import split
from src.training.training_1.scaling_experiment import normalize_features
from src.training.training_1.smotenc_experiment import apply_smotenc


def run_training_mlp(df):

    columns_no_train = ["arxiu", "dislexia"] + [
        col for col in df.columns if "emb_" in col
    ]

    X_train, X_test, y_train, y_test = split(df, columns_no_train)
    columns_to_normalize_not_emb = [
         "confianca_mitjana", "confianca_std", "ritme_std", "silencis",
    "cer", "wer", "velocitat", "ratio"
    ]

    X_train_norm, X_test_norm, _ = normalize_features(
        X_train, X_test, columns_to_normalize_not_emb
    )

    categorical_cols = [
        col for col in X_train.columns if "lm_" in col
    ]

    X_res, y_res = apply_smotenc(
        X_train_norm,
        y_train,
        categorical_cols
    )


    X_tr, y_tr, X_val, y_val, X_test_scaled, scaler = prepare_mlp_data(
        X_train, y_train, X_test, categorical_cols
    )

    configs = [
        {"lr": 1e-3, "batch": 8},
        {"lr": 1e-4, "batch": 8},
        {"lr": 1e-3, "batch": 16},
    ]

    results = tune_mlp(
        configs,
        X_tr, y_tr,
        X_val, y_val,
        crear_mlp_robusta
    )

    best_config = max(results, key=lambda x: x["score"])["config"]

    model = train_best_mlp(
        best_config,
        X_res,
        y_res,
        crear_mlp_robusta
    )

    y_prob = model.predict(X_test_norm).ravel()
    y_pred = (y_prob > 0.5).astype(int)

    plot_mlp_results(y_test, y_prob, y_pred)