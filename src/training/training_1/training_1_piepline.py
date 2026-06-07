from src.training.split import split
from src.training.training_1.scaling_experiment import normalize_features
from src.training.training_1.smotenc_experiment import apply_smotenc
from src.visualization.class_balance_plot import plot_class_balance
from src.training.trainer import Training
from src.training.models import get_models, get_scoring
from src.training.pipelines.approach_1 import get_pipeline_approach1
from src.training.params_grid import get_param_grids
from sklearn.model_selection import RepeatedStratifiedKFold


def run_training1(df):
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
    categorical_index = [X_train.columns.get_loc(col) for col in categorical_cols]

    X_res, y_res = apply_smotenc(
        X_train_norm,
        y_train,
        categorical_cols
    )

    plot_class_balance(y_train, y_res)

    dict_models = get_models()
    model = dict_models["rf"]
    scoring = get_scoring()
    params_grid=get_param_grids()
    cv= RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    pipeline= get_pipeline_approach1(model, categorical_index, columns_to_normalize_not_emb)
    training= Training(X_train, y_train, X_test, y_test)
    entrenament=training.entrenament(pipeline,dict_models,scoring,params_grid,cv)
    training.print_results()
    training.print_results_profund()
    training.plot_metrics()




