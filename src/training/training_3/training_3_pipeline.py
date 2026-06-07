from src.training.split import split
from src.training.trainer import Training
from src.training.models import get_models, get_scoring
from src.training.pipelines.approach_3 import get_pipeline_approach3
from src.training.params_grid import get_param_grids
from sklearn.model_selection import RepeatedStratifiedKFold
from training.pipelines.approach_3 import get_pipeline_approach3


def run_training3(df):

    columns_no_train=["arxiu","dislexia","stratification"]

    X_train3, X_test3, y_train3, y_test3 = split(df, columns_no_train)
    
    pca_cols=[col for col in df.columns if "emb" in col]

    categorical_cols = [
        col for col in X_train3.columns if "lm_" in col
    ]
    categorical_index = [X_train3.columns.get_loc(col) for col in categorical_cols]


    scale_cols= ["confianca_mitjana", "confianca_std", "ritme_std", "silencis",
    "cer", "wer", "velocitat", "ratio"]

    dict_models = get_models()
    model = dict_models["rf"]
    scoring = get_scoring()
    params_grid=get_param_grids()
    cv= RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    pipeline= get_pipeline_approach3(model, categorical_index, pca_cols, scale_cols)
    training= Training(X_train3, y_train3, X_test3, y_test3)
    entrenament=training.entrenament(pipeline,dict_models,scoring,params_grid,cv)
    training.print_results()
    training.print_results_profund()
    training.plot_metrics()




