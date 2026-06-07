from src.training.split import split
from src.training.trainer import Training
from src.training.models import get_models, get_scoring
from src.training.pipelines.approach_2 import get_pipeline_approach2
from src.training.params_grid import get_param_grids
from sklearn.model_selection import RepeatedStratifiedKFold


def run_training2(df):

    columns_no_train=[column for column in df.columns if "emb_" not in column]

    X_train2, X_test2, y_train2, y_test2 = split(df, columns_no_train)

    dict_models = get_models()
    model = dict_models["rf"]
    scoring = get_scoring()
    params_grid=get_param_grids()
    cv= RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    pipeline= get_pipeline_approach2(model)
    training= Training(X_train2, y_train2, X_test2, y_test2)
    entrenament=training.entrenament(pipeline,dict_models,scoring,params_grid,cv)
    training.print_results()
    training.print_results_profund()
    training.plot_metrics()




