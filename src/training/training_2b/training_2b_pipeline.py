from src.training.split import split
from src.training.trainer import Training
from src.training.models import get_models, get_scoring
from src.training.pipelines.approach_2 import get_pipeline_approach2
from src.training.params_grid import get_params_grid
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold

from training.pipelines.approach_2b import get_pipeline_approach2b
from training.training_2b.avaluacio_nen import evaluate_by_audio
from training.training_2b.plotting import plot_audio_results


def run_training2b(df):
    df["stratification"]=df["curs"].astype(str)+"_"+df["label"].astype(str)
    groups=df["arxiu"]

    sfdk=StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(sfdk.split(df, df["stratification"], groups=groups))
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    y_train_emb=train_df["label"]
    y_test_emb=test_df["label"]

    X_train_emb=train_df[[column for column in train_df.columns if "emb_" in column]]
    X_test_emb=test_df[[column for column in test_df.columns if "emb_" in column]]
    groups_train=train_df["arxiu"]


    dict_models = get_models()
    model = dict_models["rf"]
    scoring = get_scoring()
    params_grid=get_params_grid()
    cv_groups = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    pipeline= get_pipeline_approach2b(model)
    training= Training(X_train_emb, y_train_emb, X_test_emb, y_test_emb, groups_train)
    entrenament=training.entrenament(pipeline,dict_models,scoring,params_grid,cv_groups)
    y_pred_emb, y_test_emb_probs=training.print_results()
    training.print_results_profund()
    training.plot_metrics()

    y_pred_audio, y_test_audio, y_probs_audio, y_pred_audio_default=evaluate_by_audio(test_df, X_test_emb, y_pred_emb, y_test_emb, y_test_emb_probs)
    plot_audio_results(y_test_audio, y_probs_audio)






