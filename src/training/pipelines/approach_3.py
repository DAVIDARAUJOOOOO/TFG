from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTENC
from sklearn.decomposition import PCA


def get_pipeline_approach3(model, categorical_index, pca_cols, scale_cols):

    preprocessor5 = ColumnTransformer(
        transformers=[
            ("pca_block", Pipeline([
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=30, random_state=42))
            ]), pca_cols),

            ("scaler_only", StandardScaler(), scale_cols)
        ],
        remainder="passthrough"
    )

    pipeline5 = Pipeline([
        ("smote", SMOTENC(
            categorical_features=categorical_index,
            random_state=42
        )),
        ("prep", preprocessor5),
        ("model", model)
    ])
    return pipeline5