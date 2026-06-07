from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTENC


def get_pipeline_approach1(model, categorical_index, columns_to_normalize_not_emb):

    preprocessor = ColumnTransformer(
        transformers=[
            ("scaler", StandardScaler(), columns_to_normalize_not_emb)
        ],
        remainder="passthrough"
    )

    pipeline = Pipeline([
        ('smote', SMOTENC(categorical_features=categorical_index, random_state=42)),
        ('prep', preprocessor),
        ('model', model)
    ])

    return pipeline