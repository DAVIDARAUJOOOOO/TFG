from imblearn.over_sampling import SMOTENC


def apply_smotenc(X_train, y_train, categorical_cols):
    categorical_index = [
        X_train.columns.get_loc(col)
        for col in categorical_cols
    ]

    smote = SMOTENC(
        categorical_features=categorical_index,
        random_state=42
    )

    X_res, y_res = smote.fit_resample(X_train, y_train)

    return X_res, y_res