from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.training.training_1.smotenc_experiment import apply_smotenc
import pandas as pd

def prepare_mlp_data(X_train, y_train, X_test, categorical_columns):

    # split interno validation
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.125,
        stratify=y_train,
        random_state=42
    )

    # scaling
    scaler = StandardScaler()

    X_tr = pd.DataFrame(
        scaler.fit_transform(X_tr),
        columns=X_train.columns,
        index=X_tr.index
    )

    X_val = pd.DataFrame(
        scaler.transform(X_val),
        columns=X_train.columns,
        index=X_val.index
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_train.columns,
        index=X_test.index
    )

    # SMOTE
    X_tr_res, y_tr_res = apply_smotenc(X_tr, y_tr, categorical_columns)

    return X_tr_res, y_tr_res, X_val, y_val, X_test_scaled, scaler