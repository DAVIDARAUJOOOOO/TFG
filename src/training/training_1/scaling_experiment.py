from sklearn.preprocessing import StandardScaler


def normalize_features(X_train, X_test, cols_to_normalize):
    scaler = StandardScaler()

    X_train_norm = X_train.copy()
    X_test_norm = X_test.copy()

    X_train_norm[cols_to_normalize] = scaler.fit_transform(X_train[cols_to_normalize])
    X_test_norm[cols_to_normalize] = scaler.transform(X_test[cols_to_normalize])

    return X_train_norm, X_test_norm, scaler