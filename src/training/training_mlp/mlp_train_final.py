def train_best_mlp(best_config, X_train, y_train, build_model):

    model = build_model(
        input_shape=X_train.shape[1],
        learning_rate=best_config["lr"]
    )

    model.fit(
        X_train,
        y_train,
        epochs=200,
        batch_size=best_config["batch"],
        verbose=0
    )

    return model