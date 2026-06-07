import tensorflow as tf


def tune_mlp(configs, X_train, y_train, X_val, y_val, build_model):

    results = []

    for config in configs:

        print(f"Config: {config}")

        model = build_model(
            input_shape=X_train.shape[1],
            learning_rate=config["lr"]
        )

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_auc_pr',
            patience=20,
            mode='max',
            restore_best_weights=True
        )

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=200,
            batch_size=config["batch"],
            callbacks=[early_stop],
            verbose=0
        )

        best_val = max(history.history["val_auc_pr"])

        results.append({
            "config": config,
            "score": best_val,
            "model": model
        })

        print(f"AUC-PR: {best_val:.4f}")

    return results