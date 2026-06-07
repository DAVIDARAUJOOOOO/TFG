import tensorflow as tf
from tensorflow.keras import layers, models

def crear_mlp_robusta(input_shape, learning_rate=1e-3):
    model = models.Sequential([
        layers.Input(shape=(input_shape,)),

        layers.Dense(64, activation='relu',
                     kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(32, activation='relu',
                     kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        layers.Dropout(0.2),

        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=[
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.AUC(curve='PR', name='auc_pr'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    return model