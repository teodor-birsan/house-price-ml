import tensorflow as tf
import pandas as pd
import numpy as np
from keras.callbacks import EarlyStopping
import keras

def deep_learning_model(dataset: pd.DataFrame):
    (X_train, X_valid, y_train, y_valid) = dataset[0]
    early_stopping = EarlyStopping(
        min_delta=0.001, # minimium amount of change to count as an improvement
        patience=20, # how many epochs to wait before stopping
        restore_best_weights=True,
    )
    test_data = dataset[1]
    model = keras.Sequential([
        keras.layers.Input(shape=(X_train.shape[1], )),
        keras.layers.Dense(units=512, activation='elu'),
        keras.layers.Dropout(rate=0.6, seed=42),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(units=256, activation='elu'),
        keras.layers.Dropout(rate=0.6, seed=10),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(units=32, activation='elu'),
        keras.layers.Dense(1) # output layer
    ])

    model.compile(
        optimizer='adam',
        loss='mae'
    )
    model.fit(X_train, y_train,
              validation_data=(X_valid, y_valid),
              batch_size=128,
              callbacks=[early_stopping],
              epochs=800)
    
    metrics = model.get_metrics_result()
    mae = metrics['loss']
    predictions = model.predict(test_data)
    predictions = np.reshape(predictions, -1)
    predictions = predictions.tolist()
    print(metrics)
    return mae, predictions