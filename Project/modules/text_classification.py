import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

def get_imdb_data(max_features=10000, maxlen=200):
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    return (x_train, y_train), (x_test, y_test)

def build_rnn_model(max_features=10000, maxlen=200):
    model = models.Sequential([
        layers.Embedding(max_features, 32, input_length=maxlen),
        layers.SimpleRNN(32, kernel_initializer='glorot_uniform'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

def build_lstm_model(max_features=10000, maxlen=200):
    model = models.Sequential([
        layers.Embedding(max_features, 32, input_length=maxlen),
        layers.LSTM(32, kernel_initializer='glorot_uniform', return_sequences=False),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

def build_gru_model(max_features=10000, maxlen=200):
    model = models.Sequential([
        layers.Embedding(max_features, 32, input_length=maxlen),
        layers.GRU(32, kernel_initializer='glorot_uniform'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

def train_and_evaluate_text(model, x_train, y_train, x_test, y_test, epochs=5, optimizer='adam'):
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=1)
    ]
    
    history = model.fit(x_train, y_train, epochs=epochs, 
                        validation_split=0.2, 
                        callbacks=callbacks,
                        verbose=1)
    
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    return history, test_acc
