import tensorflow as tf
from tensorflow.keras import layers, models

def compare_optimizers(model_fn, x_train, y_train, x_test, y_test, optimizers=['adam', 'sgd'], epochs=5):
    results = {}
    for opt_name in optimizers:
        print(f"Training with optimizer: {opt_name}")
        model = model_fn()
        model.compile(optimizer=opt_name, loss='categorical_crossentropy', metrics=['accuracy'])
        history = model.fit(x_train, y_train, epochs=epochs, validation_split=0.1, verbose=0)
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        results[opt_name] = {'history': history.history, 'accuracy': test_acc}
    return results

def compare_initializations(x_train, y_train, x_test, y_test, inits=['glorot_uniform', 'he_normal'], epochs=5):
    results = {}
    for init in inits:
        print(f"Training with initialization: {init}")
        model = models.Sequential([
            layers.Flatten(input_shape=(32, 32, 3)),
            layers.Dense(128, activation='relu', kernel_initializer=init),
            layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        history = model.fit(x_train, y_train, epochs=epochs, validation_split=0.1, verbose=0)
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        results[init] = {'history': history.history, 'accuracy': test_acc}
    return results
