from tensorflow import keras

def get_model():
    # Define the model using Keras.
    mbd_model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
    ])
    return mbd_model
