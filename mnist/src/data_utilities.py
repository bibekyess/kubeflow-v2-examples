from tensorflow import keras
import numpy as np

def load_data():
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    
    return train_images, train_labels, test_images, test_labels

def preprocess_images(image_data: np.ndarray) -> np.ndarray:
    return image_data / 255.0
