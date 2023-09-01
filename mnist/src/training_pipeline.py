import logging
from kfp import dsl
from kfp.dsl import Dataset, Model, Markdown, Input, Output
from kfp.client import Client
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd


import data_utilities as du
import model_utilities as mu

kfp_endpoint = 'http://localhost:8080'

@dsl.component(base_image='tensorflow/tensorflow:latest',
               target_image='bibekyess/fashion_mnist:v1.8',
               packages_to_install=['pandas==2.0.3'])
def train_op(model: Output[Model], test_X: Output[Dataset],
             test_y: Output[Dataset], test_results: Output[Markdown]) -> None:
    
    logger = logging.getLogger('kfp_logger')
    logger.setLevel(logging.INFO)

    train_images, train_labels, test_images, test_labels = du.load_data()

    train_images = du.preprocess_images(train_images)
    test_images = du.preprocess_images(test_images)

    # np.save(test_X.path, test_images)
    # np.save(test_y.path, test_labels)
    # test_images is a 3-dimensional array (10000, 28, 28)
    reshaped_images = test_images.reshape((test_images.shape[0], -1))  # Reshape to (10000, 784)
    test_images_df = pd.DataFrame(reshaped_images)
    test_labels_df = pd.DataFrame(test_labels)
    test_images_df.to_csv(test_X.path, index=False)
    test_labels_df.to_csv(test_y.path, index=False)    

    mbd_model = mu.get_model()

    mbd_model.compile(optimizer='adam',
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    mbd_model.fit(train_images, train_labels, epochs=10)
    test_loss, test_acc = mbd_model.evaluate(test_images,  test_labels, verbose=2)

    display_info = 'Testing loss: **' + str(test_loss) + '**'
    logger.info(display_info)

    with open(test_results.path, 'w') as f:
        f.write('Testing accuracy: **' + str(test_acc) + '**')

    mbd_model.save(model.path)

    logger.info("Training succeded succesfully")


@dsl.component(base_image='tensorflow/tensorflow:latest',
               target_image='bibekyess/fashion_mnist:v1.8',
               packages_to_install=['pandas==2.0.3'])
def predict_op(test_X: Input[Dataset], test_y: Input[Dataset], 
               model: Input[Model], image_number: int, predict_results: Output[Markdown]):
    logger = logging.getLogger('kfp_logger')
    logger.setLevel(logging.INFO)

    mbd_model = keras.models.load_model(model.path)

    test_images_df = pd.read_csv(test_X.path)
    test_labels_df = pd.read_csv(test_y.path)

    test_images = test_images_df.to_numpy()
    original_shape = (test_images.shape[0], 28, 28)
    test_images = test_images.reshape(original_shape)
    test_labels = list(test_labels_df.to_numpy())

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # Define a Softmax layer to define outputs as probabilities
    probability_model = tf.keras.Sequential([mbd_model, 
                                            tf.keras.layers.Softmax()])

    # Grab an image from the test dataset.
    img = test_images[image_number]

    # Add the image to a batch where it is the only member.
    img = (np.expand_dims(img,0))

    # Predict the label of the image.
    predictions = probability_model.predict(img)

    # Take the prediction with the highest probability
    prediction = np.argmax(predictions[0])

    # Retrieve the true label of the image from the test labels.
    true_label = test_labels[image_number]
    
    class_prediction = class_names[prediction]
    confidence = 100*np.max(predictions)
    logger.info(true_label)
    logger.info(type(true_label))
    actual = class_names[int(true_label)]

    with open(predict_results.path, 'w') as result:
        result.write(" Prediction: {} | Confidence: {:2.0f}% | Actual: {}.<br>".format(class_prediction,
                                                                        confidence,
                                                                        actual))
    logger.info("Prediction succeded succesfully")

@dsl.pipeline(
   name='mnist-ipeline',
   description='A toy pipeline that performs mnist model training and prediction.'
)
def mnist_container_pipeline(image_number: int) -> Markdown:
    mnist_train_comp = train_op()
    mnist_pred_comp = predict_op(test_X = mnist_train_comp.outputs['test_X'],
                                 test_y = mnist_train_comp.outputs['test_y'],
                                 model = mnist_train_comp.outputs['model'],
                                 image_number = image_number)
    return mnist_pred_comp.outputs['predict_results']
    
def start_pipeline_run():
    client = Client(host=kfp_endpoint)
    run = client.create_run_from_pipeline_func(mnist_container_pipeline, 
                                          experiment_name = 'Convert MNIST Fashion-v1 to v2',
                                          enable_caching = False,
                                          arguments={
                                              'image_number': 0
                                          })
    url = f'{kfp_endpoint}/#/runs/details/{run.run_id}'
    print(url)


if __name__ == '__main__':
    start_pipeline_run()
