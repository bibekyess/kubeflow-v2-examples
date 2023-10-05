import logging
from kfp import dsl
from kfp.dsl import Dataset, Model, Input, Output
from kfp.client import Client

import dill
import pandas as pd
from sklearn.svm import SVC

import mlflow
from mlflow.models.signature import infer_signature
from mlflow.utils.environment import _mlflow_conda_env

import data_utilities as du

kfp_endpoint = 'http://localhost:8082'

@dsl.component(base_image='python:3.11.3-slim',
               target_image='bibekyess/iris:v2.1',
               packages_to_install=['pandas', 'scikit-learn', 'dill', 'mlflow==2.6.0'])
def train_op(kernel: str,
             model: Output[Model],
             input_example: Output[Dataset],
             signature: Output[Dataset],
             conda_env: Output[Dataset],
             ) -> None:
    logger = logging.getLogger('kfp_logger')
    logger.setLevel(logging.INFO)

    train_data, train_target = du.load_iris_data()

    train_data = train_data.dropna(axis="columns")

    clf = SVC(kernel=kernel)
    clf.fit(train_data, train_target)

    with open(model.path, mode='wb') as file_writer:
        dill.dump(clf, file_writer)

    input_example_val = train_data.sample(1)
    with open(input_example.path, 'wb') as file_writer:
        dill.dump(input_example_val, file_writer)
    
    signature_val = infer_signature(train_data, clf.predict(train_data))
    with open(signature.path, 'wb') as file_writer:
        dill.dump(signature_val, file_writer)

    conda_env_val = _mlflow_conda_env(additional_pip_deps=['dill', 'pandas', 'scikit-learn'])
    with open(conda_env.path, 'wb') as file_writer:
        dill.dump(conda_env_val, file_writer)

    logger.info("Training succeded succesfully!!")

@dsl.component(base_image='python:3.11.3-slim',
               target_image='bibekyess/iris:v2.1',
               packages_to_install=['pandas', 'scikit-learn', 'dill', 'mlflow==2.6.0', 'boto3'])
def upload_sklearn_model_to_mlflow_op(
    model: Input[Model],
    input_example: Input[Dataset],
    signature: Input[Dataset],
    conda_env: Input[Dataset]
)-> None:
    import os

    logger=logging.getLogger('kfp_logger')
    logger.setLevel(logging.INFO)

    with open(model.path, mode='rb') as file_reader:
        clf = dill.load(file_reader)

    with open(input_example.path, 'rb') as file_reader:
        input_example_val = dill.load(file_reader)

    with open(signature.path, 'rb') as file_reader:
        signature_val = dill.load(file_reader)

    with open(conda_env.path, 'rb') as file_reader:
        conda_env_val = dill.load(file_reader)

    mlflow.set_tracking_uri('http://192.168.0.29:5000')

    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://192.168.0.29:9000"
    os.environ["AWS_ACCESS_KEY_ID"] = "FIX_ME" # FIXME
    os.environ["AWS_SECRET_ACCESS_KEY"] = "FIX_ME" # FIXME
    os.environ["MLFLOW_S3_IGNORE_TLS"] = "true"

    mlflow.set_experiment('iris')

    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(
            sk_model=clf,
            artifact_path="sklearn-model",
            signature=signature_val,
            conda_env=conda_env_val,
            input_example=input_example_val,
        )

    logger.info('Upload succeeded successfully!!')

@dsl.pipeline(
    name='iris-pipeline',
    description='A toy pipeline that performs iris model training and prediction'
)
def iris_container_pipeline(kernel: str='rbf')-> None:
    iris_train_comp = train_op(kernel=kernel)
    _ = upload_sklearn_model_to_mlflow_op(
        model= iris_train_comp.outputs['model'],
        input_example=iris_train_comp.outputs['input_example'],
        signature=iris_train_comp.outputs['signature'],
        conda_env=iris_train_comp.outputs['conda_env']
    )

def start_pipeline_run():
    # client = Client(host=kfp_endpoint)
    import requests

    USERNAME = "user@example.com"
    PASSWORD = "12341234" 
    NAMESPACE = "kubeflow-user-example-com"
    HOST = "http://127.0.0.1:8082" # your istio-ingressgateway pod ip:8080

    session = requests.Session()
    response = session.get(HOST)

    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
    }

    data = {"login": USERNAME, "password": PASSWORD}
    session.post(response.url, headers=headers, data=data)
    session_cookie = session.cookies.get_dict()["authservice_session"]

    client = Client(
        host=f"{HOST}/pipeline",
        namespace=f"{NAMESPACE}",
        cookies=f"authservice_session={session_cookie}",
    )

    print(client.list_experiments())

    run = client.create_run_from_pipeline_func(
        iris_container_pipeline,
        experiment_name = 'Playing with using mlflow and kubeflow together',
        enable_caching=False,
    )
    url = f'{kfp_endpoint}/#/runs/details/{run.run_id}'
    print(url)

if __name__ == '__main__':
    start_pipeline_run()
