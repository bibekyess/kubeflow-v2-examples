# IRIS Kubeflow Pipeline with MLflow Model Registry

This repository contains a Kubeflow Pipeline for training a model on the IRIS dataset using the Kubeflow Pipelines SDK V2. One the Kubeflow pipeline components uploads the model in the MLflow model registry.

## Getting Started

Follow these steps to run the code and set up the pipeline.

### Prerequisites

- You need a container runtime (e.g., Docker) and MLflow installed on your machine.
- You should have a container registry like DockerHub set up to push container images.
- You need a kubeflow v2 setup on your kubernetes environment. I ran this code on Minikube environment.

### Steps

1. Clone this repository to your local machine.

2. Navigate to the repository directory:

   ```bash
   cd kubeflow_mlflow
   ```
3. Build and push the image:

    ```bash
    kfp component build ./src --component-filepattern training_pipeline.py --push-image
    ```

4. Run the pipeline:
    ```bash
    python src/training_pipeline.py
    ```

5. Access the pipeline UI and view the pipelines. You can use port forwarding to view it in local computer.
    ```bash
    kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80
    ```

**NOTE:** When setting the image-name for the `target_image` in each component definition, make sure you make it consistent with your container runtime environment.

6. Run `python sample_test_mlflow.py` to perform inference on the model residing in MLflow model registry.


## `HAPPY CONTINUOUS LEARNING!! :smile:` 