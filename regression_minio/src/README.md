# Regression with Kubeflow and Minio Pipeline

This repository contains a Kubeflow Pipeline for training a model for microbusiness forecasting using the Kubeflow Pipelines SDK V2. It demonstrates the usage of Containerized Python Components for better production practices.

## Getting Started

Follow these steps to run the code and set up the pipeline.

### Prerequisites

- You need a container runtime (e.g., Docker) installed on your machine.
- You should have a container registry like DockerHub set up to push container images.
- You need a kubeflow v2 setup on your kubernetes environment. I ran this code on Minikube environment.

### Steps

1. Clone this repository to your local machine.

2. Navigate to the repository directory:

   ```bash
   cd regression_minio
   ```
3. Download `train.csv` file from https://www.kaggle.com/competitions/godaddy-microbusiness-density-forecasting/data?select=train.csv and put it in the `./regression_minio/src` path.

4. Build and push the image:

    ```bash
    kfp component build ./src --component-filepattern model_training_pipeline.py --push-image
    ```

5. Run the pipeline:
    ```bash
    python src/model_training_pipeline.py
    ```

6. Access the pipeline UI and view the pipelines. You can use port forwarding to view it in local computer.
    ```bash
    kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80
    ```
*NOTE:* When setting the image-name for the `target_image` in each component definition, make sure you make it consistent with your container runtime environment.

### Acknowledgments
This codebase is a replication of https://github.com/minio/blog-assets/tree/main/kfp-training-pipeline/src?ref=blog.min.io, where I added some data utilities.

## `HAPPY CONTINUOUS LEARNING!! :smile:` 
