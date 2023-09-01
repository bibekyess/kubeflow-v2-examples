# Fashion MNIST Kubeflow Pipeline

This repository contains a Kubeflow Pipeline for training a model on the Fashion MNIST dataset using the Kubeflow Pipelines SDK V2. It demonstrates the usage of Containerized Python Components for better production practices.

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
   cd fashion-mnist-kubeflow-pipeline
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

*NOTE:* When setting the image-name for the `target_image` in each component definition, make sure you make it consistent with your container runtime environment.
### Acknowledgments
This codebase is inspired by https://github.com/manceps/fashion-mnist-kfp-lab, where the original code is written in Kubeflow SDK V1 semantics and uses Lightweight Python Components. I've converted it into SDK V2 using Containerized Python Components.

## `HAPPY CONTINUOUS LEARNING!! :smile:` 
