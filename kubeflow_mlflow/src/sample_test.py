import mlflow
logged_model = 's3://mlflow/1/d3653d7fc44747a98947c126c7a92f57/artifacts/sklearn-model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

data = {
  "columns": [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)"
  ],
  "data": [
    [
      5.9,
      3.2,
      4.8,
      1.8
    ]
  ]
}

# Predict on a Pandas DataFrame.
import pandas as pd
test_df = pd.DataFrame(data["data"], columns=data["columns"])

print(loaded_model.predict(test_df))