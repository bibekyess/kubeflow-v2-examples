import pandas as pd
from sklearn.datasets import load_iris


def load_iris_data():
    iris = load_iris()
    # dir(iris)
    data = pd.DataFrame(iris['data'], columns=iris['feature_names'])
    target = pd.DataFrame(iris['target'], columns=['target'])
    target_names = iris['target_names']
    label_map = {0: target_names[0], 1: target_names[1], 2: target_names[2]}
    return data, target['target'].map(label_map)
