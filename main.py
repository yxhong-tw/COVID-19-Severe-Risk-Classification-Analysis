import os

import matplotlib.pyplot as plt
import pandas as pd

from argparse import ArgumentParser
from pydotplus import graph_from_dot_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from typing import Any


def train(model_name: str, train_x: pd.DataFrame, train_y: pd.Series) -> Any:
    model = None

    if model_name == "decision_tree":
        model = DecisionTreeClassifier()
    elif model_name == "random_forest":
        model = RandomForestClassifier()
    else:
        raise ValueError("The model name is invalid.")

    model = model.fit(X=train_x, y=train_y)

    return model


def save(data_name: str, model: Any, model_name: str, output_directory: str,
         test_x: pd.DataFrame, test_y: pd.Series):
    labels = model.classes_
    prediction = model.predict(X=test_x)

    # Save comfusion matrix
    cm = confusion_matrix(y_true=test_y, y_pred=prediction, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    plt.title(label=f"{model_name}'s Confusion Matrix ({data_name})")
    image_path = os.path.join(
        output_directory, f"{model_name}_confusion_matrix_{data_name}.png")
    plt.savefig(image_path)

    # Save classification report
    report_path = os.path.join(
        output_directory,
        f"{model_name}_classification_report_{data_name}.txt")
    with open(file=report_path, mode="w", encoding="UTF-8") as report:
        report.write(f"{model_name}'s Classification Report ({data_name}):\n")

        model_report = classification_report(y_true=test_y,
                                             y_pred=prediction,
                                             labels=[0, 1])
        report.write(model_report)

        report.close()

    # Save decision tree
    if model_name == "decision_tree":
        features_name = test_x.columns.values.tolist()
        dot_data = export_graphviz(decision_tree=model,
                                   feature_names=features_name,
                                   filled=True,
                                   rounded=True)

        image_path = os.path.join(output_directory,
                                  f"{model_name}_{data_name}.png")
        graph_from_dot_data(data=dot_data).write_png(image_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-dp",
                        "--data_path",
                        type=str,
                        required=True,
                        help="Path to the data.")
    parser.add_argument("-mn",
                        "--model_name",
                        type=str,
                        choices=["decision_tree", "random_forest"],
                        required=True,
                        help="Model name.")

    args = parser.parse_args()
    data_path = args.data_path
    model_name = args.model_name

    output_directory = "outputs"
    os.makedirs(name=output_directory, exist_ok=True)

    data = pd.read_csv(filepath_or_buffer=data_path)
    train_x, test_x, train_y, test_y = train_test_split(
        data.drop(columns=["IS_HIGH_RISK_FOR_SEVERE_COVID_19"]),
        data["IS_HIGH_RISK_FOR_SEVERE_COVID_19"],
        test_size=0.2)

    model = train(model_name=model_name, train_x=train_x, train_y=train_y)
    data_name = os.path.basename(data_path).split(".")[0]
    save(data_name=data_name,
         model=model,
         model_name=model_name,
         output_directory=output_directory,
         test_x=test_x,
         test_y=test_y)
