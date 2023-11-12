import os

import pandas as pd

from argparse import ArgumentParser
from copy import deepcopy
from numpy import clip
from numpy.random import choice, normal
from typing import Any, Dict, List

DATA = {
    # With Multinomial Distribution
    "CHRONIC_NUMBER": range(0, 8),
    "GENDER": [0, 1],
    "IS_PREGNANCY": [0, 1],
    "IS_SMOKE": [0, 1],
    "REGULAR_LIFE_NUMBER": range(0, 6),
    "REINFECTION_NUMBER": range(0, 5),
    "VACCINE_NUMBER": range(0, 6),

    # With Normal Distribution
    "AGE": {
        "mean": 43.0,
        "std": 5.0
    },
    "BMI": {
        "mean": 21,
        "std": 0.5
    },
    "ECONOMICS": {
        "mean": 3.0,
        "std": 1.0
    },
}

# With Multinomial Distribution
PROBABILITIES = {
    "CHRONIC_NUMBER": [0.5, 0.2, 0.1, 0.08, 0.06, 0.03, 0.02, 0.01],
    "GENDER": [0.5, 0.5],
    "IS_PREGNANCY": [0.9, 0.1],
    "IS_SMOKE": [0.7, 0.3],
    "REGULAR_LIFE_NUMBER": [0.4, 0.2, 0.2, 0.1, 0.05, 0.05],
    "REINFECTION_NUMBER": [0.8, 0.1, 0.05, 0.03, 0.02],
    "VACCINE_NUMBER": [0.1, 0.05, 0.05, 0.05, 0.5, 0.25]
}


def _generate(alter: bool) -> Dict[str, Any]:
    data = deepcopy(DATA)

    data["CHRONIC_NUMBER"] = choice(a=data["CHRONIC_NUMBER"],
                                    p=PROBABILITIES["CHRONIC_NUMBER"])
    data["GENDER"] = choice(a=data["GENDER"], p=PROBABILITIES["GENDER"])
    data["IS_SMOKE"] = choice(a=data["IS_SMOKE"], p=PROBABILITIES["IS_SMOKE"])
    data["REGULAR_LIFE_NUMBER"] = choice(
        a=data["REGULAR_LIFE_NUMBER"], p=PROBABILITIES["REGULAR_LIFE_NUMBER"])
    data["REINFECTION_NUMBER"] = choice(a=data["REINFECTION_NUMBER"],
                                        p=PROBABILITIES["REINFECTION_NUMBER"])
    data["VACCINE_NUMBER"] = choice(a=data["VACCINE_NUMBER"],
                                    p=PROBABILITIES["VACCINE_NUMBER"])

    data["AGE"] = normal(loc=data["AGE"]["mean"], scale=data["AGE"]["std"])
    data["AGE"] = clip(a=data["AGE"], a_min=0, a_max=100)
    data["BMI"] = normal(loc=data["BMI"]["mean"], scale=data["BMI"]["std"])
    data["BMI"] = clip(a=data["BMI"], a_min=15, a_max=40)
    data["ECONOMICS"] = normal(loc=data["ECONOMICS"]["mean"],
                               scale=data["ECONOMICS"]["std"])
    data["ECONOMICS"] = clip(a=data["ECONOMICS"], a_min=1, a_max=5)

    if data["GENDER"] == 0 and data["AGE"] >= 15 and data["AGE"] <= 55:
        data["IS_PREGNANCY"] = choice(a=data["IS_PREGNANCY"],
                                      p=PROBABILITIES["IS_PREGNANCY"])
    else:
        data["IS_PREGNANCY"] = 0

    data[
        "IS_HIGH_RISK_FOR_SEVERE_COVID_19"] = is_high_risk_for_severe_covid_19(
            alter=alter, data=data)

    return data


def generate(alter: bool, data_number: int) -> List:
    data = []

    for _ in range(data_number):
        data.append(_generate(alter=alter))

    return data


def is_high_risk_for_severe_covid_19(alter: bool, data: Dict[str, Any]) -> int:
    condition_1 = is_condition_1(alter=alter, data=data)
    condition_2 = is_condition_2(alter=alter, data=data)
    condition_3 = is_condition_3(alter=alter, data=data)
    condition_4 = data["VACCINE_NUMBER"] == 0 and data[
        "REINFECTION_NUMBER"] >= 1

    if condition_1 or condition_2 or condition_3 or condition_4:
        return 1

    return 0


def is_condition_1(alter: bool, data: Dict[str, Any]) -> int:
    conditions = [
        data["BMI"] >= 30,
        data["CHRONIC_NUMBER"] >= 1,
        data["IS_PREGNANCY"] == 1,
        data["IS_SMOKE"] == 1,
        data["REGULAR_LIFE_NUMBER"] <= 3,
    ]

    condition = (data["AGE"] >= 50) and (data["VACCINE_NUMBER"]
                                         <= 2) and (sum(conditions) >= 3)

    if alter:
        condition = (data["AGE"] >= 50) and (sum(conditions) >= 3)

    if condition:
        return 1

    return 0


def is_condition_2(alter: bool, data: Dict[str, Any]) -> int:
    conditions = [
        data["BMI"] >= 30,
        data["CHRONIC_NUMBER"] >= 2,
        data["IS_PREGNANCY"] == 1,
        data["IS_SMOKE"] == 1,
        data["REGULAR_LIFE_NUMBER"] <= 3,
        data["ECONOMICS"] <= 2,
    ]

    condition = (data["AGE"] > 10
                 and data["AGE"] < 50) and (data["VACCINE_NUMBER"]
                                            <= 2) and (sum(conditions) >= 3)

    if alter:
        condition = (data["AGE"] > 10
                     and data["AGE"] < 50) and (sum(conditions) >= 3)

    if condition:
        return 1

    return 0


def is_condition_3(alter: bool, data: Dict[str, Any]) -> int:
    conditions = [
        data["BMI"] >= 30,
        data["CHRONIC_NUMBER"] >= 1,
        data["REGULAR_LIFE_NUMBER"] <= 3,
        data["ECONOMICS"] <= 2,
    ]

    condition = (data["AGE"]
                 <= 10) and data["VACCINE_NUMBER"] <= 2 and (sum(conditions)
                                                             >= 2)

    if alter:
        condition = (data["AGE"] <= 10) and (sum(conditions) >= 2)

    if condition:
        return 1

    return 0


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-a",
                        "--alter",
                        default=False,
                        type=bool,
                        help="alter data or not")

    args = parser.parse_args()
    alter = args.alter

    data_name = "data.csv" if not alter else "data2.csv"
    data_number = 10000

    output_directory = "inputs"
    os.makedirs(name=output_directory, exist_ok=True)

    output_path = os.path.join(output_directory, data_name)

    data = generate(alter=alter, data_number=data_number)
    data = pd.DataFrame(data=data)
    data.to_csv(path_or_buf=output_path, index=False)

    generation_report_path = os.path.join(output_directory,
                                          "generation_report.txt")
    with open(file=generation_report_path, mode="a",
              encoding="UTF-8") as report:
        report.write(f"{data_name}'s Generation Report:\n")
        report.write(
            f"Number of IS_HIGH_RISK_FOR_SEVERE_COVID_19 == 1: {data['IS_HIGH_RISK_FOR_SEVERE_COVID_19'].sum()}\n"
        )
        report.write(f"Number of data: {len(data)}\n")
        report.write("\n")

        report.close()
