# COVID-19-Severe-Risk-Classification-Analysis
The project is to generate a dataset and use Decision Tree and Random Forest to classify and analyze the severe risk of COVID-19 patients.

If you want to know more about the project, please read the [report.pdf](https://github.com/yxhong-tw/COVID-19-Severe-Risk-Classification-Analysis/blob/main/report.pdf)

## Usage
```
python3 generate.py -a [IS_ALTERING]
```
> Generate the C19SR dataset.  
> [IS_ALTERING] is a boolean value, which can be `True` or `False`.

```
python3 main.py -dp [DATA_PATH] -mn [MODEL_NAME]
```
> Train and test the C19SR dataset with Decision Tree or Random Forest.  
> [DATA_PATH] is the path of the C19SR dataset.  
> [MODEL_NAME] is the name of the model, which can be `decision_tree` or `random_forest`.

## Requirements
```
matplotlib   == 3.7.3
numpy        == 1.24.4
pandas       == 2.0.3
pydotplus    == 2.0.2
scikit-learn == 1.3.2
```
> You can install the requirements by `pip3 install -r requirements.txt`.

## My Environment
- Windows 11 Home
    - Version: 22H2
    - OS build: 22621.2428
- Intel(R) Core(TM) i5-8300H CPU @ 2.30GHz 2.30 GHz
- NVIDIA GeForce GTX 1050
    - Driver Version: 537.42
    - CUDA Version: 12.2

## Common Issues
### graphviz
```
Graphviz's executables are not found.
```
> Solution: `conda install graphviz`  
> Reference: [Graphviz's executables are not found (Python 3.4)](https://stackoverflow.com/questions/28312534/graphvizs-executables-are-not-found-python-3-4)
