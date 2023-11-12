# COVID-19-Severe-Risk-Classification-Analysis

## Usage
```
python3 main.py
```
> Train and test the C19SR dataset with Decision Tree and Random Forest.

```
python3 generate.py
```
> Generate the C19SR dataset.

## Requirements
```
matplotlib   == 3.7.3
numpy        == 1.24.4
pandas       == 2.0.3
pydotplus    == 2.0.2
scikit-learn == 1.3.2
```
> You can install the requirements by `pip install -r requirements.txt`.

## Environment
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
