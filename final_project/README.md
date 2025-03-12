# Automated Feature Transformation Recommender

## Overview
This project presents an automated system for optimal feature transformation selection in tabular datasets, focusing on linear models (Linear/Logistic Regression). The system automatically analyzes feature distributions and evaluates a comprehensive set of candidate transformations using cross-validation performance metrics and normality tests.

The feature transformation recommender aims to:
- Reduce manual effort in feature engineering
- Identify non-obvious beneficial transformations
- Provide quantitative justification for transformation choices
- Streamline the preprocessing workflow

## Structure
```bash
final_project/
│
├── src/                      # Source code
│   ├── __init__.py
│   ├── feature_transformation_recommender.py     # Core recommender component
│   └── basic_evaluator.py                        # Evaluation metrics component
│
├── notebooks/                # Jupyter notebooks for demonstration and experiments
│   ├── demo.ipynb                                # Usage demonstration of the system
│   └── feature_recommender_experiments.ipynb     # System evaluation and performance comparisons
│
├── report.pdf                # Detailed project report with methodology and results
├── README.md                 # This file
└── requirements.txt          # Required dependencies
```

## [Requirements](requirements.txt)
To install the required dependencies:
```bash
pip install -r requirements.txt
```

## [Jupyter Notebooks](./notebooks)

* [*Demo Notebook*](notebooks/feature_transformation_demo.ipynb): Demonstrates a step-by-step usage of the recommender system.
* [*Experiments Notebook*](notebooks/feature_recommender_experiments.ipynb): Contains system evaluation and performance comparisons based on multiple datasets.

## [Report](report.pdf)
