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
│   ├── feature_transformation_recommender.py    # Core recommender component
│   └── basic_evaluator.py                       # Evaluation metrics component
│
├── notebooks/                # Jupyter notebooks for demonstration and experiments
│   ├── demo.ipynb            # Usage demonstration of the system
│   └── experiments.ipynb     # System evaluation and performance comparisons
│
├── report.pdf                # Detailed project report with methodology and results
├── README.md                 # This file
└── requirements.txt          # Required dependencies
```

## Requirements
To install the required dependencies:
```bash
pip install -r requirements.txt
```

# Feature Transformation Recommender

## Overview
This project is a tool designed to suggest feature transformations in order to improve model performance.

## Project Structure

- [Report](report.pdf)
- [requirements](requirements.txt)
- **Source Code**:
  - [feature_transformation_recommender.py](./src/feature_transformation_recommender.py): Implements the core functionality of the feature transformation recommender system. The recommender system handles both regression and classification tasks.
  - [basic_evaluator.py](./src/basic_evaluator.py): Implements a basic evaluator class that compares feature distributions before and after transformation.
  
- **Notebooks**:
  - [Feature Transformation Demo](./notebooks/feature_transformation_demo.ipynb): A demo notebook showing an example of how to use the feature transformation recommender system.
  - [Feature Recommender Experiments](./notebooks/feature_recommender_experiments.ipynb): A comparison of the system’s performance on four different datasets.



