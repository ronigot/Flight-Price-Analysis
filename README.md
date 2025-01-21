# Flight-Price-Analysis
**Tabular Data Science Project**

**Submitting Students:** Roni Gotlib & Hallel Weinberg.
## Overview
In this project, we worked with a [dataset containing flight data](data/flight_data.csv) to predict flight prices. We trained an initial basic pipeline model and later improved it to achieve better predictions.

### Part 1
You can find our work in [the notebook](first_part.ipynb).
* The partitioned data includes [X_train](basic_model/X_train_basic.csv), [X_test](basic_model/X_test_basic.csv), [y_train](basic_model/y_train_basic.csv) and [y_test](basic_model/y_test_basic.csv).
* The model is available in [this file](basic_model/basic_model.json).

![plot](https://github.com/user-attachments/assets/bebc96b5-eede-48c6-bd0f-8716458bd2ff)

### Part 2
You can find our work in [the notebook](second_part.ipynb).
* Please upload the following files to the notebook: [flight_data.csv](data/flight_data.csv), [basic_model.json](basic_model/basic_model.json), [X_train_basic.csv](basic_model/X_train_basic.csv), [X_test_basic.csv](basic_model/X_test_basic.csv), [y_train_basic.csv](basic_model/y_train_basic.csv) and [y_test_basic.csv](basic_model/y_test_basic.csv).

![plot2](https://github.com/user-attachments/assets/38804b62-b3cd-4e49-9d80-c9335c98045d)

### Basic Model vs Improvement Model

![plot3](https://github.com/user-attachments/assets/e52bbcd1-3e3c-46cc-b0ee-3b1250796339)

## Dataset
The dataset used in this project is from [Kaggle](https://www.kaggle.com/datasets/jillanisofttech/flight-price-prediction-dataset).

Please note that we made a slight change to the [original Kaggle dataset](https://www.kaggle.com/datasets/jillanisofttech/flight-price-prediction-dataset): in the Arrival_Time column, the year of departure is 2024, while the year in the Date_of_Journey column is 2023. As a result, we decided to remove the year from the Arrival_Time column and keep only the time value.

We made this change because the data already contains the departure date, flight duration, and arrival time, which allows us to manually calculate the arrival date if necessary.

This is [our dataset](data/flight_data.csv) after the modification.

![image](https://github.com/user-attachments/assets/6b999459-30d8-4c75-9de0-00d6d5e655d5)
