# Flight-Price-Analysis
**Tabular Data Science Project**

**Submitting Students:** Roni Gotlib & Hallel Weinberg.
## Overview
We worked with a [dataset containing flight data](data/flight_data.csv). We trained a basic pipeline model that predicts flight prices.

### Part 1
You can find our work in [the notebook](first_part.ipynb).
* The partitioned data includes [X_train](basic_model/X_train_basic.csv), [X_test](basic_model/X_test_basic.csv), [y_train](basic_model/y_train_basic.csv) and [y_test](basic_model/y_test_basic.csv).
* The model is available in [this file](basic_model/basic_model.json).

![plot](https://github.com/user-attachments/assets/bebc96b5-eede-48c6-bd0f-8716458bd2ff)

### Part 2
You can find our work in [the notebook](second_part.ipynb).

![plot2](https://github.com/user-attachments/assets/38804b62-b3cd-4e49-9d80-c9335c98045d)

### Basic Model vs Improvement Model

![plot3](https://github.com/user-attachments/assets/e52bbcd1-3e3c-46cc-b0ee-3b1250796339)

## Dataset
https://www.kaggle.com/datasets/jillanisofttech/flight-price-prediction-dataset

Please note that we have made a slight change to the original Kaggle dataset. In the Arrival_Time column of the date, the year of departure is 2024. This is compared to the year that appears in the Date_of_Journey column (2023). Therefore, we chose to remove all dates from the Arrival_Time column and leave only the time. 

Our consideration for acting this way was that the data already contains the departure date, the flight duration and the arrival time, from which, if necessary, the arrival date can be calculated manually.

This is [our dataset](data/flight_data.csv) after the changes.

![image](https://github.com/user-attachments/assets/6b999459-30d8-4c75-9de0-00d6d5e655d5)
