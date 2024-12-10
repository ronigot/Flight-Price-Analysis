# Flight-Price-Analysis
**Tabular Data Science Project**

**Submitting Students:** Roni Gotlib & Hallel Weinberg.
## Overview
We worked with a [dataset containing flight data](data/flight_data.csv). We trained a basic pipeline model that predicts flight prices.

You can see our work in [the notebook](TDS__Project.ipynb).

![plot](https://github.com/user-attachments/assets/bebc96b5-eede-48c6-bd0f-8716458bd2ff)

## Dataset
https://www.kaggle.com/datasets/jillanisofttech/flight-price-prediction-dataset

Please note that we have made a slight change to the original Kaggle dataset. In the Arrival_Time column of the date, the year of departure is 2024. This is compared to the year that appears in the Date_of_Journey column (2023). Therefore, we chose to remove all dates from the Arrival_Time column and leave only the time. 

Our consideration for acting this way was that the data already contains the departure date, the flight duration and the arrival time, from which, if necessary, the arrival date can be calculated manually.

[This is our dataset after the changes.](data/flight_data.csv)

![image](https://github.com/user-attachments/assets/6b999459-30d8-4c75-9de0-00d6d5e655d5)
