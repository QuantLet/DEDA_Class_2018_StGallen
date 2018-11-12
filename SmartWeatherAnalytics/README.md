# Smart Weather Analytics
Apelt, Gosselke, Sommer & Tuchschmid

## Installation instructions
(these steps have been tested on MacOS. For Linux/Windows, please adjust accordingly)

Use your terminal/command line to  navigate to the preferred location on your drive and execute the following command to download the whole project:
```bash
git clone https://github.com/Sommer1872/smart_weather_analytics.git
```

We highly encurage the use of anaconda as a package manager. If you have anaconda, you can simply `cd smart_weather_analytics` and then run:
```bash
conda env create -f requirements.yml
```
This creates a separate environment named `SWA` and automatically installs all the necessary packages. Before running the python scripts, you should run `conda activate SWA` in order to be able to use the installed packages.

## Run instructions
There are two main approaches used:
- You can run  [train_neural_networks.py](train_neural_networks.py) to train Neural Networks (fully connected and LSTM variants) that try to predict stock movements from weather data. 
- [run_OLS_ARMA.py](run_OLS_ARMA.py) runs different regressions to detect a relationship between weather data and stock movements/volatility.

## Data
Currently, the analyses are done based on a small sample (October 2018) of the data. This will not deliver any meaningful results.

Unfortunately, this code uses proprietary data that we are not allowed to make widely accessible. If interested, please contact the owner of this repo directly.

Otherwise, you can retrieve the data yourself from your preferred provider. 

You are going to need 2 files:

`StockIndices.csv` has 3 columns:
```python
Index;Date;Price Close
```

and `Weather_ALL.csv` should look like this (8 columns):
```python
Date;City;Mean Temperature Actual;Low Temperature Actual;High Temperature Actual;Precipitation Actual;Wind Speed Actual;Relative Humidity Actual
```