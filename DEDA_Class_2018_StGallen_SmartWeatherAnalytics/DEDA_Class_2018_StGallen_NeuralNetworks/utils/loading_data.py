#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
# suppress the scientific notation when printing numpy arrays
np.set_printoptions(suppress=True)


def dateparse(x):
    return pd.datetime.strptime(x, '%d/%m/%Y')


def load_data(stock_path, weather_path):
    stocks = pd.read_csv(
        stock_path,
        sep=';',
        parse_dates=['Date'],
        date_parser=dateparse,
        decimal=',')

    # data cleansing
    stocks['Index'] = [
        name.replace(".", "") for name in stocks['Index'].values
    ]
    data_per_index = {}

    # # Loading the weather data

    weather_per_city = pd.read_csv(
        weather_path,
        sep=';',
        parse_dates=['Date'],
        date_parser=dateparse,
        decimal=',')

    cities = pd.DataFrame({
        'City': [
            "New York", "Boston", "San Francisco", "Chicago", "London",
            "Zurich", "Tokyo"
        ],
        'Country': ["USA", "USA", "USA", "USA", "UK", "Switzerland", "Japan"]
    })

    weather_per_city = pd.merge(weather_per_city, cities, on="City")
    # look at how many NaNs we have
    # df.isna().sum()
    # drop NaNs
    weather_per_city.dropna(inplace=True)

    for stock_index in stocks.Index.unique():
        data_per_index[stock_index] = pd.merge(
            stocks[stocks["Index"] == stock_index],
            weather_per_city,
            on="Date")

    # show all cities
    print([city for city in weather_per_city['City'].unique()])
    # assumes (nrows x ncols) episodes
    return data_per_index, weather_per_city


def load_data_OLS(stock_path, weather_path, stock_index):
    # # Loading the data

    # Stocks
    stocks = pd.read_csv(stock_path, sep=';', decimal=',')

    # stocks date format: 29/10/2018
    stocks['Date'] = pd.to_datetime(stocks['Date'], format='%d/%m/%Y')
    stocks.set_index('Date', inplace=True)

    # converting prices to floats
    stocks['Price Close'] = [float(price) for price in stocks['Price Close']]

    # data cleansing
    stocks['Index'] = [
        name.replace(".", "") for name in stocks['Index'].values
    ]

    # selecting a specific index
    stock_index = stocks[stocks['Index'] == stock_index]

    # Weather
    weather = pd.read_csv(weather_path, sep=';', decimal=',')

    # weather date format: 29/10/2018
    weather['Date'] = pd.to_datetime(weather['Date'], format='%d/%m/%Y')
    weather.set_index('Date', inplace=True)
    weather.head()

    # drop NaNs
    weather.dropna(inplace=True)

    return stock_index, weather


def main():
    pass


if __name__ == "__main__":
    main()
