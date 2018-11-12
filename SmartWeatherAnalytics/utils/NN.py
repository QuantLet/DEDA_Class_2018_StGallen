# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 14:00:43 2018
"""
from keras.models import Sequential
from keras.layers import Dense, LSTM


def Fully_Connected_OneL(x, y):
    # Hyperparameters
    loss = {}
    for i in range(1, 10):
        number_of_nodes = i * 5
        print("Number of Nodes " + str(number_of_nodes))
        model = Sequential(
            [Dense(number_of_nodes, input_shape=(42, )),
             Dense(1)])
        model.compile(optimizer='rmsprop', loss='mse')
        model.summary()
        history_callback = model.fit(
            x, y, validation_split=0.33, epochs=10, batch_size=128)
        dict_name = "i = " + str(number_of_nodes)
        loss[dict_name] = history_callback.history["loss"]
    return loss


def Fully_Connected_TwoL(x, y):
    # Hyperparameters
    history = {}
    for i in range(1, 2):
        number_of_nodes_l1 = i * 5
        for j in range(1, 10):
            number_of_nodes_l2 = j * 5
            print("Number of Nodes " + str(number_of_nodes_l1))
            model = Sequential([
                Dense(number_of_nodes_l1, input_shape=(42, )),
                Dense(number_of_nodes_l2),
                Dense(1)
            ])
            model.compile(optimizer='sgd', loss='mse')
            model.summary()
            history_callback = model.fit(
                x, y, validation_split=0.33, epochs=10, batch_size=128)
            dict_name = "i = " + str(number_of_nodes_l1) + \
                " j = " + str(number_of_nodes_l2)
            history[dict_name] = history_callback
    return history


def Fully_Connected_TwoL_relu(x, y):
    # Hyperparameters
    history = {}
    for i in range(1, 10):
        number_of_nodes_l1 = i * 5
        for j in range(1, 10):
            number_of_nodes_l2 = j * 5
            print("Number of Nodes " + str(number_of_nodes_l1))
            model = Sequential([
                Dense(number_of_nodes_l1, input_shape=(42, ), activation='relu'),
                Dense(number_of_nodes_l2),
                Dense(1)
            ])
            model.compile(optimizer='sgd', loss='mse')
            model.summary()
            history_callback = model.fit(
                x, y, validation_split=0.33, epochs=10, batch_size=128)
            dict_name = "i = " + str(number_of_nodes_l1) + \
                " j = " + str(number_of_nodes_l2)
            history[dict_name] = history_callback
    return history


def build_LSTM(x, y, timesteps):
    model = Sequential()
    model.add(LSTM(42, input_shape=(timesteps, 42), return_sequences=True))
    model.add(LSTM(42, return_sequences=True))
    model.add(Dense(1))
    model.compile('sgd', loss='mse')
    history = model.fit(x, y, validation_split=0.33, epochs=10, batch_size=128)
    return history
