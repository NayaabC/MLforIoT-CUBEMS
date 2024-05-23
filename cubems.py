import requests
import numpy as np
import pandas as pd

import os
import time
import datetime as datetime

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import tensorflow as tf
from keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten, Input, Attention, Dropout, Embedding, MultiHeadAttention, Add, LayerNormalization, TimeDistributed
from keras.models import Sequential, Model
from keras.optimizers import Adam

# Plot Imports
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.graph_objs as go
import cufflinks as cf
# from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import matrixprofile

# Model Imports
import lightgbm as lgb

cf.set_config_file(offline=True)
# cf.go_offline()
pd.options.plotting.backend = 'plotly'


def load_data(directory):
    data_2018 = pd.DataFrame()
    data_2019 = pd.DataFrame()  
    merged = pd.DataFrame()     

    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            df = pd.read_csv(file_path)

            # Append to the respective DataFrames based on the year in the filename
            if filename.startswith("2018"):
                data_2018 = pd.concat([data_2018, df], ignore_index=True)
                merged = pd.concat([merged, df], ignore_index=True)
            elif filename.startswith("2019"):
                data_2019 = pd.concat([data_2019, df], ignore_index=True)
                merged = pd.concat([merged, df], ignore_index=True)
    
    print("Done Loading Data")
    return data_2018, data_2019, merged


def plotPower(data, ret=False, save=[False, ""]):
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    powerDataDates = data.loc[:, data.columns[data.columns.str.contains('kW') | (data.columns == 'Date')]].copy()
    # print(powerDataDates)
    powerData = powerDataDates.set_index(powerDataDates.columns[0],drop=True)
    # powerData['Date'] = pd.to_datetime(powerData['Date'])
    powerData['total_demand'] = powerData.sum(axis=1)
    # print(powerData)
    powerData = powerData.resample('H').mean()

    plt.figure(figsize=(18, 5))
    plt.plot(powerData)
    plt.xlabel('Date')
    plt.ylabel('Power Consumption (kW)')
    plt.title('Power Consumption vs Time')

    if save[0]:
        path = r"C:\Users\NC\Documents\Rutgers\Grad\MLIoT\Programming\Final Project\outputs\power" 
        path = os.path.join(path, save[1])
        plt.savefig(path)

    if ret:
        plt.close()
        return powerData
    else: plt.show()

def plotTemp(data, ret=False, save=[False, ""], debug=False):
    

    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    tempDataDates = data.loc[:, data.columns[data.columns.str.contains('degC') | (data.columns == 'Date')]].copy()

    if debug:
        print(tempDataDates)
        nan_counts = tempDataDates.isna().sum()

        print("Number of NaN entries: ", nan_counts)
        print("Number of Non-NaN Entries: ", tempDataDates.count())

    tempData = tempDataDates.set_index(tempDataDates.columns[0],drop=True)
    # tempData['Date'] = pd.to_datetime(tempData['Date'])
    # tempData = tempData.sum(axis=1).rename('total_demand')
    tempData['avg_temperature'] = tempData.mean(axis=1)
    # print(tempData)
    tempData = tempData.resample('H').mean()
    # print(tempData)

    plt.figure(figsize=(18, 5))
    plt.plot(tempData)
    plt.xlabel('Date')
    plt.ylabel('Temperature (degC)')
    plt.title('Temperature vs Date')

    if save[0]:
        path = r"C:\Users\NC\Documents\Rutgers\Grad\MLIoT\Programming\Final Project\outputs\temp" 
        path = os.path.join(path, save[1])
        plt.savefig(path)
    
    if ret:
        plt.close()
        return tempData
    else: plt.show()

def plotRH(data, ret=False, save=[False, ""], debug=False):
    

    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    humidDataDates = data.loc[:, data.columns[data.columns.str.contains('RH%') | (data.columns == 'Date')]].copy()

    if debug:
        print(humidDataDates)
        nan_counts = humidDataDates.isna().sum()

        print("Number of NaN entries: ", nan_counts)
        print("Number of Non-NaN Entries: ", humidDataDates.count())

    humid = humidDataDates.set_index(humidDataDates.columns[0],drop=True)
    # humid['Date'] = pd.to_datetime(tempData['Date'])
    # humid = tempData.sum(axis=1).rename('total_demand')
    humid['avg_humidity'] = humid.mean(axis=1)
    # print(humid)
    humid = humid.resample('H').mean()
    # print(humid)

    plt.figure(figsize=(18, 5))

    # for col in humid.columns:
    #     plt.plot(humid.index, humid[col], label=col)
    # plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.plot(humid)
    plt.xlabel('Date')
    plt.ylabel('Average Relative Humidity (RH%)')
    plt.title('Relative Humidity vs Date')

    if save[0]:
        path = r"C:\Users\NC\Documents\Rutgers\Grad\MLIoT\Programming\Final Project\outputs\relativeHumidity" 
        path = os.path.join(path, save[1])
        plt.savefig(path)
    
    if ret:
        plt.close()
        return humid
    else: plt.show()

def plotLux(data, ret=False, save=[False, ""], debug=False):
    

    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    luxDataDates = data.loc[:, data.columns[data.columns.str.contains('lux') | (data.columns == 'Date')]].copy()

    if debug:
        print(luxDataDates)
        nan_counts = luxDataDates.isna().sum()

        print("Number of NaN entries: ", nan_counts)
        print("Number of Non-NaN Entries: ", luxDataDates.count())

    lux = luxDataDates.set_index(luxDataDates.columns[0],drop=True)
    # lux['Date'] = pd.to_datetime(lux['Date'])
    # lux = tempData.sum(axis=1).rename('total_demand')
    lux = lux.mean(axis=1).rename('avg_lux')
    # print(lux)
    lux = lux.resample('H').mean()
    # print(lux)

    plt.figure(figsize=(18, 5))

    # for col in lux.columns:
    #     plt.plot(lux.index, lux[col], label=col)
    # plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.plot(lux)
    plt.xlabel('Date')
    plt.ylabel('Illuminance Levels (lux)')
    plt.title('Illuminance Levels vs Date')

    if save[0]:
        path = r"C:\Users\NC\Documents\Rutgers\Grad\MLIoT\Programming\Final Project\outputs\lux" 
        path = os.path.join(path, save[1])
        plt.savefig(path)
    
    if ret:
        plt.close()
        return lux
    else: plt.show()

def plotMatrixProfile(data, windowSize, dataType, show=True, save=[False, ""]):
    timeSeries = data.values
    mat = matrixprofile.compute(timeSeries, windowSize)
    # print(mat)
    profile = mat['mp']

    plt.figure(figsize=(18, 12))
    plt.subplot(2, 1, 1)
    plt.plot(data)
    plt.xlabel('Time')
    plt.ylabel('Energy (kW)')
    plt.title('Power Consumption vs Time')


    plt.subplot(2, 1, 2)
    plt.plot(profile)
    plt.xlabel('Window Index')
    plt.ylabel(f"Distance Profile of {dataType}")
    plt.title('Matrix Profile Calculation')

    if save[0]:
        path = r"C:\Users\NC\Documents\Rutgers\Grad\MLIoT\Programming\Final Project\outputs\matrixProfiles" 
        path = os.path.join(path, save[1])
        plt.savefig(path)
    
    if show:
        plt.show()
    else:
        plt.close()

def PCA_analysis(data, components, decomp=False, save=[False, ""]):
    
    def PCA_decomp(data, components):
        print(data)

        pca = PCA(n_components=0.99)
        reducedData = pca.fit_transform(data)
        print(reducedData)

        n_pcs = pca.n_components_

        pcColumns = []
        for i in range(n_pcs):
            pccount = f"PC{i + 1}"
            pcColumns.append(pccount)

        pca_df = pd.DataFrame(data=reducedData, columns=pcColumns)
        final_data = pd.concat([data, pca_df], axis=1)

        df = pd.DataFrame(pca.components_, columns=data.columns)
        
        # Index of most important feature on each componend
        most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]
        feature_names = df.columns
        most_important_names = [feature_names[most_important[i]] for i in range(n_pcs)]
        print("Features with the most impact on the data (most important features): ", most_important_names)

        # Visualize 2D Projection
        
        plt.figure(figsize=(18, 12))
        plt.subplot(2,1,1)

        for col in data.columns:
          plt.plot(data.index, data[col], label=col)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('Original Demand and Averages vs Time')

        plt.subplot(2,1,2)
        for component in most_important_names:
            plt.plot(final_data.index, final_data[component], label=component)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('PCA Reduced Data/Signal')

        plt.tight_layout()

        if save[0]:
            path = r"C:\Users\NC\Documents\Rutgers\Grad\MLIoT\Programming\Final Project\outputs\pca" 
            path = os.path.join(path, save[1])
            plt.savefig(path)


    if decomp:
        PCA_decomp(data, components)
    else:
        sc = StandardScaler()
        pca = PCA()
        for d in data:
            dataSTD = sc.fit_transform(d.values)
            pcaData = pca.fit(dataSTD)
            plt.plot(np.cumsum(pcaData.explained_variance_ratio_))

        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')

        if save[0]:
            path = r"C:\Users\NC\Documents\Rutgers\Grad\MLIoT\Programming\Final Project\outputs\pca" 
            path = os.path.join(path, save[1])
            plt.savefig(path)
        
        plt.close()


def plotStage(data, stage):
    df_plot = data.copy()
    df_plot = df_plot[['total_demand', 'weekday', 'hour']]
    df_plot['date'] = pd.to_datetime(df_plot.index.date)
    print(df_plot)
    # df_plot['date'] = pd.to_datetime(df_plot.index.date)
    # .plot(figsize=(15,5), color='black', alpha=0.1, legend=False)


    # df_plot['weekday_hour_pair'] = list(zip(df_plot['weekday'], df_plot['hour']))
    pivot_table = df_plot.pivot_table(columns=['weekday', 'hour'], index='date', values='total_demand')
    # pivot_table = df_plot.pivot_table(columns='weekday_hour_pair', index='date', values='total_demand')
    ax = pivot_table.T.plot(figsize=(15, 5), color='black', alpha=0.1, legend=False)

    # plt.show()
    # pivot_table = pivot_table.T
    # print(pivot_table)
    
    # pivot_table.plot(figsize=(15, 5), color='black', alpha=0.1, legend=False)
    # plt.show()

    # plt.figure(figsize=(15, 5))

    # plt.plot(pivot_table, color='black', alpha=0.1)

    # for column in pivot_table.columns:
    #     plt.plot(pivot_table.index, pivot_table[column], color='black', alpha=0.1)

    # sns.lineplot(data=pivot_table, palette='tab20', alpha=0.1, legend=False)

    # for (weekday, hour), group in df_plot.groupby(['weekday', 'hour']):
    #     plt.plot((weekday, hour), group['total_demand'], color='black', alpha=0.1)


    plt.title(f"{stage} Energy Demand")
    plt.xlabel('(Weekday, Hour)')
    plt.ylabel('Total Demand Measured')
    # plt.legend().set_visible(False)
    plt.show()

    # weekly = df_plot.groupby(['weekday', 'hour']).mean()
    # plt.figure(figsize=(15, 5))

    # # plt.plot(weekly['date'], weekly['total_demand_measured'], marker='o', linestyle='-', color='black')
    # plt.show()

def plotModel_power(modelOutput, model="", save=[False, ""]):
    plt.close()
    plt.figure(figsize=(18, 15))
    plt.subplot(3,1,1)

    plt.plot(modelOutput.index, modelOutput['total_demand'])
    plt.xlabel('Time')
    plt.ylabel('Energy Demand (kW)')
    plt.title('Total Energy Demand vs Time')

    plt.subplot(3,1,2)
    plt.plot(modelOutput.index, modelOutput['total_demand_prediction'])
    plt.xlabel('Time')
    plt.ylabel('Energy Demand (kW)')
    plt.title(f'Predicted Energy Demand vs Time - {model}')

    plt.subplot(3,1,3)
    plt.plot(modelOutput.index, modelOutput['total_demand'])
    plt.plot(modelOutput.index, modelOutput['total_demand_prediction'])
    plt.xlabel('Time')
    plt.ylabel('Energy Demand (kW)')
    plt.title('Predicted Energy Demand vs Time - Combined')

    if save[0]:
            path = r"C:\Users\NC\Documents\Rutgers\Grad\MLIoT\Programming\Final Project\outputs\lightGBM" 
            path = os.path.join(path, save[1])
            plt.savefig(path)
        
    plt.close()

def plotEnergyTemp(data, mode=None):
    plt.close()
    plt.figure(figsize=(18, 15))
    df_plot = data.copy()
    df_plot = df_plot.resample('D').mean()
    df_plot['weekday/weekend'] = 'weekday'
    df_plot.loc[df_plot['weekday'] > 4, 'weekday/weekend'] = 'weekend'

    # ax = sns.relplot(x='avg_temperature', y='total_demand', col='weekday/weekend', kind='scatter', data=df_plot, alpha=0.8)

    if mode:
        ax = sns.relplot(x=mode, y='total_demand', col='weekday/weekend', kind='scatter', data=df_plot, alpha=0.8)
    plt.show()


def LSTM_power(trainData_power, testData_power, trainLabels_power, testLabels_power, trainFeatures_power, testFeatures_power, metrics=False):

    look_back = 24  # sliding window
    
    def create_sequences(data, labels, look_back):
        x, y = [], []
        for i in range(len(data) - look_back):
            x.append(data[i:(i + look_back)])
            y.append(labels[i + look_back])
        return np.array(x), np.array(y)
    
    scalar = MinMaxScaler()
    trainFeatures_pwrScaled = scalar.fit_transform(trainFeatures_power)
    testFeatures_pwrScaled = scalar.transform(testFeatures_power)

    X_train, y_train = create_sequences(trainFeatures_pwrScaled, trainLabels_power, look_back)
    X_test, y_test = create_sequences(testFeatures_pwrScaled, testLabels_power, look_back)

    model = Sequential()
    model.add(LSTM(units=50, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

    y_pred = model.predict(X_test)

    y_pred_index = testData_power.index[-len(y_pred):]

    testData_power['total_demand_prediction'] = np.nan
    testData_power.loc[y_pred_index, 'total_demand_prediction'] = y_pred
    # testData_power.index = y_pred_index
    
    y_pred_inv = scalar.inverse_transform(np.concatenate((X_test[:, -1, 1:], y_pred), axis=1))[:, -1]
    y_test_inv = scalar.inverse_transform(np.concatenate((X_test[:, -1, 1:], y_test.reshape(-1, 1)), axis=1))[:, -1]

    mse = mean_squared_error(y_test_inv, y_pred_inv)
    print(f'Mean Squared Error: {mse}')

    model.summary()

    plt.figure(figsize=(12, 6))
    plt.plot(testData_power.index[-len(y_test_inv):], y_test_inv, label='Actual Demand', color='blue')
    plt.plot(testData_power.index[-len(y_test_inv):], y_pred_inv, label='Predicted Demand', color='red')
    plt.title('Total Demand Forecasting with LSTM')
    plt.xlabel('Date')
    plt.ylabel('Total Demand')
    plt.legend()

    if not metrics:
        plt.show()
    else:
        plt.close()

        errors = abs(testData_power['total_demand_prediction'] - testLabels_power)

        mape = 100 * np.mean((errors / testLabels_power))
        nmbe = 100 * (sum(testData_power.dropna()['total_demand'] - testData_power.dropna()['total_demand_prediction']) / (testData_power.dropna()['total_demand'].count() * np.mean(testData_power.dropna()['total_demand'])))
        cvrsme = 100 * ((sum((testData_power.dropna()['total_demand'] - testData_power.dropna()['total_demand_prediction']) ** 2) / (testData_power.dropna()['total_demand'].count() - 1)) ** (0.5)) / np.mean(testData_power.dropna()['total_demand'])
        rsquared = r2_score(testData_power.dropna()['total_demand'], testData_power.dropna()['total_demand_prediction'])

        print(mape, nmbe, cvrsme, rsquared)
        return mape, nmbe, cvrsme, rsquared


def temporaL_CNN(trainData, testData, trainLabels, testLabels, trainFeatures, testFeatures, metrics=False):

    # Define the number of time steps to use for prediction
    look_back = 24 # sliding window

    # Normalize
    scaler = MinMaxScaler()
    trainFeatures_power_scaled = scaler.fit_transform(trainFeatures)
    testFeatures_power_scaled = scaler.transform(testFeatures)

    def reshape_data(data, labels, look_back):
        x, y = [], []
        for i in range(len(data) - look_back):
            x.append(data[i:(i + look_back)])
            y.append(labels.iloc[i + look_back])
        return np.array(x), np.array(y)

    X_train, y_train = reshape_data(trainFeatures_power_scaled, trainLabels, look_back)

    X_test, y_test = reshape_data(testFeatures_power_scaled, testLabels, look_back)

    # Specify the input shape for the CNN
    input_shape = (look_back, X_train.shape[2])

    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(units=100, activation='relu'))
    model.add(Dense(units=1))

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=200, batch_size=256, verbose=1)
    y_pred = model.predict(X_test)

    y_pred_index = testData.index[-len(y_pred):]

    testData['total_demand_prediction'] = np.nan
    testData.loc[y_pred_index, 'total_demand_prediction'] = y_pred

    y_pred_inv = scaler.inverse_transform(np.concatenate((X_test[:, -1, 1:], y_pred), axis=1))[:, -1]
    y_test_inv = scaler.inverse_transform(np.concatenate((X_test[:, -1, 1:], y_test.reshape(-1, 1)), axis=1))[:, -1]

    mse = mean_squared_error(y_test_inv, y_pred_inv)
    print(f'Mean Squared Error: {mse}')

    model.summary()

    plt.figure(figsize=(12, 6))
    plt.plot(testData.index[-len(y_test_inv):], y_test_inv, label='Actual Demand', color='blue')
    plt.plot(testData.index[-len(y_test_inv):], y_pred_inv, label='Predicted Demand', color='red')
    plt.title('Total Demand Forecasting with Temporal CNN')
    plt.xlabel('Date')
    plt.ylabel('Total Demand')
    plt.legend()
    
    if not metrics:
        plt.show()
    else:
        plt.close()

        errors = abs(testData['total_demand_prediction'] - testLabels_power)

        mape = 100 * np.mean((errors / testLabels_power))
        nmbe = 100 * (sum(testData.dropna()['total_demand'] - testData.dropna()['total_demand_prediction']) / (testData.dropna()['total_demand'].count() * np.mean(testData.dropna()['total_demand'])))
        cvrsme = 100 * ((sum((testData.dropna()['total_demand'] - testData.dropna()['total_demand_prediction']) ** 2) / (testData.dropna()['total_demand'].count() - 1)) ** (0.5)) / np.mean(testData.dropna()['total_demand'])
        rsquared = r2_score(testData.dropna()['total_demand'], testData.dropna()['total_demand_prediction'])

        print(mape, nmbe, cvrsme, rsquared)
        return mape, nmbe, cvrsme, rsquared


def lateTCNN_powerTemp(data):
    print(data)
    trainData_power = data.loc['2018-07':'2019-06'].copy()
    testData_power = data.loc['2019-07':].copy()

    trainLabels_power = trainData_power['total_demand']
    trainFeatures_power = trainData_power.drop('total_demand', axis=1)

    testLabels_power = testData_power['total_demand']
    testFeatures_power = testData_power.drop('total_demand', axis=1)

    # Normalize
    scaler_power = MinMaxScaler()
    trainFeatures_power_scaled = scaler_power.fit_transform(trainFeatures_power)
    testFeatures_power_scaled = scaler_power.transform(testFeatures_power)

    # Define the number of time steps to use for prediction
    look_back = 24  # sliding window

    def reshape_data(data, labels, look_back):
        x, y = [], []
        for i in range(len(data) - look_back):
            x.append(data[i:(i + look_back)])
            y.append(labels.iloc[i + look_back])
        return np.array(x), np.array(y)

    X_train_power, y_train_power = reshape_data(trainFeatures_power_scaled, trainLabels_power, look_back)
    X_test_power, y_test_power = reshape_data(testFeatures_power_scaled, testLabels_power, look_back)

    model_total_demand = Sequential()
    model_total_demand.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(look_back, X_train_power.shape[2])))
    model_total_demand.add(MaxPooling1D(pool_size=2))
    model_total_demand.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
    model_total_demand.add(MaxPooling1D(pool_size=2))
    model_total_demand.add(Flatten())
    model_total_demand.add(Dense(units=100, activation='relu'))
    model_total_demand.add(Dense(units=1))

    model_total_demand.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    model_total_demand.fit(X_train_power, y_train_power, epochs=50, batch_size=256, verbose=1)
    y_pred_total_demand = model_total_demand.predict(X_test_power)

    y_pred_inv_total_demand = scaler_power.inverse_transform(np.concatenate((X_test_power[:, -1, 1:], y_pred_total_demand), axis=1))[:, -1]
    y_test_inv_total_demand = scaler_power.inverse_transform(np.concatenate((X_test_power[:, -1, 1:], y_test_power.reshape(-1, 1)), axis=1))[:, -1]

    mse_total_demand = mean_squared_error(y_test_inv_total_demand, y_pred_inv_total_demand)
    print(f'Mean Squared Error for Total Demand: {mse_total_demand}')

    model_avg_temperature = Sequential()
    model_avg_temperature.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(look_back, X_train_power.shape[2])))
    model_avg_temperature.add(MaxPooling1D(pool_size=2))
    model_avg_temperature.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
    model_avg_temperature.add(MaxPooling1D(pool_size=2))
    model_avg_temperature.add(Flatten())
    model_avg_temperature.add(Dense(units=100, activation='relu'))
    model_avg_temperature.add(Dense(units=1))

    model_avg_temperature.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    model_avg_temperature.fit(X_train_power, trainData_power['avg_temperature'].iloc[look_back:], epochs=50, batch_size=256, verbose=1)
    y_pred_avg_temperature = model_avg_temperature.predict(X_test_power)

    mse_avg_temperature = mean_squared_error(testData_power['avg_temperature'].iloc[look_back:], y_pred_avg_temperature)
    print(f'Mean Squared Error for Avg Temperature: {mse_avg_temperature}')


    # Define weights
    weight_total_demand = 0.8  # Adjust as needed
    weight_avg_temperature = 0.2  # Adjust as needed

    weighted_average_predictions = (weight_total_demand * y_pred_inv_total_demand) + (weight_avg_temperature * y_pred_avg_temperature.flatten())

    y_pred_fusion = weighted_average_predictions

    plt.figure(figsize=(12, 6))
    plt.plot(testData_power.index[-len(y_test_inv_total_demand):], y_test_inv_total_demand, label='Actual Demand', color='blue')
    # plt.plot(testData.index[-len(y_test_inv):], y_test_inv, label='Actual Demand', color='blue')
    plt.plot(testData_power.index[-len(y_test_inv_total_demand):], y_pred_fusion, label='Fused Prediction', color='green')
    plt.xlabel('Date')
    plt.ylabel('Total Demand')
    plt.title('Total Demand Prediction with Late Fusion')
    plt.legend()
    plt.show()


# Code copied from lateTCNN_powerTemp, didnt get time to change variable names
def lateTCNN_powerLux(data):
    # print(data)
    trainData_power = data.loc['2018-07':'2019-06'].copy()
    testData_power = data.loc['2019-07':].copy()
    
    trainLabels_power = trainData_power['total_demand']
    trainFeatures_power = trainData_power.drop('total_demand', axis=1)

    testLabels_power = testData_power['total_demand']
    testFeatures_power = testData_power.drop('total_demand', axis=1)

    # Normalize
    scaler_power = MinMaxScaler()
    trainFeatures_power_scaled = scaler_power.fit_transform(trainFeatures_power)
    testFeatures_power_scaled = scaler_power.transform(testFeatures_power)

    look_back = 24 # sliding window size

    def reshape_data(data, labels, look_back):
        x, y = [], []
        for i in range(len(data) - look_back):
            x.append(data[i:(i + look_back)])
            y.append(labels.iloc[i + look_back])
        return np.array(x), np.array(y)

    X_train_power, y_train_power = reshape_data(trainFeatures_power_scaled, trainLabels_power, look_back)
    X_test_power, y_test_power = reshape_data(testFeatures_power_scaled, testLabels_power, look_back)

    model_total_demand = Sequential()
    model_total_demand.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(look_back, X_train_power.shape[2])))
    model_total_demand.add(MaxPooling1D(pool_size=2))
    model_total_demand.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
    model_total_demand.add(MaxPooling1D(pool_size=2))
    model_total_demand.add(Flatten())
    model_total_demand.add(Dense(units=100, activation='relu'))
    model_total_demand.add(Dense(units=1))

    model_total_demand.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    model_total_demand.fit(X_train_power, y_train_power, epochs=50, batch_size=256, verbose=1)

    y_pred_total_demand = model_total_demand.predict(X_test_power)

    y_pred_inv_total_demand = scaler_power.inverse_transform(np.concatenate((X_test_power[:, -1, 1:], y_pred_total_demand), axis=1))[:, -1]
    y_test_inv_total_demand = scaler_power.inverse_transform(np.concatenate((X_test_power[:, -1, 1:], y_test_power.reshape(-1, 1)), axis=1))[:, -1]

    mse_total_demand = mean_squared_error(y_test_inv_total_demand, y_pred_inv_total_demand)
    print(f'Mean Squared Error for Total Demand: {mse_total_demand}')

    # Build the model for avg_temperature
    model_avg_lux = Sequential()
    model_avg_lux.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(look_back, X_train_power.shape[2])))
    model_avg_lux.add(MaxPooling1D(pool_size=2))
    model_avg_lux.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
    model_avg_lux.add(MaxPooling1D(pool_size=2))
    model_avg_lux.add(Flatten())
    model_avg_lux.add(Dense(units=100, activation='relu'))
    model_avg_lux.add(Dense(units=1))

    model_avg_lux.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    model_avg_lux.fit(X_train_power, trainData_power['avg_lux'].iloc[look_back:], epochs=50, batch_size=256, verbose=1)
    y_pred_avg_lux = model_avg_lux.predict(X_test_power)

  
    mse_avg_lux = mean_squared_error(testData_power['avg_lux'].iloc[look_back:], y_pred_avg_lux)
    print(f'Mean Squared Error for Avg Lux: {mse_avg_lux}')


    # weights for the models
    weight_total_demand = 0.8
    weight_avg_lux = 0.2

    # Perform weighted late data fusion
    weighted_average_predictions = (weight_total_demand * y_pred_inv_total_demand) + (weight_avg_lux * y_pred_avg_lux.flatten())

    y_pred_fusion = weighted_average_predictions

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(testData_power.index[-len(y_test_inv_total_demand):], y_test_inv_total_demand, label='Actual Demand', color='blue')
    # plt.plot(testData.index[-len(y_test_inv):], y_test_inv, label='Actual Demand', color='blue')
    plt.plot(testData_power.index[-len(y_test_inv_total_demand):], y_pred_fusion, label='Fused Prediction', color='green')
    plt.xlabel('Date')
    plt.ylabel('Total Demand')
    plt.title('Total Demand Prediction with Late Fusion')
    plt.legend()
    plt.show()

# Attempt at implementing a transformer architecture - did not work
def transformer_powerTemp(data):
    df = data
    X = df.drop('total_demand', axis=1)
    y = df['total_demand']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_reshaped = X_train_scaled[:, np.newaxis, :]
    X_test_reshaped = X_test_scaled[:, np.newaxis, :]

    model = Sequential([
        MultiHeadAttention(num_heads=2, key_dim=X_train_scaled.shape[1]),
        TimeDistributed(Dense(64, activation='relu')),
        LayerNormalization(epsilon=1e-6),
        TimeDistributed(Dense(1))
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X_train_reshaped, y_train, epochs=10, batch_size=64)

    y_pred = model.predict(X_test_reshaped)
    mse = mean_squared_error(y_test, y_pred.reshape(-1, 1))
    print(f'Mean Squared Error on Test Set: {mse}')

    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['total_demand'], label='Actual total_demand', color='blue')

    test_indices = X_test.index
    plt.plot(test_indices, y_pred, label='Predicted total_demand', color='red', linestyle='dashed')
    plt.title('Actual vs Predicted Total Demand')
    plt.xlabel('Date')
    plt.ylabel('Total Demand')
    plt.legend()
    plt.show()





if __name__ == "__main__":
    """]
    
        TODO:

        0) Plot RH% and lux measurements normally and matrixProfile --------------- DONE (maybe not necessary)
        !) Plot Temperature - Figure out why NaN values exist   ------------------- DONE
        2) Matrix Profile - Identify motifs from domain knowledge ----------------- DONE (maybe not necessary)
        3) PCA / Dimensionality Reduction   (might take a long time) -------------- DONE (not necessary in the end)
        4) LightBGM model to predict measured load -> use power and temp as fused dataa
            4b - Do the same thing between power and humidity
            4c - Try to do the same thing using power, temperature and humidity as a fused modality data - early fusion (concatentation), late fusion (one model for each modal), transformer

        5) repeat (4) with other multimodal models -------------------------------- DONE using LSTM and Temporal CNN
        6) Compare the model results, provide visuals ----------------------------- DONE using statistical metrics.
        7) Conclusion in Report --------------------------------------------------- DONE
    """


    # Performance Metrics

    all_errors = mapes = nmbes = cvrsmes = rscores = []
    indices = ['power', 'powerTemp', 'powerHumidity', 'powerLux']

    lstm_metrics_early = pd.DataFrame(index = indices, columns=['mape', 'nmbe', 'cvrsme', 'rscore'])
    tcnn_metrics_early = pd.DataFrame(columns=['mape', 'nmbe', 'cvrsme', 'rscore'])
    lightGBM_metrics_early = pd.DataFrame(columns=['mape', 'nmbe', 'cvrsme', 'rscore'])

    lstm_metrics_late = pd.DataFrame(columns=['mape', 'nmbe', 'cvrsme', 'rscore'])
    tcnn_metrics_late = pd.DataFrame(columns=['mape', 'nmbe', 'cvrsme', 'rscore'])
    lightGBM_metrics_late = pd.DataFrame(columns=['mape', 'nmbe', 'cvrsme', 'rscore'])


    # Uncomment function calls as necessary
    print(" UNCOMMENT FUNCTION CALLS AS NECESSARY ")


    # Yes, I am aware that the code gets more unorganized as the project progressed. I apologize in advance.
    # this was due to time constraints.

    print("=============== LOADING CUBEMS DATASET =========================")
    
    cubems_path = r"C:\Users\NC\Documents\Rutgers\Grad\MLIoT\Programming\Final Project\datasets\cubems"
    cubems_2 = r"C:\Users\NC\Documents\Rutgers\Grad\MLIoT\Programming\Final Project\datasets\11726517"
    data_18, data_19, merged = load_data(cubems_path)
    # print(data_18.columns)
    # print(list(merged.columns))
    
    print("================== PLOT POWER FUNCTION =============================")
    powerMeter_18 = plotPower(data_18, ret=True, save=[True, "2018_power.png"])
    powerMeter_19 = plotPower(data_19, ret=True, save=[True, "2019_power.png"])
    powerMeter_merged = plotPower(merged, ret=True, save=[True, "merged_power.png"])
    # print("==================END PLOT POWER FUNCTION===========================")

    print("================== PLOT TEMPERATURE FUNCTION =======================")
    tempAvg_18 = plotTemp(data_18, ret=True, save=[True, "2018_temp.png"])
    tempAvg_19 = plotTemp(data_19, ret=True, save=[True, "2019_temp.png"])
    tempAvg_merged = plotTemp(merged, ret=True, save=[True, "merged_temp.png"])
    # print("==================END PLOT TEMPERATURE FUNCTION===========================")


    print("================== PLOT RELATIVE HUMIDITY FUNCTION =================")
    humid_18 = plotRH(data_18, ret=True, save=[True, "2018_RH.png"])
    humid_19 = plotRH(data_19, ret=True, save=[True, "2019_RH.png"])
    humid_merged = plotRH(merged, ret=True, save=[True, "merged_RH.png"])


    print("================== PLOT LUX FUNCTION =================")
    lux_18 = plotLux(data_18, ret=True, save=[True, "2018_lux.png"])
    lux_19 = plotLux(data_19, ret=True, save=[True, "2019_lux.png"])
    lux_merged = plotLux(merged, ret=True, save=[True, "merged_lux.png"])

    print("================== MATRIX PROFILE CALCULATIONS =====================")
    # print(powerMeter_18.values)
    # Arbitrary window size
    window_size = 100

    # plotMatrixProfile(powerMeter_18, window_size, "Power (kW)", show=False, save=[True, "2018_power_mp.png"])
    # plotMatrixProfile(powerMeter_19, window_size, "Power (kW)", show=False, save=[True, "2019_power_mp.png"])
    # plotMatrixProfile(powerMeter_merged, window_size, "Power (kW)", show=False, save=[True, "merges_power_mp.png"])

    # plotMatrixProfile(tempAvg_18, window_size, "Temperature (degC)", show=False, save=[True, "2018_temp_mp.png"])
    # plotMatrixProfile(tempAvg_19, window_size, "Temperature (degC)", show=False, save=[True, "2019_temp_mp.png"])
    # plotMatrixProfile(tempAvg_merged, window_size, "Temperature (degC)", show=False, save=[True, "merged_temp_mp.png"])

    print("================== PCA DECOMPOSITION =================================")

    # Making Dataframes that contain the average temperature, humidity and lux in separate columns along with the
    # total energy demand for the same timestamp
    # We will use this in PCA analysis and in our modeling to see if energy load can be predicted.
    pwrAvgConcat_18 = pd.concat([powerMeter_18, tempAvg_18, humid_18, lux_18], axis=1).dropna()
    pwrAvgConcat_19 = pd.concat([powerMeter_19, tempAvg_19, humid_19, lux_19], axis=1).dropna()
    pwrAvgConcat_merged = pd.concat([powerMeter_merged, tempAvg_merged, humid_merged, lux_merged], axis=1).dropna()


    # PCA_analysis(pwrAvgConcat_18, 2, decomp=True, save=[True, "2018_decomp.png"])
    # PCA_analysis(pwrAvgConcat_19, 2, decomp=True, save=[True, "2019_decomp.png"])
    # PCA_analysis(pwrAvgConcat_merged, 2, decomp=True, save=[True, "merged_decomp.png"])
    # PCA_analysis(pwrAvgConcat_19, 2, decomp=True, save=[True, "2019_decomp.png"])
    # PCA_analysis(pwrAvgConcat_merged, 2, decomp=True, save=[True, "merged_decomp.png"])
    
    # PCA_analysis(pwrAvgConcat_19, 2, save=[True, "2019_analysis.png"])
    # PCA_analysis(pwrAvgConcat_merged, 2, save=[True, "merged_analysis.png"])

    # PCA_analysis([pwrAvgConcat_18, pwrAvgConcat_19, pwrAvgConcat_merged], 2, save=[True, "cer_pca_analysis.png"])
    important_merged = pwrAvgConcat_merged[['total_demand', 'avg_humidity', 'avg_temperature', 'avg_lux']]
    important_merged = important_merged.reset_index().copy()
    important_merged = important_merged.dropna()
    important_merged['weekday'] = important_merged['Date'].dt.weekday
    important_merged['hour'] = important_merged['Date'].dt.hour
    important_merged['date'] = pd.to_datetime(important_merged['Date'].dt.date)
    important_merged = important_merged.set_index('Date').drop(['date'], axis=1)
    # print(pwrAvgConcat_merged[['total_demand', 'avg_humidity', 'avg_temperature', 'avg_lux']])
    print(important_merged)
    print("================== EXTRACTING TIMESTAMP FEATURES ==========================")

    pwr_mergedTemp = powerMeter_merged.reset_index().copy()
    pwr_mergedTemp = pwr_mergedTemp.dropna()
    pwr_mergedTemp['weekday'] = pwr_mergedTemp['Date'].dt.weekday
    pwr_mergedTemp['hour'] = pwr_mergedTemp['Date'].dt.hour
    pwr_mergedTemp['date'] = pd.to_datetime(pwr_mergedTemp['Date'].dt.date)
    pwr_mergedTemp = pwr_mergedTemp.set_index('Date').drop(['date'], axis=1)
    pwr_mergedTemp = pwr_mergedTemp[['total_demand', 'weekday', 'hour']]

    # print(pwr_mergedTemp)
    # plotStage(pwr_mergedTemp, "Weekly")

    print("================== MODEL INPUT DATA PREP - ONLY POWER =================================")


    # Use portion of data as training and the remaining as test data
    trainData_power = pwr_mergedTemp.loc['2018-7' : '2019-6'].copy()
    testData_power = pwr_mergedTemp.loc['2019-7':].copy()

    trainLabels_power = trainData_power['total_demand']
    testLabels_power = testData_power['total_demand']

    trainFeatures_power = trainData_power.drop('total_demand', axis=1)
    testFeatures_power = testData_power.drop('total_demand', axis=1)

    # plotEnergyTemp(powerTemp_merged)

    print("================== LSTM - POWER =================================")
    # mape, nmbe, cvrsme, rsquared = LSTM_power(trainData_power, testData_power, trainLabels_power, testLabels_power, trainFeatures_power, testFeatures_power, metrics=True)

    # lstm_metrics_early.loc['power', :] = mape, nmbe, cvrsme, rsquared

    

    print("================== TEMPORAL CNN - POWER =================================")
    # tcnn_metrics_early.loc['power', :] = temporaL_CNN(trainData_power, testData_power, trainLabels_power, testLabels_power, trainFeatures_power, testFeatures_power, metrics=True)

    print("================== TRANSFORMER - POWER/TEMP =================================")
    # transformer_powerTemp(pwrAvgConcat_merged[['total_demand', 'avg_humidity', 'avg_temperature', 'avg_lux']])


    print("================== LIGHT GBM - POWER ==========================")
    # Use the LGBM model for forecast prediction
    LGB_model = lgb.LGBMRegressor()

    print("Train Features: ")
    print(trainFeatures_power)

    print("Train Labels: ")
    print(trainLabels_power)


    LGB_model.fit(trainFeatures_power, trainLabels_power)

    testData_power['total_demand_prediction'] = LGB_model.predict(testFeatures_power)

    powerMeter_merged.loc[testData_power.index, 'total_demand_prediction'] = testData_power['total_demand_prediction']

    # Calculate Absolkute Errors
    errors = abs(testData_power['total_demand_prediction'] - testLabels_power)

    # Calculate Mean Absolute Persentage Error (MAPE)
    mape = 100 * np.mean((errors / testLabels_power))
    nmbe = 100 * (sum(testData_power.dropna()['total_demand'] - testData_power.dropna()['total_demand_prediction']) / (testData_power.dropna()['total_demand'].count() * np.mean(testData_power.dropna()['total_demand'])))
    cvrsme = 100 * ((sum((testData_power.dropna()['total_demand'] - testData_power.dropna()['total_demand_prediction']) ** 2) / (testData_power.dropna()['total_demand'].count() - 1)) ** (0.5)) / np.mean(testData_power.dropna()['total_demand'])
    rsquared = r2_score(testData_power.dropna()['total_demand'], testData_power.dropna()['total_demand_prediction'])

    print('MAPE: ' + str(round(mape, 2)))
    print("NMBE: " + str(round(nmbe, 2)))
    print("CVRSME: " + str(round(cvrsme, 2)))
    print("R SQUARED: " + str(round(rsquared, 2)))

    lightGBM_metrics_early.loc['power', :] = str(round(mape, 2)), str(round(nmbe, 2)), str(round(cvrsme, 2)), str(round(rsquared, 2))

    
    print("================== MODEL INPUT DATA PREP - POWER/TEMP =================================")

    # print(pwrAvgConcat_merged.columns)

    powerTemp_merged = pwrAvgConcat_merged[['total_demand', 'avg_temperature']].reset_index().copy()
    powerTemp_merged = powerTemp_merged.dropna()

    powerTemp_merged['weekday'] = powerTemp_merged['Date'].dt.weekday
    powerTemp_merged['hour'] = powerTemp_merged['Date'].dt.hour
    powerTemp_merged['date'] = pd.to_datetime(powerTemp_merged['Date'].dt.date)

    powerTemp_merged = powerTemp_merged.set_index('Date').drop(['date'], axis=1)

    trainData_powerTemp = powerTemp_merged.loc['2018-7' : '2019-6'].copy()
    testData_powerTemp = powerTemp_merged.loc['2019-7':].copy()

    trainLabels_powerTemp = trainData_powerTemp['total_demand']
    testLabels_powerTemp = testData_powerTemp['total_demand']

    trainFeatures_powerTemp = trainData_powerTemp.drop('total_demand', axis=1)
    testFeatures_powerTemp = testData_powerTemp.drop('total_demand', axis=1)

    # print(powerTemp_merged)

    print("================== LSTM - POWER/TEMP =================================")
    # mape, nmbe, cvrsme, rsquared = LSTM_power(trainData_powerTemp, testData_powerTemp, trainLabels_powerTemp, testLabels_powerTemp, trainFeatures_powerTemp, testFeatures_powerTemp, metrics=True)
    # lstm_metrics_early.loc['powerTemp', :] = mape, nmbe, cvrsme, rsquared
    
    print("================== TEMPORAL CNN - POWER/TEMP - EARLY FUSION =================================")
    # tcnn_metrics_early.loc['powerTemp', :] = temporaL_CNN(trainData_powerTemp, testData_powerTemp, trainLabels_powerTemp, testLabels_powerTemp, trainFeatures_powerTemp, testFeatures_powerTemp, metrics=True)

    print("================== TEMPORAL CNN - POWER/TEMP - LATE FUSION =================================")
    # lateTCNN_powerTemp(powerTemp_merged)

    print("================== LIGHT GBM - POWER TEMP ======================")
    LGB_model = lgb.LGBMRegressor()
    print("Train Features: ")
    print(trainFeatures_powerTemp)

    print("Train Labels: ")
    print(trainLabels_powerTemp)
    LGB_model.fit(trainFeatures_powerTemp, trainLabels_powerTemp)

    testData_powerTemp['total_demand_prediction'] = LGB_model.predict(testFeatures_powerTemp)
    powerTemp_merged.loc[testData_powerTemp.index, 'total_demand_prediction'] = testData_powerTemp['total_demand_prediction']

    # Calculate Absolue Errors
    errors = abs(testData_powerTemp['total_demand_prediction'] - testLabels_powerTemp)

    # Calculate mean absolue percentage error (MAPE)
    mape = 100 * np.mean((errors / testLabels_powerTemp))
    nmbe = 100 * (sum(testData_powerTemp.dropna()['total_demand'] - testData_powerTemp.dropna()['total_demand_prediction']) / (testData_powerTemp.dropna()['total_demand'].count() * np.mean(testData_powerTemp.dropna()['total_demand'])))
    cvrsme = 100 * ((sum((testData_powerTemp.dropna()['total_demand'] - testData_powerTemp.dropna()['total_demand_prediction']) ** 2) / (testData_powerTemp.dropna()['total_demand'].count() - 1)) ** (0.5)) / np.mean(testData_powerTemp.dropna()['total_demand'])
    rsquared = r2_score(testData_powerTemp.dropna()['total_demand'], testData_powerTemp.dropna()['total_demand_prediction'])

    print('MAPE: ' + str(round(mape, 2)))
    print("NMBE: " + str(round(nmbe, 2)))
    print("CVRSME: " + str(round(cvrsme, 2)))
    print("R SQUARED: " + str(round(rsquared, 2)))
    lightGBM_metrics_early.loc['powerTemp', :] = str(round(mape, 2)), str(round(nmbe, 2)), str(round(cvrsme, 2)), str(round(rsquared, 2))

    # plotModel_power(testData_powerTemp, model='Light GBM Model', save=[False, "early_powerTemp_LGBM.png"])
    print("============== MODEL INPUT DATA PREP - POWER/HUMIDITY =============================")

    # print(pwrAvgConcat_merged.columns)
    powerHumid_merged = pwrAvgConcat_merged[['total_demand', 'avg_humidity']].reset_index().copy()
    powerHumid_merged = powerHumid_merged.dropna()

    powerHumid_merged['weekday'] = powerHumid_merged['Date'].dt.weekday
    powerHumid_merged['hour'] = powerHumid_merged['Date'].dt.hour
    powerHumid_merged['date'] = pd.to_datetime(powerHumid_merged['Date'].dt.date)

    powerHumid_merged = powerHumid_merged.set_index('Date').drop(['date'], axis=1)

    trainData_powerHumid = powerHumid_merged.loc['2018-7' : '2019-6'].copy()
    testData_powerHumid = powerHumid_merged.loc['2019-7':].copy()

    trainLabels_powerHumid = trainData_powerHumid['total_demand']
    testLabels_powerHumid = testData_powerHumid['total_demand']

    trainFeatures_powerHumid = trainData_powerHumid.drop('total_demand', axis=1)
    testFeatures_powerHumid = testData_powerHumid.drop('total_demand', axis=1)

    # plotEnergyTemp(powerHumid_merged, mode='avg_humidity')
    # lstm_metrics_early.loc['powerHumidity', :] = LSTM_power(trainData_powerHumid, testData_powerHumid, trainLabels_powerHumid, testLabels_powerHumid, trainFeatures_powerHumid, testFeatures_powerHumid, metrics=True)

    print("================== LIGHT GBM - POWER HUMIDITY ======================")
    LGB_model = lgb.LGBMRegressor()
    print("Train Features: ")
    print(trainFeatures_powerHumid)

    print("Train Labels: ")
    print(trainLabels_powerHumid)
    LGB_model.fit(trainFeatures_powerHumid, trainLabels_powerHumid)

    testData_powerHumid['total_demand_prediction'] = LGB_model.predict(testFeatures_powerHumid)
    powerHumid_merged.loc[testData_powerHumid.index, 'total_demand_prediction'] = testData_powerHumid['total_demand_prediction']

    # Calculate Absolue Errors
    errors = abs(testData_powerHumid['total_demand_prediction'] - testLabels_powerHumid)

    # Calculate mean absolue percentage error (MAPE)
    mape = 100 * np.mean((errors / testLabels_powerHumid))
    nmbe = 100 * (sum(testData_powerHumid.dropna()['total_demand'] - testData_powerHumid.dropna()['total_demand_prediction']) / (testData_powerHumid.dropna()['total_demand'].count() * np.mean(testData_powerHumid.dropna()['total_demand'])))
    cvrsme = 100 * ((sum((testData_powerHumid.dropna()['total_demand'] - testData_powerHumid.dropna()['total_demand_prediction']) ** 2) / (testData_powerHumid.dropna()['total_demand'].count() - 1)) ** (0.5)) / np.mean(testData_powerHumid.dropna()['total_demand'])
    rsquared = r2_score(testData_powerHumid.dropna()['total_demand'], testData_powerHumid.dropna()['total_demand_prediction'])

    print('MAPE: ' + str(round(mape, 2)))
    print("NMBE: " + str(round(nmbe, 2)))
    print("CVRSME: " + str(round(cvrsme, 2)))
    print("R SQUARED: " + str(round(rsquared, 2)))

    lightGBM_metrics_early.loc['powerHumidity', :] = str(round(mape, 2)), str(round(nmbe, 2)), str(round(cvrsme, 2)), str(round(rsquared, 2))

    # plotModel_power(testData_powerHumid, model='Light GBM Model', save=[True, "early_powerHumid_LGBM.png"])

    # print("Training Labels: ", trainLabels_power)
    # print("Testing Labels: ", testLabels_power)

    # print(powerMeter_18)
    # print(tempAvg_18)


    # Prepare data for modeling
    # fore_timeSeries = powerMeter_merged.reset_index().copy()
    # fore_timeSeries = fore_timeSeries.dropna()

    # # Add timestamp features
    # fore_timeSeries['weekday'] = fore_timeSeries['Date'].dt.weekday
    # fore_timeSeries['hour'] = fore_timeSeries['Date'].dt.hour
    # fore_timeSeries['date'] = pd.to_datetime(fore_timeSeries['Date'].dt.date)
    # fore_timeSeries = fore_timeSeries.set_index('Date').drop(['date'], axis=1)
    # fore_timeSeries = fore_timeSeries.rename(columns={'total_demand': 'total_demand_measured'})

    # print(fore_timeSeries)

    print("================== TEMPORAL CNN - POWER/HUMIDITY - LATE FUSION =================================")
    # lateTCNN_powerTemp(powerHumid_merged.dropna())
    # temporaL_CNN(trainData_powerHumid, testData_powerHumid, trainLabels_powerHumid, testLabels_powerHumid, trainFeatures_powerHumid, testFeatures_powerHumid)

    print("============== MODEL INPUT DATA PREP - POWER/LUX =============================")

    # print(pwrAvgConcat_merged.columns)

    powerLux_merged = pwrAvgConcat_merged[['total_demand', 'avg_lux']].reset_index().copy()
    powerLux_merged = powerLux_merged.dropna()

    powerLux_merged['weekday'] = powerLux_merged['Date'].dt.weekday
    powerLux_merged['hour'] = powerLux_merged['Date'].dt.hour
    powerLux_merged['date'] = pd.to_datetime(powerLux_merged['Date'].dt.date)

    powerLux_merged = powerLux_merged.set_index('Date').drop(['date'], axis=1)

    trainData_powerLux = powerLux_merged.loc['2018-7' : '2019-6'].copy()
    testData_powerLux = powerLux_merged.loc['2019-7':].copy()

    trainLabels_powerLux = trainData_powerLux['total_demand']
    testLabels_powerLux = testData_powerLux['total_demand']

    trainFeatures_powerLux = trainData_powerLux.drop('total_demand', axis=1)
    testFeatures_powerLux = testData_powerLux.drop('total_demand', axis=1)

    # plotEnergyTemp(powerLux_merged, 'avg_lux')
    # lstm_metrics_early.loc['powerLux', :] = LSTM_power(trainData_powerLux, testData_powerLux, trainLabels_powerLux, testLabels_powerLux, trainFeatures_powerLux, testFeatures_powerLux, metrics=True)
    print("================== TEMPORAL CNN - POWER/HUMIDITY - LATE FUSION =================================")
    # lateTCNN_powerLux(powerLux_merged)
    # tcnn_metrics_early.loc['powerHumidity', :] = temporaL_CNN(trainData_powerHumid, testData_powerHumid, trainLabels_powerHumid, testLabels_powerHumid, trainFeatures_powerHumid, testFeatures_powerHumid, metrics=True)


    print("================== LIGHT GBM - POWER/LUX ======================")
    LGB_model = lgb.LGBMRegressor()
    print("Train Features: ")
    print(trainFeatures_powerLux)

    print("Train Labels: ")
    print(trainLabels_powerLux)
    LGB_model.fit(trainFeatures_powerLux, trainLabels_powerLux)

    testData_powerLux['total_demand_prediction'] = LGB_model.predict(testFeatures_powerLux)
    powerLux_merged.loc[testData_powerLux.index, 'total_demand_prediction'] = testData_powerLux['total_demand_prediction']

    # Calculate Absolue Errors
    errors = abs(testData_powerLux['total_demand_prediction'] - testLabels_powerLux)

    # Calculate mean absolue percentage error (MAPE)
    mape = 100 * np.mean((errors / testLabels_powerLux))
    nmbe = 100 * (sum(testData_powerLux.dropna()['total_demand'] - testData_powerLux.dropna()['total_demand_prediction']) / (testData_powerLux.dropna()['total_demand'].count() * np.mean(testData_powerLux.dropna()['total_demand'])))
    cvrsme = 100 * ((sum((testData_powerLux.dropna()['total_demand'] - testData_powerLux.dropna()['total_demand_prediction']) ** 2) / (testData_powerLux.dropna()['total_demand'].count() - 1)) ** (0.5)) / np.mean(testData_powerLux.dropna()['total_demand'])
    rsquared = r2_score(testData_powerLux.dropna()['total_demand'], testData_powerLux.dropna()['total_demand_prediction'])

    print('MAPE: ' + str(round(mape, 2)))
    print("NMBE: " + str(round(nmbe, 2)))
    print("CVRSME: " + str(round(cvrsme, 2)))
    print("R SQUARED: " + str(round(rsquared, 2)))

    lightGBM_metrics_early.loc['powerLux', :] = str(round(mape, 2)), str(round(nmbe, 2)), str(round(cvrsme, 2)), str(round(rsquared, 2))


    # plotModel_power(testData_powerLux, model='Light GBM Model', save=[True, "early_powerLux_LGBM.png"])
    # tcnn_metrics_early.loc['powerLux', :] = temporaL_CNN(trainData_powerLux, testData_powerLux, trainLabels_powerLux, testLabels_powerLux, trainFeatures_powerLux, testFeatures_powerLux, metrics=True)
    print("================= METRICS ========================")

    # print("LSTM metrics")
    # print(lstm_metrics_early)

    # print("Temporal CNN metrics")
    # print(tcnn_metrics_early)

    # print("Light GBM metrics")
    # print(lightGBM_metrics_early)

