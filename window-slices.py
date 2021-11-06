
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from glob import glob
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.optimizers import Adam
from mlxtend.plotting import plot_confusion_matrix

# Plotting
sns.set()

# First number = subject
# Second number = run

# Stockpile the data
with open("/pfs/iot-readings/Data/TrainingData/subject_001_01__x.csv") as f:
    lines = f.read()
    first = lines.split('\n', 1)[0]

print(first)

with open("/pfs/iot-readings/Data/TestData/subject_010_01__x.csv") as f:
    lines = f.read()
    first = lines.split('\n', 1)[0]
    
print(first)

x_stock, x_time_stock, y_stock, y_time_stock = [], [], [], []
for subject in range(1,10):
    x_stock.append(sorted(glob(f'/pfs/iot-readings/Data/TrainingData/subject_00{subject}_0*__x.csv')))
    x_time_stock.append(sorted(glob(f'/pfs/iot-readings/Data/TrainingData/subject_00{subject}_0*__x_time.csv')))
    y_stock.append(sorted(glob(f'/pfs/iot-readings/Data/TrainingData/subject_00{subject}_0*__y.csv')))
    y_time_stock.append(sorted(glob(f'/pfs/iot-readings/Data/TrainingData/subject_00{subject}_0*__y_time.csv')))
for subject in range(3):
    x_stock.append(sorted(glob(f'/pfs/iot-readings/Data/TestData/subject_01{subject}_0*__x.csv')))
    x_time_stock.append(sorted(glob(f'/pfs/iot-readings/Data/TestData/subject_01{subject}_0*__x_time.csv')))
    y_time_stock.append(sorted(glob(f'/pfs/iot-readings/Data/TestData/subject_01{subject}_0*__y_time.csv')))
y_stock.pop()    # delete last row (it's empty)

# Flatten the stockpiles (2D to 1D)
x_stock = [item for sublist in x_stock for item in sublist]
x_time_stock = [item for sublist in x_time_stock for item in sublist]
y_stock = [item for sublist in y_stock for item in sublist]
y_time_stock = [item for sublist in y_time_stock for item in sublist]

print(x_stock)

def window_slicing(df):
    ''' Creating window of 4 sec reading for each prediction.
    At the end we loose 39 or 40 predictions (20 from the front and
    19 or 20 from the end) due to missing readings for the window.
    To mitigate that, inlude zeros in their place later. '''

    number_of_windows = int((len(df) - 160) / 4) + 1    # number of full windows in a sample list
    samples = np.zeros(160*number_of_windows*6).reshape(number_of_windows, 160, 6)    # samples, time steps per sample, features per time step
    start_index = 0     # where to start each window
    for i in range(number_of_windows):
        window = df[start_index:start_index+160]
        samples[i] = window
        start_index += 4    # move the window

    return samples


def cut_labels(x_windows, y):
    ''' Cutting the 20 first and
    19 or 20 last labels to align
    with x_windows. '''

    y = y[20:]   # cut the first 20 labels
    
    end = y.shape[0] - x_windows.shape[0]   # either 19 or 20
    y = y[:-end]    # cut the last 19 or 20 labels

    return y

def read_data_list(x_stock, y_stock):
    ''' Reads in and preprocesses data and keeps it in an array
    as a nested list of subjects. '''

    x_train, x_val, x_test, y_train, y_val = [], [], [], [], []
    counter = 0

    # Read x_train and y_train
    for file in x_stock[:27]:
        x = pd.read_csv(file, names=["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"])   # read a subject
        windows = window_slicing(x)    # cut it to windows
        x_train.append(windows)    # append to list of subjects

        y = pd.read_csv(y_stock[counter], names=["label"])   # read a subject
        cut = cut_labels(windows, y)    # cut the extra labels
        y_train.append(cut)    # append to list of subjects

        counter += 1    # increase the counter
    
    # Read x_val and y_val
    for file in x_stock[27:-4]:
        x = pd.read_csv(file, names=["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"])   # read a subject
        windows = window_slicing(x)    # cut it to windows
        x_val.append(windows)    # append to list of subjects

        y = pd.read_csv(y_stock[counter], names=["label"])   # read a subject
        cut = cut_labels(windows, y)    # cut the extra labels
        y_val.append(cut)    # append to list of subjects

        counter += 1    # increase the counter

    # Read x_test
    for file in x_stock[-4:]:
        x = pd.read_csv(file, names=["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"])   # read a subject
        windows = window_slicing(x)    # cut it to windows
        x_test.append(windows)    # append to list of subjects

    return x_train, x_val, x_test, y_train, y_val


def concat_data(x_train_subj, x_val_subj, x_test_subj, y_train_subj, y_val_subj):
    ''' Concatenates all data into respective
    x_train, x_val, x_test, y_train and y_val arrays. '''

    x_train, x_val, x_test, y_train, y_val = x_train_subj[0], x_val_subj[0], x_test_subj[0], y_train_subj[0], y_val_subj[0]   # add first subjects

    # Concatenate x_train and y_train
    for i in range(1, len(x_train_subj)):
        x_train = np.concatenate((x_train, x_train_subj[i]))
        y_train = np.concatenate((y_train, y_train_subj[i]))

    # Concatenate x_val and y_val
    for i in range(1, len(x_val_subj)):
        x_val = np.concatenate((x_val, x_val_subj[i]))
        y_val = np.concatenate((y_val, y_val_subj[i]))

    # Concatenate x_test
    for i in range(1, len(x_test_subj)):
        x_test = np.concatenate((x_test, x_test_subj[i]))

    return x_train, x_val, x_test, y_train, y_val


# Read data as subjects
x_train_subj, x_val_subj, x_test_subj, y_train_subj, y_val_subj = read_data_list(x_stock, y_stock)

# Concatinate the data
x_train, x_val, x_test, y_train, y_val = concat_data(x_train_subj, x_val_subj, x_test_subj, y_train_subj, y_val_subj)

print(len(x_train))
print(len(y_train))

print(len(x_val))
print(len(y_val))

print(len(x_test))


a_file = open("/pfs/out/xtrain/xtrain.txt", "w")
for row in x_train:
    np.savetxt(a_file, row)

a_file.close()

a_file = open("/pfs/out/ytrain/ytrain.txt", "w")
for row in y_train:
    np.savetxt(a_file, row)

a_file.close()

a_file = open("/pfs/out/xval/xval.txt", "w")
for row in x_val:
    np.savetxt(a_file, row)

a_file.close()


y_val.to_csv('/pfs/out/yval/yval.csv') 


a_file = open("/pfs/out/xtest/xtest.txt", "w")
for row in x_test:
    np.savetxt(a_file, row)

a_file.close()



