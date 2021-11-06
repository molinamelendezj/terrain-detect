
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

x_train = np.loadtxt("/pfs/xtrain/xtrain.txt").reshape(312454, 160, 6)
y_train = np.loadtxt("/pfs/ytrain/ytrain.txt").reshape(312454, 1)
x_val = np.loadtxt("/pfs/xval/xval.txt").reshape(9820, 160, 6)
y_val = pd.read_csv("/pfs/yval/yval.csv")
x_test = np.loadtxt("/pfs/xtest/xtest.txt").reshape(48417, 160, 6)


# Reshape data for standardizing
x_train_reshaped = np.reshape(x_train, (x_train.shape[0], -1))
x_val_reshaped = np.reshape(x_val, (x_val.shape[0], -1))
x_test_reshaped = np.reshape(x_test, (x_test.shape[0], -1))

# Fit Scaler object
scalar = StandardScaler()
scalar.fit(x_train_reshaped)    # fit only the training data

# Standardize
x_train_stand_reshaped = scalar.transform(x_train_reshaped)
x_val_stand_reshaped = scalar.transform(x_val_reshaped)
x_test_stand_reshaped = scalar.transform(x_test_reshaped)

# Reshape back to window shape
x_train_stand = np.reshape(x_train_stand_reshaped, (x_train.shape[0], 160, 6))
x_val_stand = np.reshape(x_val_stand_reshaped, (x_val.shape[0], 160, 6))
x_test_stand = np.reshape(x_test_stand_reshaped, (x_test.shape[0], 160, 6))



def plot_history(history):
    ''' Function for plotting the training and validation history. '''
    
	# plot loss
    plt.title('Loss')
    plt.plot(history.history['loss'], color='blue', label='train')
    plt.plot(history.history['val_loss'], color='red', label='test')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'])
    plt.show()
    
    # plot accuracy
    plt.title('Accuracy')
    plt.plot(history.history['accuracy'], color='blue', label='train')
    plt.plot(history.history['val_accuracy'], color='red', label='test')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'])
    plt.show()


print(f'x_train shape:', x_train.shape)
print(f'x_val shape:', x_val.shape)

print(f'y_train shape:', y_train.shape)
print(f'y_val shape:', y_val.shape)

y_train_1H = to_categorical(y_train, num_classes=4)
y_val_1H = to_categorical(y_val, num_classes=4)




n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[2], y_train_1H.shape[1]

model = Sequential()

model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features)))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(n_outputs, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())


from keras.utils.vis_utils import plot_model
plot_model(model, show_shapes=True, show_layer_names=True)


history = model.fit(x_train_stand, y_train_1H, epochs=10, batch_size=128, validation_data=(x_val_stand, y_val_1H))

plot_history(history)


# Confusion Matrix
pred = model.predict(x_val_stand)
pred = np.argmax(pred, axis = 1)  # transform prediction into 0-3 labels
y_true = y_val

CM = confusion_matrix(y_true, pred)
fig, ax = plot_confusion_matrix(conf_mat=CM ,  figsize=(10, 5))
plt.show()

print(classification_report(y_true, pred))