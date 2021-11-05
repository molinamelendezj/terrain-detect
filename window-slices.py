
from glob import glob
# First number = subject
# Second number = run

# Stockpile the data

x_stock, x_time_stock, y_stock, y_time_stock = [], [], [], []
for subject in range(1,10):
    x_stock.append(sorted(glob(f'/pfs/iot-readings/TrainingData/subject_00{subject}_0*__x.csv')))
    x_time_stock.append(sorted(glob(f'/pfs/iot-readings/TrainingData/subject_00{subject}_0*__x_time.csv')))
    y_stock.append(sorted(glob(f'/pfs/iot-readings/TrainingData/subject_00{subject}_0*__y.csv')))
    y_time_stock.append(sorted(glob(f'/pfs/iot-readings/TrainingData/subject_00{subject}_0*__y_time.csv')))
'''for subject in range(3):
    x_stock.append(sorted(glob(f'/pfs/iot-readings/TrainingData/subject_01{subject}_0*__x.csv')))
    x_time_stock.append(sorted(glob(f'/pfs/iot-readings/TrainingData/TestData/subject_01{subject}_0*__x_time.csv')))
    y_time_stock.append(sorted(glob(f'/pfs/iot-readings/TrainingData/TestData/subject_01{subject}_0*__y_time.csv')))'''
y_stock.pop()    # delete last row (it's empty)

# Flatten the stockpiles (2D to 1D)
x_stock = [item for sublist in x_stock for item in sublist]
x_time_stock = [item for sublist in x_time_stock for item in sublist]
y_stock = [item for sublist in y_stock for item in sublist]
y_time_stock = [item for sublist in y_time_stock for item in sublist]

print(x_stock)
