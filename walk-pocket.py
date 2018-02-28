'''
         Computes the number of steps a person took while walking in a certain direction. You can also plot graphs
         to visualize data. Peak detection is used to calculate the number of steps taken. Moving Average is applied
         to smooth the data though you can apply savgol filter also. Each column is stored in a list to plot the graph.

         Inputs:
                 The input is a CSV file which contains the data - accelerometer readings
 
         Outputs:
                The number of steps.
'''

# import files
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import csv
import matplotlib.pyplot as plt
import numpy
import pylab
import pandas as pd
import peakutils

# Open the desired file for reading
f = open('walk-pocket.csv', "rb")

# create a object of csv class and read the file
# use ',' as a delimiter 
reader = csv.reader(f, delimiter=',')

time_row = 0
#accel_1_row = 0
#accel_2_row = 0
#accel_3_row = 0

time = []
accel_1 = []
accel_2 = []
accel_3 = []

# create a list of 'Time in ms', Accelrometer reading 1, Accelerometer reading 2, Accelerometer reading 3
for row in reader:
# Skip the first row
	time_row = time_row + 1
	if time_row == 1:
		continue
	time.append(row[0])
	accel_1.append(row[1])
	accel_2.append(row[2])
	accel_3.append(row[3])

# print the contents of the list
#print time
#plt.plot(accel_1)
#plt.show()
#print accel_2
#print accel_3

# close the file
f.close()
# append all the list accelerometer list together
#final_accel = []
#final_accel.append(accel_1)
#final_accel.append(accel_2)
#final_accel.append(accel_3)

# pre processing - data smoothing
#accel_1 = savgol_filter(accel_1, window_length=3, polyorder=0)
#accel_2 = savgol_filter(accel_2, window_length=3, polyorder=0)
#accel_3 = savgol_filter(accel_3, window_length=3, polyorder=0)
#accel_1 = pd.rolling_mean(accel_1, window=3, center=True)
def moving_average(value, window):
	weight = numpy.repeat(1.0, window)/window
	avg = numpy.convolve(value, weight, 'valid')
	return avg

accel_1 = map(float,accel_1)
accel_2 = map(float,accel_2)
accel_3 = map(float,accel_3)
# use moving average to smooth the data	
accel_1 = moving_average(accel_1, 16)
accel_2 = moving_average(accel_2, 16)
accel_3 = moving_average(accel_3, 16)

#plt.plot(accel_3)
#plt.show()

# Calculate the threshold
thrld_1 = (reduce(lambda x , y: x + y, accel_1))/len(accel_1)
thrld_1 = round(thrld_1)
#print thrld_1

thrld_2 = (reduce(lambda x , y: x + y, accel_2))/len(accel_2)
thrld_2 = round(thrld_2)

thrld_3 = (reduce(lambda x , y: x + y, accel_3))/len(accel_3)
thrld_3 = round(thrld_3)


# get the peaks, using a threshold of thrld_1 and a distance between peaks of 60
indexes_1 = peakutils.peak.indexes(numpy.array(accel_1),thres=thrld_1/max(accel_1), min_dist=58)

# get the peaks, using a threshold of thrld_2 and a distance between peaks of 68
indexes_2 = peakutils.peak.indexes(numpy.array(accel_2),thres=thrld_2/max(accel_2), min_dist=67)

# get the peaks, using a threshold of thrld_3 and a distance between peaks of 55
indexes_3 = peakutils.peak.indexes(numpy.array(accel_3),thres=thrld_3/max(accel_3), min_dist=55)

sum = len((indexes_1)) + len((indexes_2)) + len((indexes_3))

#print('Steps in First Accelerometer reading are: %d' % len((indexes_1)))
#print('Steps in Second Accelerometer reading are: %d' % len((indexes_2)))
print('Steps are: %d' % len((indexes_3)))

#print('Total Steps are: %d' % sum )
