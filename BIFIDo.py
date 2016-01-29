
__author__ = "Sean Gibbons"
__copyright__ = "Copyright 2016"
__credits__ = ["Sean Gibbons; Chris Smillie; Sean Kearny; Eric Alm"]
__license__ = "GPL"
__version__ = "1.0.0-dev"
__maintainer__ = "Sean Gibbons"
__email__ = "sgibbons@mit.edu"

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import argparse
from pykalman import KalmanFilter


###imput arguments
parser = argparse.ArgumentParser()
parser.add_argument('-i', help='input file (rows = samples, columns = OTUs)', required=True)
parser.add_argument('-t', help='input data type', default='counts', choices=['counts', 'norm', 'log'])
parser.add_argument('-a', help='min coefficient', default=.1, type=float)
parser.add_argument('-b', help='max coefficient', default=10, type=float)
parser.add_argument('-d', help='field delimiter', default='\t')
parser.add_argument('-o', help='output prefix ', default='out')
args = parser.parse_args()

# read data - make sure sample names are converted to Pandas-interpretable datatime format (e.g. YYYY-MM-DD)
x = pd.DataFrame.from_csv(args.i, sep=args.d, header=0, index_col=0, infer_datetime_format=True)
# uncomment transpose if OTUs are rows and samples are columns
#x = x.transpose()

if args.t == 'counts':
    x = 1.*x.divide(x.sum(axis=1), axis=0)
if args.t == 'log':
    x = np.exp(x)
nrowx, ncolx = np.shape(x)

# check data
if nrowx == 0 or ncolx == 0:
    quit('error: input file format (check delimiter)')
if x.min().min() < 0 or x.max().max() > 1:
    quit('error: input file format (check data type)')

print "Data successfully loaded!"

# fill missing dates with 'nan'
x_nans = x.resample('D')

print "Gaps in time series filled with NaNs!"

# use numpy mask for nans - required for pykalman
x_nans_masked = np.ma.masked_invalid(np.array(x_nans))

# initialize dataframe for kalman-smoothed data
x_int_ks = pd.DataFrame(np.zeros((x_nans_masked.shape[0],x_nans_masked.shape[1])),index=x_nans.index,columns=x_nans.columns)

# initialize kalman filter function for 1-D time series
kf = KalmanFilter(n_dim_obs=1)

# run kalman smoother - estimate parameters for each OTU (5 iterations)
for i in range(x_nans_masked.shape[1]):
    (filtered_state_means, filtered_state_covariances) = kf.em(x_nans_masked[:,i], n_iter=5).smooth(x_nans_masked[:,i])
    for x in range(x_nans_masked.shape[0]):
        x_int_ks.iloc[x,i] = np.array(filtered_state_means).flatten()[x]

print "Kalman smoothing and interpolation successful!"


#log-transform kalman smoothed data
x_int_ks_log = x_int_ks.apply(np.log)

print "Log transform successful!"


#initialize dataframe for first-differenced data
x_int_ks_log_delta = x_int_ks_log.iloc[:-1,:]

#calculate delta
for i in range(x_int_ks_log.shape[1]):
    log_delta_list = []
    for x in range(x_int_ks_log.shape[0] - 1):
        log_delta = x_int_ks_log.iloc[x+1,i] - x_int_ks_log.iloc[x,i]
        log_delta_list.append(log_delta)
    x_int_ks_log_delta[x_int_ks_log.columns[i]] = np.transpose(log_delta_list)

print "First differences successfully calculated!"


nan_file_handle = '%s_nan.txt' % args.o
ks_file_handle = '%s_ks.txt' % args.o
ks_log_file_handle = '%s_ks_log.txt' % args.o
ks_log_delta_file_handle = '%s_ks_log_delta.txt' % args.o 


x_nans.to_csv(nan_file_handle, sep='\t')
x_int_ks.to_csv(ks_file_handle, sep='\t')
x_int_ks_log.to_csv(ks_log_file_handle, sep='\t')
x_int_ks_log_delta.to_csv(ks_log_delta_file_handle, sep='\t')


print "Data written to file!"

