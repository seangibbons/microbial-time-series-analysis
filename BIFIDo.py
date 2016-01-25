
__author__ = "Sean Gibbons"
__copyright__ = "Copyright 2016"
__credits__ = ["Sean Gibbons; Chris Smillie; Eric Alm; Sean Kearny"]
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


###imput arguments
parser.add_argument('-i', help='input file (rows = OTUs, columns = samples)', required=True)
parser.add_argument('-t', help='input data type', default='norm', choices=['counts', 'norm', 'log'])
parser.add_argument('-o', help='output prefix ', default='out')

