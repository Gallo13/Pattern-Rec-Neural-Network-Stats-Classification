# CSC 732
# Jessica Gallo
# Created: 2/12/2020
# Last modified: 2/27/2020
# Pattern Recognition & Neural Networks
# Descriptive Statistics, Classification and Analysis Using Python & Python Libraries
# Part 2

# imports for Listing 6
# from pandas import read_csv  # Pairwise Pearson Corr/Skew/Histograms/Density/Box & Whisker/Correlation Matrix
from pandas import set_option  # Pairwise Pearson Corr
from matplotlib import pyplot  # Histograms/Density/Box & Whisker/Correlation Matrix
import numpy  # Correlation Matrix

# imports for Listing 7
# from pandas import read_csv  # Rescaling/Standardize/Normalize
from numpy import set_printoptions  # Rescaling/Standardize/Normalize/Binarize
from sklearn.preprocessing import MinMaxScaler, Binarizer  # Rescaling
from sklearn.preprocessing import StandardScaler  # Standardize
from sklearn.preprocessing import Normalizer  # Normalize
import pandas as pd

# Dataset for both Listings 6 & 7
filename = "data_banknote_authentication.txt"  # csv/text file
names = ['variance', 'skewness', 'curtosis', 'entropy', 'class']  # columns on dataset
# data = read_csv(filename, names=names)  # reads the info on csv/txt file
dataSetCsv = pd.read_csv(filename, ',', error_bad_lines=False, names=names)
dataset = pd.DataFrame(dataSetCsv)

# Listing 6
# ##############################################################################
# -----------------------------
# Pairwise Pearson Correlation |
# -----------------------------
set_option('display.width', 100)  # width of the display in characters
set_option('precision', 3)  # sets number of digits
correlations = dataset.corr(method='pearson')  # corr() function to calculate correlation matrix
print("Pairwise Pearson Correlction of Bank Authentication Dataset:\n")
print(correlations)

# ------------------------
# Skew for Each Attribute |
# ------------------------
skew = dataset.skew()  # calculates skew of dataset
print("\nSkewness for Each Attribute of Bank Authentication Dataset:")
print(skew)

# -------------------------
# Univariable Density Plot |
# -------------------------
# Histogram
# =========
dataset.hist()  # calculates histogram
print("\nUnivariable Histogram of Bank Authentication Dataset:")
pyplot.show()  # displays the plot

# Density Plots
# =============
dataset.plot(kind='density', subplots=True, layout=(3,3), sharex=False)
# subplots True: make separate subplot for each column
# sharex False: an ax is passed in
print("\nUnivariable Density Plots of Bank Authentication Dataset:")
pyplot.show()  # displays the plot

# Box and Whisker Plots
# =====================
dataset.plot(kind='box', subplots=True, layout=(3,3), sharex=False)
print("\nUnivariable Box and Whisker Plots of Bank Authentication Dataset:")
pyplot.show()  # displays plot

# ------------------------
# Correlation Matrix Plot |
# ------------------------
fig = pyplot.figure()
ax = fig.add_subplot(111)  # subplot grid parameters 1x1 grid, 1st subplot
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)  # creates colorbar on axes
ticks = numpy.arange(0,5,1)  # returns evenly spaced values within a given interval
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
print("\nCorrelation Matrix Plot for Banknote Authentication Dataset:")
pyplot.show()  # displays plot

# Listing 7
# ##############################################################################
# Listing 7 data
array = dataset.values
# separate array into input and output components
X = array[:,0:4]
Y = array[:,4]

# ---------------
# Rescaling Data |
# ---------------
scaler = MinMaxScaler(feature_range=(0,1))  # desired range of transformed data
rescaledX = scaler.fit_transform(X)  # fit to data, then transform it
set_printoptions(precision=3)  # sets number of digits
print("\nRescaling Data between 0 and 1 from Banknote Authentication Dataset:")
print(rescaledX[0:2,:])

# -----------------
# Standardize Data |
# -----------------
scaler = StandardScaler().fit(X)  # standardize features by removing the mean and scaling to unit variance
rescaledX = scaler.transform(X)
set_printoptions(precision=3)  # sets number of digits
print("\nStandardize Data from Banknote Authentication Dataset:")
print(rescaledX[0:2,:])

# ---------------
# Normalize Data |
# ---------------
scaler = Normalizer().fit(X)  # normalize samples individually to unit norm
normalizedX = scaler.transform(X)
set_printoptions(precision=3)  # sets number of digits
print("\nNormalized Data from Banknote Authentication Dataset:")
print(normalizedX[0:2,:])

# --------------
# Binarize Data |
# --------------
binarizer = Binarizer(threshold=0.0).fit(X)  # binarize data according to threshold
# threshold: feature values below or equal to this are replaced by 0, above it by 1
# fit is there to implement the usual API
binaryX = binarizer.transform(X)
set_printoptions(precision=3)  # sets number of digits
print("\nBinarized Data from Banknote Authentication Dataset:")
print(binaryX[0:2,:])
