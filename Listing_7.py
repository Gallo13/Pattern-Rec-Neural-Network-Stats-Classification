# Jessica Gallo
# Created: 2/12/2020
# Last modified: 2/15/2020
# Pattern Recognition & Neural Networks
# Descriptive Statistics, Classification and Analysis Using Python & Python Libraries
# Part 2
# Listing 7

# ======================================================================================================================

from pandas import read_csv  # Rescaling/Standardize/Normalize
from numpy import set_printoptions  # Rescaling/Standardize/Normalize/Binarize
from sklearn.preprocessing import MinMaxScaler, Binarizer  # Rescaling
from sklearn.preprocessing import StandardScaler  # Standardize
from sklearn.preprocessing import Normalizer  # Normalize

# Dataset
filename = 'data_banknote_authentication.txt'
names = ['variance', 'skewness', 'curtosis', 'entropy', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values
# separate array into input and output components
X = array[:,0:8]
Y = array[:,4]

# ---------------
# Rescaling Data |
# ---------------
scaler = MinMaxScaler(feature_range=(0,1))  # desired range of transformed data
rescaledX = scaler.fit_transform(X)  # fit to data, then transform it
set_printoptions(precision=3)  # sets number of digits
print("\nRescaling Data between 0 and 1 from Banknote Authentication Dataset:")
print(rescaledX[0:5,:])

# -----------------
# Standardize Data |
# -----------------
scaler = StandardScaler().fit(X)  # standardize features by removing the mean and scaling to unit variance
rescaledX = scaler.transform(X)
set_printoptions(precision=3)  # sets number of digits
print("\nStandardize Data from Banknote Authentication Dataset:")
print(rescaledX[0:5,:])

# ---------------
# Normalize Data |
# ---------------
scaler = Normalizer().fit(X)  # normalize samples individually to unit norm
normalizedX = scaler.transform(X)
set_printoptions(precision=3)  # sets number of digits
print("\nNormalized Data from Banknote Authentication Dataset:")
print(normalizedX[0:5,:])

# --------------
# Binarize Data |
# --------------
binarizer = Binarizer(threshold=0.0).fit(X)  # binarize data according to threshold
# threshold: feature values below or equal to this are replaced by 0, above it by 1
# fit is there to implement the usual API
binaryX = binarizer.transform(X)
set_printoptions(precision=3)  # sets number of digits
print("\nBinarized Data from Banknote Authentication Dataset:")
print(binaryX[0:5,:])
