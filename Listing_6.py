# Jessica Gallo
# Created: 2/12/2020
# Last modified: 2/15/2020
# Pattern Recognition & Neural Networks
# Descriptive Statistics, Classification and Analysis Using Python & Python Libraries
# Part 2
# Listing 6

# ======================================================================================================================

from pandas import read_csv  # Pairwise Pearson Corr/Skew/Histograms/Density/Box & Whisker/Correlation Matrix
from pandas import set_option  # Pairwise Pearson Corr
from matplotlib import pyplot  # Histograms/Density/Box & Whisker/Correlation Matrix
import numpy  # Correlation Matrix

# Dataset
filename = "data_banknote_authentication.txt"  # csv/text file
names = ['variance', 'skewness', 'curtosis', 'entropy', 'class']  # columns on dataset
data = read_csv(filename, names=names)  # reads the info on csv/txt file

# -----------------------------
# Pairwise Pearson Correlation |
# -----------------------------
set_option('display.width', 100)  # width of the display in characters
set_option('precision', 3)  # sets number of digits
correlations = data.corr(method='pearson')  # corr() function to calculate correlation matrix
print("Pairwise Pearson Correlction of Bank Authentication Dataset:")
print(correlations)

# ------------------------
# Skew for Each Attribute |
# ------------------------
skew = data.skew()  # calculates skew of dataset
print("\nSkewness for Each Attribute of Bank Authentication Dataset:")
print(skew)

# -------------------------
# Univariable Density Plot |
# -------------------------
# Histogram
# =========
data.hist()  # calculates histogram
print("\nUnivariable Histogram of Bank Authentication Dataset:")
pyplot.show()  # displays the plot

# Density Plots
# =============
data.plot(kind='density', subplots=True, layout=(3,3), sharex=False)
# subplots True: make separate subplot for each column
# sharex False: an ax is passed in
print("\nUnivariable Density Plots of Bank Authentication Dataset:")
pyplot.show()  # displays the plot

# Box and Whisker Plots
# =====================
data.plot(kind='box', subplots=True, layout=(3,3), sharex=False)
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
