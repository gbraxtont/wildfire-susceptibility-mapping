import clean
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import csv
def firePCA():
	# Read in the data
	y, x = clean.clean()

	# Scale data between 0 and 1
	scaler = MinMaxScaler()
	data = scaler.fit_transform(x)

	# Fit data based on 95% expected variance
	pca = PCA(n_components = 0.95)
	# Apply transformation and reduce components
	fitted_var = pca.fit(data)
	a = pca.explained_variance_ratio_
	explained_var_matrix = np.diag(a)

	#for i in explained_var_matrix:
	#	print(i, "\n")
	# print(explained_var_matrix)

	reduced = pca.fit_transform(data)

	# print(reduced)

	# with open("transformed.csv", mode = "w") as myFile:
	# 	csv_writer = csv.writer("transformed.csv", delimeter = ",")



	# myFile = open("transformed.csv", "w")
	# csv_writer = csv.writer("transformed.csv", delimeter = ",")
	# for row in reduced:
	# 	for item in row:
	# 		csv_writer.writerow(str(item))
	# myFile.close()



	# Visualization
	#fig, ax = plt.subplots()
	# Point for each component
	#xi = np.arange(1, 9, step=1)
	# Cumlative variance for y axis
	#y = np.cumsum(pca.explained_variance_ratio_)

	# Standard between 0 and 1
	#plt.ylim(0.0,1.1)

	# Plot
	#plt.plot(xi, y, marker='X', linestyle='--', color='g')
	#plt.xlabel('Number of Components')
	#plt.xticks(np.arange(0, 13, step=1))
	#plt.ylabel("Cumulative Expected Variance")
	#plt.title("Components Needed to Explain Variance")

	#plt.axhline(y=0.95, color='r', linestyle='-')
	#plt.text(0.5, 0.85, '95% cut-off', color = 'red', fontsize=10)

	#ax.grid(axis='x')
	#plt.show()

	return reduced


