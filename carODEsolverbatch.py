import numpy as np
import openmdao.api as om
import dymos as dm
import matplotlib.pyplot as plt
from combinedODE import CombinedODE
import matplotlib as mpl
import csv

#track curvature imports
from scipy import interpolate
from scipy import signal
from Track import Track
import tracks
from spline import *
from linewidthhelper import *

from runDymosProblem import runDymosProblem

track = tracks.Barcelona
points = getTrackPoints(track)
finespline,gates,gatesd,curv,slope = getSpline(points,s=0.0)
s_final = track.getTotalLength()


segment_array = np.array([800])

for i in range(len(segment_array)):
	for k in range(2,3):
		if k == 0:
			tol = 1e-2
		elif k == 1:
			tol = 1e-4
		else:
			tol = 1e-6
		for j in range(1,2):
			if j == 0:
				solver = 'IPOPT'
			else:
				solver = 'SNOPT'
			print(segment_array[i],solver,tol)
			res,duration = runDymosProblem(track,int(segment_array[i]),solver,tol)
			print(res,duration)
			with open('results.csv','a') as file:
				writer = csv.writer(file,delimiter=',')
				writer.writerow(np.array([segment_array[i],solver,tol,res,duration]).astype('str'))
		

