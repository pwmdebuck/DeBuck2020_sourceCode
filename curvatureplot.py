import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy import signal
from Track import Track
import tracks

from spline import *
import time

track = tracks.Barcelona

points = getTrackPoints(track)
print(track.getTotalLength())
finespline,gates,gatesd,curv,slope = getSpline(points)

plt.plot(finespline[0],finespline[1])
plt.axis('equal')

normals = getGateNormals(finespline,slope)
gateDisplacements = 4*np.ones(np.array(finespline).shape[1])

newgates = setGateDisplacements(gateDisplacements,finespline,normals)

newgates = np.array(transformGates(newgates))
print(newgates.shape)
print(newgates)
plt.plot(newgates[:,0],newgates[:,1])
plt.show()
# finespline,gates,gatesd,curv,slope = getSpline(newgates)

# plt.plot(finespline[0],finespline[1])

# plt.show()