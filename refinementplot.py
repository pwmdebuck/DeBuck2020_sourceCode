import numpy as np
import matplotlib.pyplot as plt

segments = np.array([10, 20,30,50,75,100,125,150,175,200,225,250,300,400,500,700,1000,1300])
time = np.array([137.648,146.61,122.539,126.804,118.942,120.708,121.607,120.799,119.92,120.14,119.672,119.556,119.584,119.511,119.46,119.455,119.476,119.505])
trackLength = 4785.32
segmentLength = trackLength/segments
resolution = segmentLength/2
print(resolution)

fig, ax = plt.subplots(figsize=(6,3))
plt.semilogx(np.flip(resolution),np.flip(time),'-',marker='o',markersize=4)
plt.xlim(1,300)
ax.invert_xaxis()
plt.ylabel('Optimal lap time (s)')
plt.xlabel('Grid resolution (m)')
plt.savefig('refinement.pdf',bbox_inches='tight')
plt.show()