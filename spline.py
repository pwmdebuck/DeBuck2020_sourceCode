import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy import signal
from Track import Track
import tracks
import copy

points = [(3.28,0.00),(4.00,0.50),(4.40,1.0),(4.60,1.52),(5.00,2.5),(5.00,3.34),(4.70,3.8),(4.50,3.96),(4.20,4.0),(3.70,3.90),(3.00,3.5),(2.00,2.9)]
data = np.array(points)
track = tracks.Spa

def getTrackPoints(track):
	pos = np.array([0,0])
	direction = np.array([1,0])

	points = [[0,0]]

	for i in range(len(track.segments)):
	# for i in range(3):
		# points.append([pos[0], pos[1]])
		radius = track.getSegmentRadius(i)
		length = track.getSegmentLength(i)
		if radius==0:
			#on a straight
			endpoint = pos+direction*length

			for j in range(1,length.astype(int)-1):
				if j%5==0:
					points.append(pos+direction*j)
			# if length>400:
			# 	points.append(pos+direction*(length/6))
			# 	points.append(pos+direction*(2*length/6))
			# 	points.append(pos+direction*(3*length/6))
			# 	points.append(pos+direction*(4*length/6))
			# 	points.append(pos+direction*(5*length/6))
			# if length>250:
			# 	points.append(pos+direction*(length/4))
			# 	points.append(pos+direction*(2*length/4))
			# 	points.append(pos+direction*(3*length/4))
			# 	points.append(pos+direction*(15*length/16))
			# elif length>50:
			# 	points.append(pos+direction*(length/2))
			# 	points.append(pos+direction*(15*length/16))
			pos = endpoint
		else:
			#corner
			#length is sweep in radians
			side = track.getCornerDirection(i)
			if side == 0:
				normal = np.array([-direction[1],direction[0]])
			else:
				normal = np.array([direction[1],-direction[0]])

			xc = pos[0]+radius*normal[0]
			yc = pos[1]+radius*normal[1]
			theta_line = np.arctan2(direction[1],direction[0])
			theta_0 = np.arctan2(pos[1]-yc,pos[0]-xc)
			if side == 0:
				theta_end = theta_0+length
				direction = np.array([np.cos(theta_line+length),np.sin(theta_line+length)])
			else:
				theta_end = theta_0-length
				direction = np.array([np.cos(theta_line-length),np.sin(theta_line-length)])
			theta_vector = np.linspace(theta_0, theta_end, 100)

			x,y = parametric_circle(theta_vector, xc, yc, radius)
			# points.append([x[0],y[0]])
			# # if length>2.0:
			for j in range(len(x)):
				if j%10 == 0:
					points.append([x[j],y[j]])
			# #if length>0.6:
			# # 	points.append([x[25],y[25]])
			# # 	points.append([x[50],y[50]])
			# # 	points.append([x[75],y[j]])
			# #else:
			# points.append([x[50],y[50]])
			# # points.append([x[-1],y[-1]])
			
			pos = np.array([x[-1],y[-1]])
			
	# plt.axis('equal')
	# points.append([0,0])
	return np.array(points)

def parametric_circle(t,xc,yc,R):
    x = xc + R*np.cos(t)
    y = yc + R*np.sin(t)
    return x,y

points = getTrackPoints(track)
data = np.array(points)
# for i in range(len(points)):
# 	plt.plot(points[i][0],points[i][1],'o')
# plt.show()

tck,u = interpolate.splprep(data.transpose(), s=0, k=3)
unew = np.arange(0, 1.0, 0.0001)
out = interpolate.splev(unew, tck)

gates = interpolate.splev(u, tck)
gatesd = interpolate.splev(u, tck, der = 1)


def getSpline(points,interval=0.00001,s=0.3):
	tck,u = interpolate.splprep(points.transpose(),s=s, k=5)
	unew = np.arange(0, 1.0, interval)
	finespline = interpolate.splev(unew, tck)

	gates = interpolate.splev(u, tck)
	gatesd = interpolate.splev(u, tck, der = 1)

	single = interpolate.splev(unew, tck,der=1)
	double = interpolate.splev(unew, tck,der=2)
	curv = (single[0]*double[1]-single[1]*double[0])/(single[0]**2+single[1]**2)**(3/2)

	return finespline,gates,gatesd,curv,single

def getSplineLength(spline):
	length = 0
	for i in range(1,len(spline[0])):
		prevpoint = [spline[0][i-1],spline[1][i-1]]
		currpoint = [spline[0][i],spline[1][i]]
		dx = np.sqrt((currpoint[0]-prevpoint[0])**2+(currpoint[1]-prevpoint[1])**2)
		length = length+dx
	return length


def getGateNormals(gates,gatesd):
	normals = []
	for i in range(len(gates[0])):
		der = [gatesd[0][i],gatesd[1][i]]
		mag = np.sqrt(der[0]**2+der[1]**2)
		normal1 = [-der[1]/mag,der[0]/mag]
		normal2 = [der[1]/mag,-der[0]/mag]

		normals.append([normal1,normal2])
		# plt.plot([gates[0][i],gates[0][i]+normal1[0]],[gates[1][i],gates[1][i]+normal1[1]],'g',linewidth=3)
		# plt.plot([gates[0][i],gates[0][i]+normal2[0]],[gates[1][i],gates[1][i]+normal2[1]],'b',linewidth=3)

	return normals

def transformGates(gates):
	#transforms from [[x positions],[y positions]] to [[x0, y0],[x1, y1], etc..]
	newgates = []
	for i in range(len(gates[0])):
		newgates.append(([gates[0][i],gates[1][i]]))
	return newgates

def reverseTransformGates(gates):
	#transforms from [[x0, y0],[x1, y1], etc..] to [[x positions],[y positions]]
	newgates = np.zeros((2,len(gates)))
	for i in range(len(gates)):
		newgates[0,i] = gates[i][0]
		newgates[1,i] = gates[i][1]
	return newgates

def setGateDisplacements(gateDisplacements,gates,normals):
	#does not modify original gates, returns updated version
	newgates = np.copy(gates)
	for i in range(len(gates[0])):
		if i > len(gateDisplacements)-1:
			disp = 0
		else:
			disp = gateDisplacements[i]
		#if disp>0:
		normal = normals[i][0] #always points outwards
		#else:
		#	normal = normals[i][1] #always points inwards
		newgates[0][i] = newgates[0][i] + disp*normal[0]
		newgates[1][i] = newgates[1][i] + disp*normal[1]
	return newgates



# def getCurvature(points):

# 	# curvature = [0]
# 	# for i in range(1,len(points[0])):
# 	# 	point1 = [points[0][i-1],points[1][i-1]]
# 	# 	point2 = [points[0][i],points[1][i]]
# 	# 	if i == len(points[0])-1:
# 	# 		# point3 = [points[0][0],points[1][0]]
# 	# 		curvature.append(curvature[-1])
# 	# 		continue
# 	# 	else:
# 	# 		point3 = [points[0][i+1],points[1][i+1]]

# 	# 	side1 = getLength(point1,point2)
# 	# 	side2 = getLength(point1,point3)
# 	# 	side3 = getLength(point2,point3)

# 	# 	area = getArea(point1,point2,point3)

# 	# 	if side1 == 0 or side2 == 0 or side3 == 0 or area == 0:
# 	# 		curvature.append(0)
# 	# 	else:
# 	# 		curv = 4*area/(side1*side2*side3)
# 	# 		curvature.append(curv)
# 	# return curvature

def getArea(point1,point2,point3):
	area = np.abs(0.5*(point1[0]*point2[1]+point2[0]*point3[1]+point3[0]*point1[1]-point1[1]*point2[0]-point2[1]*point3[0]-point3[1]*point1[0]))
	# area = ((point2[0]-point1[0])*(point3[1]-point1[1]) - (point2[1]-point1[1])*(point3[0]-point1[0]))
	return np.abs(area)

def getLength(point1,point2):
	return np.sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)

def getApex(curvature):
	apex = signal.find_peaks(curvature,height=0.005,distance=40)
	return apex


# curv2 = []
# for i in range(len(curv[0])):
# 	curv2.append(np.sqrt(curv[0][i]**2+curv[1][i]**2))

# fig = plt.figure()
# ax = fig.add_subplot(111)
# curvature = getCurvature(out)
# plt.plot(length_vector,curvature)
# plt.ylim(0,250)
# plt.show()


# apex = apex[0]
# for i in range(len(apex)):
# 	index = apex[i]
# 	plt.plot(length_vector[index],curvature[index],'o')
# # ax.plot(length_vector,curv2)
# # ax.set_ybound(-0.1,0.1)
# plt.show()
# print(getGateNormals(gates,gatesd))

# plt.figure()
# # # plt.ylim((-0.1,0.1))
# # # plt.show()

# plt.plot(out[0], out[1], color='orange')
# plt.plot(gates[0], gates[1],'o', color='blue')
# plt.show()





