import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import csv

from scipy import interpolate
from scipy import signal
from Track import Track
import tracks
from spline import *
from linewidthhelper import *

plt.rcParams.update({'font.size': 10})

track = tracks.Monaco
file = '4wdactive_monaco.csv'

def getCornerDistances(track):
	dist = 0
	turnNum = 1
	cornerEntries = []
	cornerExits = []
	for i in range(len(track.segments)):
		segment = track.segments[i]
		if segment[0] == 0:
			#straight
			dist = dist+segment[1]
		else:
			#corner
			cornerEntries.append(dist)
			#print('Turn ',str(turnNum),' entry: ',str(dist),' m')
			dist = dist+segment[1]*segment[2]
			#print('Turn ',str(turnNum),' exit: ',str(dist),' m')
			cornerExits.append(dist)
			print('Turn ',str(turnNum),' midpoint: ',str((cornerEntries[-1]+cornerExits[-1])/2),' m')
			turnNum = turnNum+1

	print('Total distance: ',str(dist))
	return np.array(cornerEntries),np.array(cornerExits)

cornerEntries,cornerExits = getCornerDistances(track)



for i in range(1):

	filename = file[:-4]

	data = np.array([])
	with open(file, newline='') as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		next(reader)
		for row in reader:
			if len(data)==0:
				data = np.array([row]).astype(float)
			else:
				data = np.append(data,np.array([row]).astype(float),axis=0)
			
	fig,axes = plt.subplots(nrows=1,ncols=1,figsize=(6,3),sharex=True)

	s = data[:,0]
	V = data[:,1]
	gamma = data[:,2]
	n = data[:,3]
	t = data[:,4]
	c_fl = data[:,5]
	c_fr = data[:,6]
	c_rl = data[:,7]
	c_rr = data[:,8]
	powerFL = data[:,9]
	powerFR = data[:,10]
	powerRL = data[:,11]
	powerRR = data[:,12]
	delta = data[:,13]

	p_max = np.maximum.reduce([powerFL,powerFR,powerRL,powerRR])/75000
	p_max = np.maximum(p_max,np.zeros(len(p_max)))

	# axes.plot(s,np.maximum.reduce([c_fl,c_rl,c_fr,c_rr]),label=r'$c_{max}$')
	# axes.plot(s,p_max,label=r'$c_{max}$')
	# axes.plot(s,c_fr,label=r'$c_fr$')
	# axes.plot(s,c_rl,label=r'$c_rl$')
	# axes.plot(s,c_rr,label=r'$c_rr$')

	axes.plot(s,gamma)

	axes.set_xlabel('Turn number')
	axes.set_ylabel('Wing angle (deg)')
	
	axes.set_xlim(0,s[-1])
	axes.set_ylim(0,50)
	axes.grid()

	turns = np.array([0,2,5,7,10,15,20])
	turnLabels = np.array([1,4,6,8,10,13,16])

	axes.set_xticks(np.rint((cornerEntries+cornerExits)/2)[turns])
	axes.set_xticklabels(turnLabels)

	axes.tick_params(
		axis='x',          # changes apply to the x-axis
		which='both',      # both major and minor ticks are affected
		bottom=False,      # ticks along the bottom edge are off
		)

	ax3 = axes.twiny()
	ax3.xaxis.set_ticks_position('bottom')
	ax3.xaxis.set_label_position('bottom')
	ax3.spines['bottom'].set_position(("axes",-0.25))
	ax3.set_frame_on(True)
	ax3.patch.set_visible(False)
	for sp in ax3.spines.values():
		sp.set_visible(False)
	ax3.spines['bottom'].set_visible(True)

	dist_ticks = np.array([1000,2000,3000])
	ax3.set_xticks(dist_ticks/track.getTotalLength())
	ax3.set_xticklabels(dist_ticks)
	ax3.set_xlabel('Distance along centerline (m)')


	plt.subplots_adjust(hspace=0.10)

	pltstring = filename+'gamma.pdf'
	plt.savefig(pltstring,bbox_inches='tight')
# plt.show()

########### TRACK PLOT ##########

points = getTrackPoints(track)
finespline,gates,gatesd,curv,slope = getSpline(points,s=0.0)
s_final = track.getTotalLength()

trackLength = track.getTotalLength()
normals = getGateNormals(finespline,slope)

# fig, ax = plt.subplots(figsize=(4,4))
# plt.plot(finespline[0],finespline[1],linewidth=0.1,solid_capstyle="butt")

# plt.axis('equal')
# plt.plot(finespline[0],finespline[1],'k',linewidth=linewidth_from_data_units(8.5,ax),solid_capstyle="butt")
# plt.plot(finespline[0],finespline[1],'w',linewidth=linewidth_from_data_units(8,ax),solid_capstyle="butt")
# plt.xlabel('x (m)')
# plt.ylabel('y (m)')


def plotTrack(s,n,finespline,color):
	newgates = []
	newnormals = []
	newn = []
	for i in range(len(n)):
		index = ((s[i]/s_final)*np.array(finespline).shape[1]).astype(int)
		#print(index[0])
		if index==np.array(finespline).shape[1]:
			index = np.array(finespline).shape[1]-1
		if i>0 and s[i] == s[i-1]:
			continue
		else:
			newgates.append([finespline[0][index],finespline[1][index]])
			newnormals.append(normals[index])
			newn.append(n[i])

	newgates = reverseTransformGates(newgates)
	displacedGates = setGateDisplacements(newn,newgates,newnormals)
	displacedGates = np.array((transformGates(displacedGates)))

	displacedSpline,gates,gatesd,curv,slope = getSpline(displacedGates,0.0005,0)

	s_new = np.linspace(0,s_final,2000)

	#plot spline with color
	for i in range(1,len(displacedSpline[0])):
		# index = ((s_new[i]/s_final)*np.array(finespline).shape[1]).astype(int)
		s_spline = s_new[i]
		index_greater = np.argwhere(s>=s_spline)[0][0]
		index_less = np.argwhere(s<s_spline)[-1][0]

		point = [displacedSpline[0][i],displacedSpline[1][i]]
		prevpoint = [displacedSpline[0][i-1],displacedSpline[1][i-1]]
		if i <=5 or i == len(displacedSpline[0])-1:
			plt.plot([point[0],prevpoint[0]],[point[1],prevpoint[1]],color=color,linewidth=linewidth_from_data_units(1.5,ax),solid_capstyle="butt",antialiased=True)
		else:
			plt.plot([point[0],prevpoint[0]],[point[1],prevpoint[1]],color=color,linewidth=linewidth_from_data_units(1.5,ax),solid_capstyle="projecting",antialiased=True)

def plotTrackWithData(s,n,finespline,state):
	newgates = []
	newnormals = []
	newn = []
	for i in range(len(n)):
		index = ((s[i]/s_final)*np.array(finespline).shape[1]).astype(int)
		#print(index[0])
		if index==np.array(finespline).shape[1]:
			index = np.array(finespline).shape[1]-1
		if i>0 and s[i] == s[i-1]:
			continue
		else:
			newgates.append([finespline[0][index],finespline[1][index]])
			newnormals.append(normals[index])
			newn.append(n[i])

	newgates = reverseTransformGates(newgates)
	displacedGates = setGateDisplacements(newn,newgates,newnormals)
	displacedGates = np.array((transformGates(displacedGates)))

	displacedSpline,gates,gatesd,curv,slope = getSpline(displacedGates,0.0005,0)
	s_new = np.linspace(0,s_final,2000)

	cmap = mpl.cm.get_cmap('viridis')
	norm = mpl.colors.Normalize(vmin=np.amin(state),vmax=np.amax(state))

	fig, ax = plt.subplots(figsize=(6,3.6))
	plt.plot(displacedSpline[0],displacedSpline[1],linewidth=0.1,solid_capstyle="butt")

	plt.axis('equal')
	plt.plot(finespline[0],finespline[1],'k',linewidth=linewidth_from_data_units(8.5,ax),solid_capstyle="butt")
	plt.plot(finespline[0],finespline[1],'w',linewidth=linewidth_from_data_units(8,ax),solid_capstyle="butt")
	plt.xlabel('x (m)')
	plt.ylabel('y (m)')

	#plot spline with color
	for i in range(1,len(displacedSpline[0])):
		# index = ((s_new[i]/s_final)*np.array(finespline).shape[1]).astype(int)
		s_spline = s_new[i]
		index_greater = np.argwhere(s>=s_spline)[0][0]
		index_less = np.argwhere(s<s_spline)[-1][0]

		x = s_spline
		xp = np.array([s[index_less],s[index_greater]])
		fp = np.array([state[index_less],state[index_greater]])
		interp_state = np.interp(x,xp,fp)

		#print(index_less,index_greater,s[index_greater],s[index_less],s_spline,interp_state,fp[0],fp[1])
		state_color = norm(interp_state)
		color = cmap(state_color)
		color = mpl.colors.to_hex(color)
		point = [displacedSpline[0][i],displacedSpline[1][i]]
		prevpoint = [displacedSpline[0][i-1],displacedSpline[1][i-1]]
		if i <=5 or i == len(displacedSpline[0])-1:
			plt.plot([point[0],prevpoint[0]],[point[1],prevpoint[1]],color,linewidth=linewidth_from_data_units(1.5,ax),solid_capstyle="butt",antialiased=True)
		else:
			plt.plot([point[0],prevpoint[0]],[point[1],prevpoint[1]],color,linewidth=linewidth_from_data_units(1.5,ax),solid_capstyle="projecting",antialiased=True)
	clb = plt.colorbar(mpl.cm.ScalarMappable(norm=norm,cmap=cmap),fraction = 0.02, pad=0.04)
	# if np.array_equal(state,V[:,0]):
	# 	clb.set_label('Velocity (m/s)')
	# elif np.array_equal(state,thrustRL[:,0]):
	# 	clb.set_label('Thrust')
	# elif np.array_equal(state,delta[:,0]):
	# 	clb.set_label('Delta')
	# elif np.array_equal(state,ClA[:,0]):
	# 	clb.set_label('Wing angle')
	clb.set_label('Velocity (m/s)')
	plt.tight_layout()
	plt.grid()

plotTrackWithData(s,n,finespline,V)
# plotTrackWithData(s2,n2,finespline,'tab:orange')
# plt.tight_layout()
# plt.grid()
# plt.xlabel('x (m)')
# plt.ylabel('y (m)')

pltstring = filename+'trackV.pdf'
plt.savefig(pltstring,bbox_inches='tight')
plt.show()


