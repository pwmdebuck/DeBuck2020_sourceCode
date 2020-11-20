import numpy as np
import openmdao.api as om
import dymos as dm
import matplotlib.pyplot as plt
from combinedODE import CombinedODE
import matplotlib as mpl

#track curvature imports
from scipy import interpolate
from scipy import signal
from Track import Track
import tracks
from spline import *
from linewidthhelper import *

track = tracks.Barcelona
points = getTrackPoints(track)
finespline,gates,gatesd,curv,slope = getSpline(points,s=0.0)
s_final = track.getTotalLength()

print('Config: 4WD individual thrust')

# Define the OpenMDAO problem
p = om.Problem(model=om.Group())

# Define a Trajectory object
traj = dm.Trajectory()
p.model.add_subsystem('traj', subsys=traj)

# Define a Dymos Phase object with GaussLobatto Transcription


phase = dm.Phase(ode_class=CombinedODE,
		     transcription=dm.GaussLobatto(num_segments=200, order=3,compressed=True))

traj.add_phase(name='phase0', phase=phase)

# Set the time options
phase.set_time_options(fix_initial=True,fix_duration=True,duration_val=s_final,targets=['curv.s'],units='m',duration_ref=s_final,duration_ref0=10) #50,150,50

#Define states
phase.add_state('t', fix_initial=True, fix_final=False, units='s', lower = 0,rate_source='dt_ds',ref=100)
phase.add_state('n', fix_initial=False, fix_final=False, units='m', upper = 4.0, lower = -4.0, rate_source='dn_ds',targets=['n'],ref=4.0)
phase.add_state('V', fix_initial=False, fix_final=False, units='m/s', ref = 40, ref0=5,rate_source='dV_ds', targets=['V'])
phase.add_state('alpha', fix_initial=False, fix_final=False, units='rad', rate_source='dalpha_ds',targets=['alpha'],ref=0.15)
phase.add_state('lambda', fix_initial=False, fix_final=False, units='rad',rate_source='dlambda_ds',targets=['lambda'],ref=0.01)
phase.add_state('omega', fix_initial=False, fix_final=False, units='rad/s',rate_source='domega_ds',targets=['omega'],ref=0.3)
phase.add_state('ax',fix_initial=False,fix_final=False,units='m/s**2',rate_source='dax_ds',targets=['ax'],ref=8)
phase.add_state('ay',fix_initial=False,fix_final=False,units='m/s**2', rate_source='day_ds',targets=['ay'],ref=8)

#Define Controls
phase.add_control(name='delta', units='rad', lower=None, upper=None,fix_initial=False,fix_final=False, targets=['delta'],ref=0.04)
phase.add_control(name='thrustFL', units=None,fix_initial=False,fix_final=False, targets=['thrustFL'])
phase.add_control(name='thrustFR', units=None,fix_initial=False,fix_final=False, targets=['thrustFR'])
phase.add_control(name='thrustRL', units=None,fix_initial=False,fix_final=False, targets=['thrustRL'])
phase.add_control(name='thrustRR', units=None,fix_initial=False,fix_final=False, targets=['thrustRR'])
phase.add_control(name='gamma',units='deg',lower=0.0,upper=50.0,fix_initial=False,fix_final=False,targets='gamma',ref=50.0)

#Performance Constraints
pmax = 300000/4 #W
phase.add_path_constraint('powerFL',shape=(1,),units='W',upper=pmax,ref=100000)
phase.add_path_constraint('powerFR',shape=(1,),units='W',upper=pmax,ref=100000)
phase.add_path_constraint('powerRL',shape=(1,),units='W',upper=pmax,ref=100000)
phase.add_path_constraint('powerRR',shape=(1,),units='W',upper=pmax,ref=100000)
phase.add_path_constraint('c_rr',shape=(1,),units=None,upper=1)
phase.add_path_constraint('c_rl',shape=(1,),units=None,upper=1)
phase.add_path_constraint('c_fr',shape=(1,),units=None,upper=1)
phase.add_path_constraint('c_fl',shape=(1,),units=None,upper=1)


#Minimize final time.
phase.add_objective('t', loc='final')

#Add output timeseries
phase.add_timeseries_output('lambdadot',units='rad/s',shape=(1,))
phase.add_timeseries_output('Vdot',units='m/s**2',shape=(1,))
phase.add_timeseries_output('alphadot',units='rad/s',shape=(1,))
phase.add_timeseries_output('omegadot',units='rad/s**2',shape=(1,))
phase.add_timeseries_output('powerFL',units='W',shape=(1,))
phase.add_timeseries_output('powerFR',units='W',shape=(1,))
phase.add_timeseries_output('powerRL',units='W',shape=(1,))
phase.add_timeseries_output('powerRR',units='W',shape=(1,))
phase.add_timeseries_output('sdot',units='m/s',shape=(1,))
phase.add_timeseries_output('c_rr',units=None,shape=(1,))
phase.add_timeseries_output('c_fl',units=None,shape=(1,))
phase.add_timeseries_output('c_fr',units=None,shape=(1,))
phase.add_timeseries_output('c_rl',units=None,shape=(1,))
phase.add_timeseries_output('N_rr',units='N',shape=(1,))
phase.add_timeseries_output('N_fr',units='N',shape=(1,))
phase.add_timeseries_output('N_fl',units='N',shape=(1,))
phase.add_timeseries_output('N_rl',units='N',shape=(1,))
phase.add_timeseries_output('curv.kappa',units='1/m',shape=(1,))

traj.link_phases(phases=['phase0', 'phase0'], vars=['V','n','alpha','omega','lambda','ax','ay'], locs=('++', '--'))

# Set the driver.
# p.driver = om.ScipyOptimizeDriver(maxiter=100,optimizer='SLSQP')
p.driver = om.pyOptSparseDriver(optimizer='SNOPT')

p.driver.opt_settings['Scale option'] = 1
p.driver.opt_settings['Minor iterations limit']=100000
p.driver.opt_settings['Major iterations limit']=100000
p.driver.opt_settings['Superbasics limit']=100000
p.driver.opt_settings['Iterations limit']=20000000
p.driver.opt_settings['Major feasibility tolerance'] = 1e-6    # -6 for benchmark
p.driver.opt_settings['Major optimality tolerance'] = 1e-6    # -6 by default
p.driver.opt_settings['Verify level'] = -1  # do not check gradient
p.driver.opt_settings['Function precision'] = 1e-12
p.driver.opt_settings['Linesearch tolerance'] = 0.3  # default 0.9, smaller for more accurate search
p.driver.opt_settings['Major step limit'] = 0.8      # default 2.,

# p.driver.opt_settings['linear_solver'] = 'ma27'
# p.driver.opt_settings['mu_init'] = 1e-3
# p.driver.opt_settings['max_iter'] = 1000
# p.driver.opt_settings['acceptable_tol'] = 1e-3
# p.driver.opt_settings['constr_viol_tol'] = 1e-3
# p.driver.opt_settings['compl_inf_tol'] = 1e-3
# p.driver.opt_settings['acceptable_iter'] = 0
# p.driver.opt_settings['tol'] = 1e-3
# #p.driver.opt_settings['mu_max'] = 10.0
# p.driver.opt_settings['hessian_approximation'] = 'exact'
# p.driver.opt_settings['nlp_scaling_method'] = 'none'
# p.driver.opt_settings['print_level'] = 5
p.driver.hist_file = '4wd.hist'
#p.driver.opt_settings['mu_strategy'] = 'adaptive'

# p.driver.opt_settings['MAXIT'] = 5
# p.driver.opt_settings['ACC'] = 1e-3
# p.driver.opt_settings['IPRINT'] = 0

# p.driver.recording_options['includes'] = ['*']
# p.driver.recording_options['includes'] = ['*']
# p.driver.recording_options['record_objectives'] = True
# p.driver.recording_options['record_constraints'] = True
# p.driver.recording_options['record_desvars'] = True
# p.driver.recording_options['record_inputs'] = True
# p.driver.recording_options['record_outputs'] = True
# p.driver.recording_options['record_residuals'] = True

# recorder = om.SqliteRecorder("cases.sql")
# p.driver.add_recorder(recorder)

# Allow OpenMDAO to automatically determine our sparsity pattern.
# Doing so can significant speed up the execution of Dymos.
p.driver.declare_coloring()

# Setup the problem
p.setup(check=True) #force_alloc_complex=True
# Now that the OpenMDAO problem is setup, we can set the values of the states.

p.set_val('traj.phase0.states:V',phase.interpolate(ys=[30,10], nodes='state_input'),units='m/s')
p.set_val('traj.phase0.states:lambda',phase.interpolate(ys=[0.0,0.0], nodes='state_input'),units='rad')
p.set_val('traj.phase0.states:omega',phase.interpolate(ys=[0.0,0.0], nodes='state_input'),units='rad/s')
p.set_val('traj.phase0.states:alpha',phase.interpolate(ys=[0.0,0.0], nodes='state_input'),units='rad')
p.set_val('traj.phase0.states:ax',phase.interpolate(ys=[0.0,0.0], nodes='state_input'),units='m/s**2')
p.set_val('traj.phase0.states:ay',phase.interpolate(ys=[0.0,0.0], nodes='state_input'),units='m/s**2')
#p.set_val('traj.phase0.states:t',phase.interpolate(ys=[0.0,s_final], nodes='state_input'),units='m')
p.set_val('traj.phase0.states:n',phase.interpolate(ys=[0.0,0.0], nodes='state_input'),units='m')
p.set_val('traj.phase0.states:t',phase.interpolate(ys=[0.0,100.0], nodes='state_input'),units='s')

p.set_val('traj.phase0.controls:delta',phase.interpolate(ys=[0.0,0.0], nodes='control_input'),units='rad')

p.set_val('traj.phase0.controls:thrustFL',phase.interpolate(ys=[0.0, 0.0], nodes='control_input'),units=None)
p.set_val('traj.phase0.controls:thrustFR',phase.interpolate(ys=[0.0, 0.0], nodes='control_input'),units=None)
p.set_val('traj.phase0.controls:thrustRL',phase.interpolate(ys=[0.0, 0.0], nodes='control_input'),units=None)
p.set_val('traj.phase0.controls:thrustRR',phase.interpolate(ys=[0.0, 0.0], nodes='control_input'),units=None)
p.set_val('traj.phase0.controls:gamma',phase.interpolate(ys=[10.0, 10.0], nodes='control_input'),units='deg')


#enable refinement
#p.model.traj.phases.phase0.set_refine_options(refine=True,max_order=3)
#dm.run_problem(p,refine=True,refine_iteration_limit=4)

#p.cleanup()
# cr = om.CaseReader('cases.sql')

# driver_cases = cr.list_cases('driver')

# last_case = cr.get_case(driver_cases[-1])

# objectives = last_case.get_objectives()
# design_vars = last_case.get_design_vars()
# constraints = last_case.get_constraints()

# print(objectives['ob'])

# Run the driver to solve the problem
#check partials
# p.check_partials(show_only_incorrect=True,compact_print=True)
p.run_driver()
#p.run_model()
#p.check_partials(show_only_incorrect=True,compact_print=True,method='cs')

n = p.get_val('traj.phase0.timeseries.states:n')
t = p.get_val('traj.phase0.timeseries.states:t')
print(t[-1])

s = p.get_val('traj.phase0.timeseries.time')
V = p.get_val('traj.phase0.timeseries.states:V')
thrustFL = p.get_val('traj.phase0.timeseries.controls:thrustFL')
thrustFR = p.get_val('traj.phase0.timeseries.controls:thrustFR')
thrustRL = p.get_val('traj.phase0.timeseries.controls:thrustRL')
thrustRR = p.get_val('traj.phase0.timeseries.controls:thrustRR')
delta = p.get_val('traj.phase0.timeseries.controls:delta')
ClA = p.get_val('traj.phase0.timeseries.controls:gamma')
gamma = p.get_val('traj.phase0.timeseries.controls:gamma')
powerFL = p.get_val('traj.phase0.timeseries.powerFL', units='W')
powerFR = p.get_val('traj.phase0.timeseries.powerFR', units='W')
powerRL = p.get_val('traj.phase0.timeseries.powerRL', units='W')
powerRR = p.get_val('traj.phase0.timeseries.powerRR', units='W')

print(np.array(s).shape)
print(np.array(delta).shape)
print(np.array(thrustRL).shape)
print(np.array(ClA).shape)
print(np.array(V).shape)

trackLength = track.getTotalLength()

normals = getGateNormals(finespline,slope)
newgates = []
newnormals = []
newn = []
for i in range(len(n)):
	index = ((s[i]/s_final)*np.array(finespline).shape[1]).astype(int)
	#print(index[0])
	if index[0]==np.array(finespline).shape[1]:
		index[0] = np.array(finespline).shape[1]-1
	if i>0 and s[i] == s[i-1]:
		continue
	else:
		newgates.append([finespline[0][index[0]],finespline[1][index[0]]])
		newnormals.append(normals[index[0]])
		newn.append(n[i][0])

newgates = reverseTransformGates(newgates)
displacedGates = setGateDisplacements(newn,newgates,newnormals)
displacedGates = np.array((transformGates(displacedGates)))

displacedSpline,gates,gatesd,curv,slope = getSpline(displacedGates,0.0005,0)

plt.rcParams.update({'font.size': 10})

# interp = interpolate.splrep(s,delta,k=5,s=0.01)
# s_new = np.linspace(0,s_final,1000)
# delta_interp = interpolate.splev(s_new,interp)

# interp = interpolate.splrep(s,ClA,k=5,s=1.0)
# s_new = np.linspace(0,s_final,1000)
# ClA_interp = interpolate.splev(s_new,interp)

# interp = interpolate.splrep(s,V,k=3,s=10.0)
# s_new = np.linspace(0,s_final,1000)
# V_interp = interpolate.splev(s_new,interp)

#np.savetxt('newgatesfixeddx.csv',newgates,delimiter=',')

# newgates2 = np.genfromtxt('./newgates.csv',delimiter=',')

def plotTrackWithData(state,s):
	# if np.array_equal(state,V):
	# 	interp = interpolate.splrep(s,state,k=3,s=10.0)
	# elif np.array_equal(state,delta):
	# 	interp = interpolate.splrep(s,state,k=5,s=0.01)
	# elif np.array_equal(state,ClA):
	# 	interp = interpolate.splrep(s,state,k=5,s=1.0)
	# else:
	# 	interp = interpolate.splrep(s,state,k=5,s=1.0)

	# s_new = np.linspace(0,s_final,1000)
	# V_interp = interpolate.splev(s_new,interp)
	state = np.array(state)[:,0]
	s = np.array(s)[:,0]
	s_new = np.linspace(0,s_final,2000)

	cmap = mpl.cm.get_cmap('viridis')
	norm = mpl.colors.Normalize(vmin=np.amin(state),vmax=np.amax(state))

	fig, ax = plt.subplots(figsize=(15,6))
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
	if np.array_equal(state,V[:,0]):
		clb.set_label('Velocity (m/s)')
	elif np.array_equal(state,thrustRL[:,0]):
		clb.set_label('Thrust')
	elif np.array_equal(state,delta[:,0]):
		clb.set_label('Delta')
	elif np.array_equal(state,ClA[:,0]):
		clb.set_label('Wing angle')
	plt.tight_layout()
	plt.grid()

def plotTrackWithNodes():

	fig, ax = plt.subplots(figsize=(15,6))
	plt.plot(newgates[0],newgates[1],'o',markersize=1)

	plt.axis('equal')
	plt.plot(finespline[0],finespline[1],'k',linewidth=linewidth_from_data_units(8.5,ax),solid_capstyle="butt")
	plt.plot(finespline[0],finespline[1],'w',linewidth=linewidth_from_data_units(8,ax),solid_capstyle="butt")
	plt.xlabel('x (m)')
	plt.ylabel('y (m)')

	for i in range(len(newgates[0])):
		if i%2 == 1:
			continue
			#plt.plot(newgates[0][i],newgates[1][i],'o',color='tab:orange',markersize=3)
		else:
			plt.plot(newgates[0][i],newgates[1][i],'o',color='tab:blue',markersize=3)
			#plt.plot(newgates2[0][i],newgates2[1][i],'o',color='tab:orange',markersize=3)

	plt.tight_layout()
	plt.grid()

plotTrackWithData(V,s)
plotTrackWithData(thrustRL,s)
plotTrackWithData(delta,s)
plotTrackWithData(ClA,s)
plotTrackWithNodes()

print(len(V),len(delta),len(s))


# Check the validity of our results by using scipy.integrate.solve_ivp to integrate the solution.
run_sim = False
if run_sim:
	sim_out = traj.simulate()

print('Done simulating')

# Plot the results
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(18, 8))



#V vs s
axes[0][0].plot(s,
		 p.get_val('traj.phase0.timeseries.states:V'),
		 'ro', label='solution')

if run_sim:
	axes[0][0].plot(sim_out.get_val('traj.phase0.timeseries.states:t'),
			 sim_out.get_val('traj.phase0.timeseries.states:V'),
			 'b-', label='simulation')

axes[0][0].set_xlabel('s (m)')
axes[0][0].set_ylabel('V (m/s)')

# axes[0][0].legend()
axes[0][0].grid()


#ax vs s
axes[0][1].plot(s,
		 p.get_val('traj.phase0.timeseries.states:ax', units='m/s**2'),
		 'ro', label='solution')

if run_sim:
	axes[0][1].plot(sim_out.get_val('traj.phase0.timeseries.states:t'),
			 sim_out.get_val('traj.phase0.timeseries.states:ax', units='m/s**2'),
			 'b-', label='simulation')

axes[0][1].set_xlabel('s (m)')
axes[0][1].set_ylabel(r'$ax$ (m/s**2)')
# axes[0][1].legend()
axes[0][1].grid()

#ay vs s
axes[0][2].plot(s,
		 p.get_val('traj.phase0.timeseries.states:ay', units='m/s**2'),
		 'ro', label='solution')

if run_sim:
	axes[0][2].plot(sim_out.get_val('traj.phase0.timeseries.states:t'),
			 sim_out.get_val('traj.phase0.timeseries.states:ay', units='m/s**2'),
			 'b-', label='simulation')

axes[0][2].set_xlabel('s (m)')
axes[0][2].set_ylabel(r'$ay$ (m/s**2)')
# axes[0][2].legend()
axes[0][2].grid()


#n vs s
axes[1][0].plot(s,
		 p.get_val('traj.phase0.timeseries.states:n', units='m'),
		 'ro', label='solution')

if run_sim:
	axes[1][0].plot(sim_out.get_val('traj.phase0.timeseries.states:t'),
			 sim_out.get_val('traj.phase0.timeseries.states:n', units='m'),
			 'b-', label='simulation')

axes[1][0].set_xlabel('s (m)')
axes[1][0].set_ylabel('n (m)')
# axes[1][0].legend()
axes[1][0].grid()


#throttle vs s
axes[1][1].plot(s,thrustRL,
		 'ro', label='solution')

if run_sim:
	axes[1][1].plot(sim_out.get_val('traj.phase0.timeseries.states:t'),
			 sim_out.get_val('traj.phase0.timeseries.controls:thrust', units=None),
			 'b-', label='simulation')

axes[1][1].set_xlabel('s (m)')
axes[1][1].set_ylabel('thrust')
# axes[1][1].legend()
axes[1][1].grid()

#cla vs s
axes[1][2].plot(s,
		 p.get_val('traj.phase0.timeseries.controls:gamma', units='deg'), label='Wing angle')
#ClA = np.array(p.get_val('traj.phase0.timeseries.controls:ClA', units='m**2'))
#axes[1][2].plot(s,0.3*ClA+0.04*ClA**2, label='CdA')


if run_sim:
	axes[1][2].plot(s,
			 sim_out.get_val('traj.phase0.timeseries.controls:gamma', units='deg'),
			 'b-', label='simulation')

axes[1][2].set_xlabel('s (m)')
axes[1][2].set_ylabel('Wing angle')
#axes[1][2].legend()
axes[1][2].grid()

#s vs time
axes[0][3].plot(t,s,
		 'ro', label='solution')

if run_sim:
	axes[0][3].plot(sim_out.get_val('traj.phase0.timeseries.time'),
			 sim_out.get_val('traj.phase0.timeseries.states:t', units='m'),
			 'b-', label='simulation')

axes[0][3].set_xlabel('t (s)')
axes[0][3].set_ylabel('s (m)')
# axes[0][3].legend()
axes[0][3].grid()

#delta vs s
axes[1][3].plot(s,
		 p.get_val('traj.phase0.timeseries.controls:delta', units=None),
		 'ro', label='solution')

if run_sim:
	axes[1][3].plot(sim_out.get_val('traj.phase0.timeseries.states:t'),
			 sim_out.get_val('traj.phase0.timeseries.controls:delta', units=None),
			 'b-', label='simulation')

axes[1][3].set_xlabel('s (m)')
axes[1][3].set_ylabel('delta')
# axes[1][3].legend()
axes[1][3].grid()

#lambdadot vs s
axes[2][0].plot(s,
		 p.get_val('traj.phase0.timeseries.lambdadot', units=None),
		 'ro', label='solution')

if run_sim:
	axes[2][0].plot(sim_out.get_val('traj.phase0.timeseries.states:t'),
			 sim_out.get_val('traj.phase0.timeseries.lambdadot', units=None),
			 'b-', label='simulation')

axes[2][0].set_xlabel('s (m)')
axes[2][0].set_ylabel('lambdadot')
# axes[2][0].legend()
axes[2][0].grid()

#vdot vs s
axes[2][1].plot(s,
		 p.get_val('traj.phase0.timeseries.Vdot', units=None),
		 'ro', label='solution')

if run_sim:
	axes[2][1].plot(sim_out.get_val('traj.phase0.timeseries.states:t'),
			 sim_out.get_val('traj.phase0.timeseries.Vdot', units=None),
			 'b-', label='simulation')

axes[2][1].set_xlabel('s (m)')
axes[2][1].set_ylabel('Vdot')
# axes[2][1].legend()
axes[2][1].grid()

#omegadot vs s
axes[2][2].plot(s,
		 p.get_val('traj.phase0.timeseries.omegadot', units=None),
		 'ro', label='solution')

if run_sim:
	axes[2][2].plot(sim_out.get_val('traj.phase0.timeseries.states:t'),
			 sim_out.get_val('traj.phase0.timeseries.omegadot', units=None),
			 'b-', label='simulation')

axes[2][2].set_xlabel('s (m)')
axes[2][2].set_ylabel('omegadot')
# axes[2][2].legend()
axes[2][2].grid()

#alphadot vs s
axes[2][3].plot(s,
		 p.get_val('traj.phase0.timeseries.alphadot', units=None),
		 'ro', label='solution')

if run_sim:
	axes[2][3].plot(sim_out.get_val('traj.phase0.timeseries.states:t'),
			 sim_out.get_val('traj.phase0.timeseries.alphadot', units=None),
			 'b-', label='simulation')

axes[2][3].set_xlabel('s (m)')
axes[2][3].set_ylabel('alphadot')
# axes[2][3].legend()
axes[2][3].grid()

#lambda vs s
axes[3][1].plot(s,
		 p.get_val('traj.phase0.timeseries.states:lambda', units=None),
		 'ro', label='solution')

if run_sim:
	axes[3][1].plot(sim_out.get_val('traj.phase0.timeseries.states:t'),
			 sim_out.get_val('traj.phase0.timeseries.states:lambda', units=None),
			 'b-', label='simulation')

axes[3][1].set_xlabel('s (m)')
axes[3][1].set_ylabel('lambda')
# axes[3][1].legend()
axes[3][1].grid()


#omega vs s
axes[3][2].plot(s,
		 p.get_val('traj.phase0.timeseries.states:omega', units=None),
		 'ro', label='solution')

if run_sim:
	axes[3][2].plot(sim_out.get_val('traj.phase0.timeseries.states:t'),
			 sim_out.get_val('traj.phase0.timeseries.states:omega', units=None),
			 'b-', label='simulation')

axes[3][2].set_xlabel('s (m)')
axes[3][2].set_ylabel('omega')
# axes[3][2].legend()
axes[3][2].grid()

#alpha vs s
axes[3][3].plot(s,
		 p.get_val('traj.phase0.timeseries.states:alpha', units=None),
		 'ro', label='solution')

if run_sim:
	axes[3][3].plot(sim_out.get_val('traj.phase0.timeseries.states:t'),
			 sim_out.get_val('traj.phase0.timeseries.states:alpha', units=None),
			 'b-', label='simulation')

axes[3][3].set_xlabel('s (m)')
axes[3][3].set_ylabel('alpha')
# axes[3][3].legend()
axes[3][3].grid()

#power vs time
axes[3][0].plot(t,
		 p.get_val('traj.phase0.timeseries.powerFL', units='W'),
		 'ro', label='solution')

if run_sim:
	axes[3][0].plot(sim_out.get_val('traj.phase0.timeseries.time'),
			 sim_out.get_val('traj.phase0.timeseries.powerFL', units='W'),
			 'b-', label='simulation')

axes[3][0].set_xlabel('t (s)')
axes[3][0].set_ylabel('power (W)')
# axes[3][0].legend()
axes[3][0].grid()

#sdot vs time
axes[0][4].plot(t,
		 p.get_val('traj.phase0.timeseries.sdot', units='m/s'),
		 'ro', label='solution')

if run_sim:
	axes[0][4].plot(sim_out.get_val('traj.phase0.timeseries.time'),
			 sim_out.get_val('traj.phase0.timeseries.sdot', units='m/s'),
			 'b-', label='simulation')

axes[0][4].set_xlabel('t (s)')
axes[0][4].set_ylabel('sdot (m/s)')
# axes[0][4].legend()
axes[0][4].grid()

#sdot vs s
axes[1][4].plot(s,
		 p.get_val('traj.phase0.timeseries.sdot', units='m/s'),
		 'ro', label='solution')

if run_sim:
	axes[1][4].plot(sim_out.get_val('traj.phase0.timeseries.states:t'),
			 sim_out.get_val('traj.phase0.timeseries.sdot', units='m/s'),
			 'b-', label='simulation')

axes[1][4].set_xlabel('s (m)')
axes[1][4].set_ylabel('sdot (m/s)')
# axes[1][4].legend()
axes[1][4].grid()

#c_rr vs s
axes[2][4].plot(s,
		 p.get_val('traj.phase0.timeseries.c_rr', units=None),
		 'ro', label='solution')

if run_sim:
	axes[2][4].plot(sim_out.get_val('traj.phase0.timeseries.states:t'),
			 sim_out.get_val('traj.phase0.timeseries.c_rr', units=None),
			 'b-', label='simulation')

axes[2][4].set_xlabel('s (m)')
axes[2][4].set_ylabel('c_rr (-)')
# axes[2][4].legend()
axes[2][4].grid()

#c_fl vs s
axes[3][4].plot(s,
		 p.get_val('traj.phase0.timeseries.c_fl', units=None),
		 'ro', label='solution')

if run_sim:
	axes[3][4].plot(sim_out.get_val('traj.phase0.timeseries.states:t'),
			 sim_out.get_val('traj.phase0.timeseries.c_fl', units=None),
			 'b-', label='simulation')

axes[3][4].set_xlabel('s (m)')
axes[3][4].set_ylabel('c_fl (-)')
# axes[3][4].legend()
axes[3][4].grid()

#N-rr vs s
axes[4][3].plot(s,
		 p.get_val('traj.phase0.timeseries.N_rr', units='N'),
		 'ro', label='solution')

if run_sim:
	axes[4][3].plot(sim_out.get_val('traj.phase0.timeseries.states:t'),
			 sim_out.get_val('traj.phase0.timeseries.N_rr', units='N'),
			 'b-', label='simulation')

axes[4][3].set_xlabel('s (m)')
axes[4][3].set_ylabel('N_rr (N)')
# axes[4][3].legend()
axes[4][3].grid()

#N_fr vs s
axes[4][1].plot(s,
		 p.get_val('traj.phase0.timeseries.N_fr', units='N'),
		 'ro', label='solution')

if run_sim:
	axes[4][1].plot(sim_out.get_val('traj.phase0.timeseries.states:t'),
			 sim_out.get_val('traj.phase0.timeseries.N_fr', units='N'),
			 'b-', label='simulation')

axes[4][1].set_xlabel('s (m)')
axes[4][1].set_ylabel('N_fr (N)')
# axes[4][1].legend()
axes[4][1].grid()

#N_rl vs s
axes[4][2].plot(s,
		 p.get_val('traj.phase0.timeseries.N_rl', units='N'),
		 'ro', label='solution')

if run_sim:
	axes[4][2].plot(sim_out.get_val('traj.phase0.timeseries.states:t'),
			 sim_out.get_val('traj.phase0.timeseries.N_rl', units='N'),
			 'b-', label='simulation')

axes[4][2].set_xlabel('s (m)')
axes[4][2].set_ylabel('N_rl (N)')
# axes[4][2].legend()
axes[4][2].grid()

#N_fl vs s
axes[4][0].plot(s,
		 p.get_val('traj.phase0.timeseries.N_fl', units='N'),
		 'ro', label='solution')

if run_sim:
	axes[4][0].plot(sim_out.get_val('traj.phase0.timeseries.states:t'),
			 sim_out.get_val('traj.phase0.timeseries.N_fl', units='N'),
			 'b-', label='simulation')

axes[4][0].set_xlabel('s (m)')
axes[4][0].set_ylabel('N_fl (N)')
# axes[4][0].legend()
axes[4][0].grid()

#kappa vs s
axes[4][4].plot(s,
		 p.get_val('traj.phase0.timeseries.kappa', units='1/m'),
		 'ro', label='solution')

if run_sim:
	axes[4][4].plot(sim_out.get_val('traj.phase0.timeseries.states:t'),
			 sim_out.get_val('traj.phase0.timeseries.kappa', units='1/m'),
			 'b-', label='simulation')

axes[4][4].set_xlabel('s (m)')
axes[4][4].set_ylabel('centerline curvature')
# axes[4][4].legend()
axes[4][4].grid()
plt.subplots_adjust(right=0.98,bottom=0.08,top=0.97,left=0.07,hspace=0.52,wspace=0.52)



fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(15, 8))

axes[0].plot(s,
		 p.get_val('traj.phase0.timeseries.states:V'), label='solution')

if run_sim:
	axes[0].plot(sim_out.get_val('traj.phase0.timeseries.states:t'),
			 sim_out.get_val('traj.phase0.timeseries.states:V'),
			 'b-', label='simulation')

axes[0].set_xlabel('s (m)')
axes[0].set_ylabel('V (m/s)')

# axes[0][0].legend()
axes[0].grid()
axes[0].set_xlim(0,s_final)

#n vs s
axes[1].plot(s,
		 p.get_val('traj.phase0.timeseries.states:n', units='m'), label='solution')

if run_sim:
	axes[1].plot(sim_out.get_val('traj.phase0.timeseries.states:t'),
			 sim_out.get_val('traj.phase0.timeseries.states:n', units='m'),
			 'b-', label='simulation')

axes[1].set_xlabel('s (m)')
axes[1].set_ylabel('n (m)')
# axes[1][0].legend()
axes[1].grid()
axes[1].set_xlim(0,s_final)


#throttle vs s
axes[2].plot(s,thrustRL, label='RL')
axes[2].plot(s,thrustRR, label='RR')
axes[2].plot(s,thrustFL, label='FL')
axes[2].plot(s,thrustFR, label='FR')

if run_sim:
	axes[2].plot(sim_out.get_val('traj.phase0.timeseries.states:t'),
			 sim_out.get_val('traj.phase0.timeseries.controls:thrust', units=None),
			 'b-', label='simulation')

axes[2].set_xlabel('s (m)')
axes[2].set_ylabel('thrust')
# axes[1][1].legend()
axes[2].grid()
axes[2].set_xlim(0,s_final)

#delta vs s
axes[3].plot(s,
		 p.get_val('traj.phase0.timeseries.controls:delta', units=None), label='solution')
if run_sim:
	axes[3].plot(sim_out.get_val('traj.phase0.timeseries.states:t'),
			 sim_out.get_val('traj.phase0.timeseries.controls:delta', units=None),
			 'b-', label='simulation')

axes[3].set_xlabel('s (m)')
axes[3].set_ylabel('delta')
# axes[1][3].legend()
axes[3].grid()
axes[3].set_xlim(0,s_final)

#ClA vs s
axes[4].plot(s,
		 p.get_val('traj.phase0.timeseries.controls:gamma', units='deg'), label='Wing angle')
#ClA = np.array(p.get_val('traj.phase0.timeseries.controls:ClA', units='m**2'))
#axes[4].plot(s,0.3*ClA+0.04*ClA**2, label='CdA')

if run_sim:
	axes[4].plot(sim_out.get_val('traj.phase0.timeseries.states:t'),
			 sim_out.get_val('traj.phase0.timeseries.controls:gamma', units='deg'),
			 'b-', label='simulation')

axes[4].set_xlabel('s (m)')
axes[4].set_ylabel('Wing angle')
#axes[4].legend()
axes[4].grid()
plt.subplots_adjust(right=0.98,bottom=0.07,top=0.97,left=0.07,hspace=0.53)

axes[4].set_xlim(0,s_final)


fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(15, 8))

#c_fl vs s
axes[0].plot(s,
		 p.get_val('traj.phase0.timeseries.c_fl', units=None), label='solution')

axes[0].set_xlabel('s (m)')
axes[0].set_ylabel('c_fl (-)')
axes[0].grid()
axes[0].set_xlim(0,s_final)


#c_fr vs s
axes[1].plot(s,
		 p.get_val('traj.phase0.timeseries.c_fr', units=None), label='solution')

axes[1].set_xlabel('s (m)')
axes[1].set_ylabel('c_fr (-)')
axes[1].grid()
axes[1].set_xlim(0,s_final)


#c_rl vs s
axes[2].plot(s,
		 p.get_val('traj.phase0.timeseries.c_rl', units=None), label='solution')

axes[2].set_xlabel('s (m)')
axes[2].set_ylabel('c_rl (-)')
axes[2].grid()
axes[2].set_xlim(0,s_final)


#c_rr vs s
axes[3].plot(s,
		 p.get_val('traj.phase0.timeseries.c_rr', units=None), label='solution')

axes[3].set_xlabel('s (m)')
axes[3].set_ylabel('c_rr (-)')
axes[3].grid()
axes[3].set_xlim(0,s_final)

axes[4].plot(s,
		 p.get_val('traj.phase0.timeseries.c_fl', units=None), label='c_fl')
axes[4].plot(s,
		 p.get_val('traj.phase0.timeseries.c_fr', units=None), label='c_fr')
axes[4].plot(s,
		 p.get_val('traj.phase0.timeseries.c_rl', units=None), label='c_rl')
axes[4].plot(s,
		 p.get_val('traj.phase0.timeseries.c_rr', units=None), label='c_rr')
max_power = np.maximum.reduce([powerRL,powerRR,powerFL,powerFR])
max_power[max_power<0] = 0
axes[4].plot(s,max_power/pmax,label='Maximum power')
#power = np.array(p.get_val('traj.phase0.timeseries.powerRL', units='W'))/pmax
#axes[4].plot(s,power,label='Power')

axes[4].legend()
axes[4].set_xlabel('s (m)')
axes[4].set_ylabel('Performance constraints')
axes[4].grid()
axes[4].set_xlim(0,s_final)
plt.subplots_adjust(right=0.98,bottom=0.07,top=0.97,left=0.07,hspace=0.53)

plt.figure(figsize=(12,4))
#throttle vs s
plt.plot(s,thrustFL,label='FL')

plt.plot(s,thrustFR,label='FR')

plt.plot(s,thrustRL,label='RL')

plt.plot(s,thrustRR,label='RR')

plt.xlabel('s (m)')
plt.ylabel('thrust')
plt.subplots_adjust(right=0.88,bottom=0.14,top=0.95,left=0.09)
plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
plt.xlim(0,s_final)
plt.grid()

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15, 8))

#ClA vs s
axes[0].plot(s,
		 p.get_val('traj.phase0.timeseries.controls:gamma', units='deg'), label='Wing angle')
#ClA = np.array(p.get_val('traj.phase0.timeseries.controls:ClA', units='m**2'))
#axes[4].plot(s,0.3*ClA+0.04*ClA**2, label='CdA')

if run_sim:
	axes[4].plot(sim_out.get_val('traj.phase0.timeseries.states:t'),
			 sim_out.get_val('traj.phase0.timeseries.controls:gamma', units='deg'),
			 'b-', label='simulation')

axes[0].set_xlabel('s (m)')
axes[0].set_ylabel('Wing angle')
#axes[4].legend()
axes[0].grid()
plt.subplots_adjust(right=0.98,bottom=0.07,top=0.97,left=0.07,hspace=0.53)

axes[0].set_xlim(0,s_final)

V = np.array(V)
gamma = np.array(gamma)
S_w = 0.8
rho = 1.2

ClA = 1.614-1.361e-3*V-4.186e-5*V**2
downforce = 0.5*rho*ClA*V**2

Cl_wing = 1.5833+0.0333*gamma
downforce_wing = 0.5*rho*Cl_wing*S_w*V**2

downforce = downforce+downforce_wing

CdA = 1.055-7.588e-4*V-9.156e-6*V**2
drag = 0.5*rho*CdA*V**2

Cd_wing = 0.0667+0.0127*gamma
drag_wing = 0.5*rho*Cd_wing*S_w*V**2

drag = drag+drag_wing

axes[1].plot(s,downforce,label='Total downforce')
axes[1].plot(s,drag,label='Total Drag')
axes[1].set_xlabel('s (m)')
axes[1].set_ylabel('Force (N)')
axes[1].legend()
axes[1].set_xlim(0,s_final)
axes[1].grid()

axes[2].plot(s,downforce_wing,label='Wing downforce')
axes[2].plot(s,drag_wing,label='Wing Drag')
axes[2].set_xlabel('s (m)')
axes[2].set_ylabel('Force (N)')
axes[2].legend()
axes[2].grid()
axes[2].set_xlim(0,s_final)

plt.subplots_adjust(right=0.98,bottom=0.10,top=0.97,left=0.07,hspace=0.53)

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(15, 4))
plt.subplots_adjust(right=0.82,bottom=0.14,top=0.97,left=0.07)

axes.plot(s,
		 p.get_val('traj.phase0.timeseries.c_fl', units=None), label='c_fl')
axes.plot(s,
		 p.get_val('traj.phase0.timeseries.c_fr', units=None), label='c_fr')
axes.plot(s,
		 p.get_val('traj.phase0.timeseries.c_rl', units=None), label='c_rl')
axes.plot(s,
		 p.get_val('traj.phase0.timeseries.c_rr', units=None), label='c_rr')
max_power = np.maximum.reduce([powerRL,powerRR,powerFL,powerFR])
max_power[max_power<0] = 0
axes.plot(s,max_power/pmax,label='Maximum power')
#power = np.array(p.get_val('traj.phase0.timeseries.powerRL', units='W'))/pmax
#axes[4].plot(s,power,label='Power')

axes.legend(bbox_to_anchor=(1.04,0.5), loc="center left")
axes.set_xlabel('s (m)')
axes.set_ylabel('Performance constraints')
axes.grid()
axes.set_xlim(0,s_final)

plt.show()