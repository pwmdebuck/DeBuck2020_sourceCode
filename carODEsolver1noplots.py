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
import csv

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
		     transcription=dm.GaussLobatto(num_segments=800, order=3,compressed=True))

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
p.driver = om.pyOptSparseDriver(optimizer='IPOPT')
# p.driver.opt_settings['Scale option'] = 2
#p.driver.opt_settings['Verify level'] = 3

# p.driver.opt_settings['linear_solver'] = 'ma27'
p.driver.opt_settings['mu_init'] = 1e-3
p.driver.opt_settings['max_iter'] = 500
p.driver.opt_settings['acceptable_tol'] = 1e-6
p.driver.opt_settings['constr_viol_tol'] = 1e-6
p.driver.opt_settings['compl_inf_tol'] = 1e-6
p.driver.opt_settings['acceptable_iter'] = 0
p.driver.opt_settings['tol'] = 1e-6
#p.driver.opt_settings['mu_max'] = 10.0
p.driver.opt_settings['hessian_approximation'] = 'exact'
p.driver.opt_settings['nlp_scaling_method'] = 'none'
p.driver.opt_settings['print_level'] = 5
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
c_fl = p.get_val('traj.phase0.timeseries.c_fl', units=None)
c_fr = p.get_val('traj.phase0.timeseries.c_fr', units=None)
c_rl = p.get_val('traj.phase0.timeseries.c_rl', units=None)
c_rr = p.get_val('traj.phase0.timeseries.c_rr', units=None)

print(np.array(s).shape)
print(np.array(delta).shape)
print(np.array(thrustRL).shape)
print(np.array(ClA).shape)
print(np.array(V).shape)

# trackLength = track.getTotalLength()

# normals = getGateNormals(finespline,slope)
# newgates = []
# newnormals = []
# newn = []
# for i in range(len(n)):
# 	index = ((s[i]/s_final)*np.array(finespline).shape[1]).astype(int)
# 	#print(index[0])
# 	if index[0]==np.array(finespline).shape[1]:
# 		index[0] = np.array(finespline).shape[1]-1
# 	if i>0 and s[i] == s[i-1]:
# 		continue
# 	else:
# 		newgates.append([finespline[0][index[0]],finespline[1][index[0]]])
# 		newnormals.append(normals[index[0]])
# 		newn.append(n[i][0])

# newgates = reverseTransformGates(newgates)
# displacedGates = setGateDisplacements(newn,newgates,newnormals)
# displacedGates = np.array((transformGates(displacedGates)))

# displacedSpline,gates,gatesd,curv,slope = getSpline(displacedGates,0.0005,0)

plt.rcParams.update({'font.size': 10})


with open('4wdactive_barcelona.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(['s','V','gamma','n','t','c_fl','c_fr','c_rl','c_rr','powerFL','powerFR','powerRL','powerRR','delta'])
    for i in range(len(V)):
    	writer.writerow([s[i][0],V[i][0],gamma[i][0],n[i][0],t[i][0],c_fl[i][0],c_fr[i][0],c_rl[i][0],c_rr[i][0],powerFL[i][0],powerFR[i][0],powerRL[i][0],powerRR[i][0],delta[i][0]])

