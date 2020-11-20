import numpy as np
import openmdao.api as om
import dymos as dm
import matplotlib.pyplot as plt
from combinedODE import CombinedODE
import time

#track curvature imports
from scipy import interpolate
from scipy import signal
from Track import Track
import tracks
from spline import *
from linewidthhelper import *
import os

def runDymosProblem(track,num_segments,solver,tol):
	points = getTrackPoints(track)
	finespline,gates,gatesd,curv,slope = getSpline(points,s=0.0)
	s_final = track.getTotalLength()

	output_file_name = str(num_segments)+solver+"{:.2E}".format(tol)+'.out'
	print('output file: '+output_file_name)

	# Define the OpenMDAO problem
	p = om.Problem(model=om.Group())

	# Define a Trajectory object
	traj = dm.Trajectory()
	p.model.add_subsystem('traj', subsys=traj)

	# Define a Dymos Phase object with GaussLobatto Transcription

	phase = dm.Phase(ode_class=CombinedODE,
			     transcription=dm.GaussLobatto(num_segments=num_segments, order=3,compressed=True))

	traj.add_phase(name='phase0', phase=phase)

	# Set the time options
	phase.set_time_options(fix_initial=True,fix_duration=True,duration_val=s_final,targets=['curv.s'],units='m',duration_ref=s_final,duration_ref0=10) #50,150,50

	#Define states
	phase.add_state('t', fix_initial=True, fix_final=False, units='s', lower = 0,rate_source='dt_ds',ref=100)
	phase.add_state('n', fix_initial=True, fix_final=True, units='m', upper = 4.0, lower = -4.0, rate_source='dn_ds',targets=['n'],ref=4.0)
	phase.add_state('V', fix_initial=True, fix_final=False, units='m/s', lower = 5.0, ref = 40, ref0=5,rate_source='dV_ds', targets=['V'])
	phase.add_state('alpha', fix_initial=True, fix_final=True, units='rad', rate_source='dalpha_ds',targets=['alpha'],ref=0.15)
	phase.add_state('lambda', fix_initial=True, fix_final=True, units='rad',rate_source='dlambda_ds',targets=['lambda'],ref=0.01)
	phase.add_state('omega', fix_initial=True, fix_final=True, units='rad/s',rate_source='domega_ds',targets=['omega'],ref=0.3)
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

	# Set the driver.

	if solver == 'IPOPT':
		p.driver = om.pyOptSparseDriver(optimizer='IPOPT')
		p.driver.opt_settings['mu_init'] = 1e-3
		p.driver.opt_settings['max_iter'] = 1000
		p.driver.opt_settings['acceptable_tol'] = tol
		p.driver.opt_settings['constr_viol_tol'] = tol
		p.driver.opt_settings['compl_inf_tol'] = tol
		p.driver.opt_settings['acceptable_iter'] = 0
		p.driver.opt_settings['tol'] = tol
		p.driver.opt_settings['hessian_approximation'] = 'exact'
		p.driver.opt_settings['nlp_scaling_method'] = 'none'
		p.driver.opt_settings['print_level'] = 5
		p.driver.opt_settings['output_file'] = output_file_name
	elif solver == 'SNOPT':
		p.driver = om.pyOptSparseDriver(optimizer='SNOPT')
		p.driver.opt_settings['Scale option'] = 1
		p.driver.opt_settings['Minor iterations limit']=100000
		p.driver.opt_settings['Major iterations limit']=100000
		p.driver.opt_settings['Superbasics limit']=100000
		p.driver.opt_settings['Iterations limit']=20000000
		p.driver.opt_settings['Major feasibility tolerance'] = tol    # -6 for benchmark
		p.driver.opt_settings['Major optimality tolerance'] = tol    # -6 by default
		p.driver.opt_settings['Verify level'] = -1  # do not check gradient
		p.driver.opt_settings['Function precision'] = 1e-12
		p.driver.opt_settings['Linesearch tolerance'] = 0.9  # default 0.9, smaller for more accurate search
		p.driver.opt_settings['Major step limit'] = 0.8      # default 2.,

	p.driver.declare_coloring()

	# Setup the problem
	p.setup(check=True) #force_alloc_complex=True
	# Now that the OpenMDAO problem is setup, we can set the values of the states.

	p.set_val('traj.phase0.states:V',phase.interpolate(ys=[54.2,10], nodes='state_input'),units='m/s')
	p.set_val('traj.phase0.states:lambda',phase.interpolate(ys=[0.0,0.0], nodes='state_input'),units='rad')
	p.set_val('traj.phase0.states:omega',phase.interpolate(ys=[0.0,0.0], nodes='state_input'),units='rad/s')
	p.set_val('traj.phase0.states:alpha',phase.interpolate(ys=[0.0,0.0], nodes='state_input'),units='rad')
	p.set_val('traj.phase0.states:ax',phase.interpolate(ys=[0.0,0.0], nodes='state_input'),units='m/s**2')
	p.set_val('traj.phase0.states:ay',phase.interpolate(ys=[0.0,0.0], nodes='state_input'),units='m/s**2')
	#p.set_val('traj.phase0.states:t',phase.interpolate(ys=[0.0,s_final], nodes='state_input'),units='m')
	p.set_val('traj.phase0.states:n',phase.interpolate(ys=[4.0,4.0], nodes='state_input'),units='m')
	p.set_val('traj.phase0.states:t',phase.interpolate(ys=[0.0,100.0], nodes='state_input'),units='s')

	p.set_val('traj.phase0.controls:delta',phase.interpolate(ys=[0.0,0.0], nodes='control_input'),units='rad')

	p.set_val('traj.phase0.controls:thrustFL',phase.interpolate(ys=[0.0, 0.0], nodes='control_input'),units=None)
	p.set_val('traj.phase0.controls:thrustFR',phase.interpolate(ys=[0.0, 0.0], nodes='control_input'),units=None)
	p.set_val('traj.phase0.controls:thrustRL',phase.interpolate(ys=[0.0, 0.0], nodes='control_input'),units=None)
	p.set_val('traj.phase0.controls:thrustRR',phase.interpolate(ys=[0.0, 0.0], nodes='control_input'),units=None)
	p.set_val('traj.phase0.controls:gamma',phase.interpolate(ys=[10.0, 10.0], nodes='control_input'),units='deg')

	start_time = time.time()
	p.run_driver()
	end_time = time.time()

	if solver == 'SNOPT':
		os.rename("./SNOPT_summary.out","./"+output_file_name)

	t = p.get_val('traj.phase0.timeseries.states:t')
	return t[-1][0],end_time-start_time