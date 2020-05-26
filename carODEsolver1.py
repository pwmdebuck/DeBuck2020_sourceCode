import numpy as np
import openmdao.api as om
import dymos as dm
import matplotlib.pyplot as plt
from combinedODE import CombinedODE

# Define the OpenMDAO problem
p = om.Problem(model=om.Group())

# Define a Trajectory object
traj = dm.Trajectory()
p.model.add_subsystem('traj', subsys=traj)

# Define a Dymos Phase object with GaussLobatto Transcription
phase = dm.Phase(ode_class=CombinedODE,
                 transcription=dm.GaussLobatto(num_segments=40, order=3,compressed=True))

traj.add_phase(name='phase0', phase=phase)

# Set the time options
# Time has no targets in our ODE.
# We fix the initial time so that the it is not a design variable in the optimization.
# The duration of the phase is allowed to be optimized, but is bounded on [0.5, 10].
phase.set_time_options(fix_initial=True, duration_bounds=(0.5, 30.0), units='s')

# Set the time options
# Initial values of positions and velocity are all fixed.
# The final value of position are fixed, but the final velocity is a free variable.
# The equations of motion are not functions of position, so 'x' and 'y' have no targets.
# The rate source points to the output in the ODE which provides the time derivative of the
# given state.
phase.add_state('s', fix_initial=True, fix_final=True, units='m', rate_source='car.sdot',targets=['car.s'])
phase.add_state('n', fix_initial=True, fix_final=False, units='m', rate_source='car.ndot',targets=['car.n'])
phase.add_state('V', fix_initial=True, fix_final=True, units='m/s',
                rate_source='car.Vdot', targets=['car.V','tire.V','accel.V'])
phase.add_state('alpha', fix_initial=True, fix_final=False, units='rad', rate_source='car.alphadot',targets=['car.alpha'])
phase.add_state('lambda', fix_initial=True, fix_final=False, units='rad', rate_source='car.lambdadot',targets=['car.lambda','tire.lambda','accel.lambda'])
phase.add_state('omega', fix_initial=True, fix_final=False, units='rad/s', rate_source='car.omegadot',targets=['car.omega','tire.omega','accel.omega'])
phase.add_state('ax',fix_initial=True,fix_final=False,units='m/s**2',rate_source='accel.axdot',targets=['accel.ax','normal.ax'])
phase.add_state('ay',fix_initial=True,fix_final=False,units='m/s**2',rate_source='accel.aydot',targets=['accel.ay','normal.ay'])


# controls
phase.add_control(name='delta', units='rad', lower=-np.pi/2, upper=np.pi/2,fix_initial=True,fix_final=False, targets=['car.delta','tire.delta'])
phase.add_control(name='throttle', units=None, lower=0, upper=1,fix_initial=False,fix_final=False, targets=['tire.throttle'])
phase.add_control(name='brake', units=None, lower=0, upper=1,fix_initial=False,fix_final=False, targets=['tire.brake'])

#track curvature
phase.add_design_parameter('kappa',val=0.0,units='1/m',opt=False,targets=['car.kappa'])

pmax = 300000 #W
#power constraint
#phase.add_path_constraint('car.power',shape=(1,),units='W',lower=-pmax,upper=pmax)

# Minimize final time.
phase.add_objective('time', loc='final')

#add output timeseries
phase.add_timeseries_output('car.lambdadot',units='rad/s',shape=(1,))
phase.add_timeseries_output('car.Vdot',units='m/s**2',shape=(1,))
phase.add_timeseries_output('car.alphadot',units='rad/s',shape=(1,))
phase.add_timeseries_output('car.omegadot',units='rad/s**2',shape=(1,))
phase.add_timeseries_output('car.power',units='W',shape=(1,))
phase.add_timeseries_output('car.sdot',units='m/s',shape=(1,))

# Set the driver.
#p.driver = om.ScipyOptimizeDriver(maxiter=1000)
p.driver = om.pyOptSparseDriver()
p.driver.opt_settings['MAXIT'] = 1000
p.driver.opt_settings['ACC'] = 1e-9

# Allow OpenMDAO to automatically determine our sparsity pattern.
# Doing so can significant speed up the execution of Dymos.
p.driver.declare_coloring()

# Setup the problem
p.setup(check=True)

# Now that the OpenMDAO problem is setup, we can set the values of the states.

p.set_val('traj.phase0.states:V',phase.interpolate(ys=[10,30], nodes='state_input'),units='m/s')
p.set_val('traj.phase0.states:lambda',phase.interpolate(ys=[0.0,0.0], nodes='state_input'),units='rad')
p.set_val('traj.phase0.states:omega',phase.interpolate(ys=[0.0,0.0], nodes='state_input'),units='rad/s')
p.set_val('traj.phase0.states:alpha',phase.interpolate(ys=[0.0,0.0], nodes='state_input'),units='rad')
p.set_val('traj.phase0.states:ax',phase.interpolate(ys=[0.0,0.0], nodes='state_input'),units='m/s**2')
p.set_val('traj.phase0.states:ay',phase.interpolate(ys=[0.0,0.0], nodes='state_input'),units='m/s**2')
p.set_val('traj.phase0.states:s',phase.interpolate(ys=[0.0,100.0], nodes='state_input'),units='m')
p.set_val('traj.phase0.states:n',phase.interpolate(ys=[0.0,0.0], nodes='state_input'),units='m')

p.set_val('traj.phase0.controls:throttle',phase.interpolate(ys=[0, 1.0], nodes='control_input'),units=None)
p.set_val('traj.phase0.controls:brake',phase.interpolate(ys=[0,0], nodes='control_input'),units=None)
p.set_val('traj.phase0.controls:delta',phase.interpolate(ys=[0,0], nodes='control_input'),units='rad')


#enable refinement
# p.model.traj.phases.phase0.set_refine_options(refine=True)
# dm.run_problem(p,refine=True,refine_iteration_limit=10)

# Run the driver to solve the problem
p.run_driver()
#p.run_model()

# Check the validity of our results by using scipy.integrate.solve_ivp to integrate the solution.
sim_out = traj.simulate()

# Plot the results
fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(21, 12))



#V vs s
axes[0][0].plot(p.get_val('traj.phase0.timeseries.states:s'),
             p.get_val('traj.phase0.timeseries.states:V'),
             'ro', label='solution')

axes[0][0].plot(sim_out.get_val('traj.phase0.timeseries.states:s'),
             sim_out.get_val('traj.phase0.timeseries.states:V'),
             'b-', label='simulation')

axes[0][0].set_xlabel('s (m)')
axes[0][0].set_ylabel('V (m/s)')
axes[0][0].axis('equal')
axes[0][0].legend()
axes[0][0].grid()


#ax vs s
axes[0][1].plot(p.get_val('traj.phase0.timeseries.states:s'),
             p.get_val('traj.phase0.timeseries.states:ax', units='m/s**2'),
             'ro', label='solution')

axes[0][1].plot(sim_out.get_val('traj.phase0.timeseries.states:s'),
             sim_out.get_val('traj.phase0.timeseries.states:ax', units='m/s**2'),
             'b-', label='simulation')

axes[0][1].set_xlabel('s (m)')
axes[0][1].set_ylabel(r'$ax$ (m/s**2)')
axes[0][1].legend()
axes[0][1].grid()

#ay vs s
axes[0][2].plot(p.get_val('traj.phase0.timeseries.states:s'),
             p.get_val('traj.phase0.timeseries.states:ay', units='m/s**2'),
             'ro', label='solution')

axes[0][2].plot(sim_out.get_val('traj.phase0.timeseries.states:s'),
             sim_out.get_val('traj.phase0.timeseries.states:ay', units='m/s**2'),
             'b-', label='simulation')

axes[0][2].set_xlabel('s (m)')
axes[0][2].set_ylabel(r'$ay$ (m/s**2)')
axes[0][2].legend()
axes[0][2].grid()


#n vs s
axes[1][0].plot(p.get_val('traj.phase0.timeseries.states:s'),
             p.get_val('traj.phase0.timeseries.states:n', units='m'),
             'ro', label='solution')

axes[1][0].plot(sim_out.get_val('traj.phase0.timeseries.states:s'),
             sim_out.get_val('traj.phase0.timeseries.states:n', units='m'),
             'b-', label='simulation')

axes[1][0].set_xlabel('s (m)')
axes[1][0].set_ylabel('n (m)')
axes[1][0].legend()
axes[1][0].grid()


#throttle vs s
axes[1][1].plot(p.get_val('traj.phase0.timeseries.states:s'),
             p.get_val('traj.phase0.timeseries.controls:throttle', units=None),
             'ro', label='solution')

axes[1][1].plot(sim_out.get_val('traj.phase0.timeseries.states:s'),
             sim_out.get_val('traj.phase0.timeseries.controls:throttle', units=None),
             'b-', label='simulation')

axes[1][1].set_xlabel('s (m)')
axes[1][1].set_ylabel('throttle')
axes[1][1].legend()
axes[1][1].grid()

#brake vs s
axes[1][2].plot(p.get_val('traj.phase0.timeseries.states:s'),
             p.get_val('traj.phase0.timeseries.controls:brake', units=None),
             'ro', label='solution')

axes[1][2].plot(sim_out.get_val('traj.phase0.timeseries.states:s'),
             sim_out.get_val('traj.phase0.timeseries.controls:brake', units=None),
             'b-', label='simulation')

axes[1][2].set_xlabel('s (m)')
axes[1][2].set_ylabel('brake')
axes[1][2].legend()
axes[1][2].grid()

#s vs time
axes[0][3].plot(p.get_val('traj.phase0.timeseries.time'),
             p.get_val('traj.phase0.timeseries.states:s', units='m'),
             'ro', label='solution')

axes[0][3].plot(sim_out.get_val('traj.phase0.timeseries.time'),
             sim_out.get_val('traj.phase0.timeseries.states:s', units='m'),
             'b-', label='simulation')

axes[0][3].set_xlabel('t (s)')
axes[0][3].set_ylabel('s (m)')
axes[0][3].legend()
axes[0][3].grid()

#delta vs s
axes[1][3].plot(p.get_val('traj.phase0.timeseries.states:s'),
             p.get_val('traj.phase0.timeseries.controls:delta', units=None),
             'ro', label='solution')

axes[1][3].plot(sim_out.get_val('traj.phase0.timeseries.states:s'),
             sim_out.get_val('traj.phase0.timeseries.controls:delta', units=None),
             'b-', label='simulation')

axes[1][3].set_xlabel('s (m)')
axes[1][3].set_ylabel('delta')
axes[1][3].legend()
axes[1][3].grid()

#lambdadot vs s
axes[2][0].plot(p.get_val('traj.phase0.timeseries.states:s'),
             p.get_val('traj.phase0.timeseries.lambdadot', units=None),
             'ro', label='solution')

axes[2][0].plot(sim_out.get_val('traj.phase0.timeseries.states:s'),
             sim_out.get_val('traj.phase0.timeseries.lambdadot', units=None),
             'b-', label='simulation')

axes[2][0].set_xlabel('s (m)')
axes[2][0].set_ylabel('lambdadot')
axes[2][0].legend()
axes[2][0].grid()

#vdot vs s
axes[2][1].plot(p.get_val('traj.phase0.timeseries.states:s'),
             p.get_val('traj.phase0.timeseries.Vdot', units=None),
             'ro', label='solution')

axes[2][1].plot(sim_out.get_val('traj.phase0.timeseries.states:s'),
             sim_out.get_val('traj.phase0.timeseries.Vdot', units=None),
             'b-', label='simulation')

axes[2][1].set_xlabel('s (m)')
axes[2][1].set_ylabel('Vdot')
axes[2][1].legend()
axes[2][1].grid()

#omegadot vs s
axes[2][2].plot(p.get_val('traj.phase0.timeseries.states:s'),
             p.get_val('traj.phase0.timeseries.omegadot', units=None),
             'ro', label='solution')

axes[2][2].plot(sim_out.get_val('traj.phase0.timeseries.states:s'),
             sim_out.get_val('traj.phase0.timeseries.omegadot', units=None),
             'b-', label='simulation')

axes[2][2].set_xlabel('s (m)')
axes[2][2].set_ylabel('omegadot')
axes[2][2].legend()
axes[2][2].grid()

#alphadot vs s
axes[2][3].plot(p.get_val('traj.phase0.timeseries.states:s'),
             p.get_val('traj.phase0.timeseries.alphadot', units=None),
             'ro', label='solution')

axes[2][3].plot(sim_out.get_val('traj.phase0.timeseries.states:s'),
             sim_out.get_val('traj.phase0.timeseries.alphadot', units=None),
             'b-', label='simulation')

axes[2][3].set_xlabel('s (m)')
axes[2][3].set_ylabel('alphadot')
axes[2][3].legend()
axes[2][3].grid()

#lambda vs s
axes[3][1].plot(p.get_val('traj.phase0.timeseries.states:s'),
             p.get_val('traj.phase0.timeseries.states:lambda', units=None),
             'ro', label='solution')

axes[3][1].plot(sim_out.get_val('traj.phase0.timeseries.states:s'),
             sim_out.get_val('traj.phase0.timeseries.states:lambda', units=None),
             'b-', label='simulation')

axes[3][1].set_xlabel('s (m)')
axes[3][1].set_ylabel('lambda')
axes[3][1].legend()
axes[3][1].grid()


#omega vs s
axes[3][2].plot(p.get_val('traj.phase0.timeseries.states:s'),
             p.get_val('traj.phase0.timeseries.states:omega', units=None),
             'ro', label='solution')

axes[3][2].plot(sim_out.get_val('traj.phase0.timeseries.states:s'),
             sim_out.get_val('traj.phase0.timeseries.states:omega', units=None),
             'b-', label='simulation')

axes[3][2].set_xlabel('s (m)')
axes[3][2].set_ylabel('omega')
axes[3][2].legend()
axes[3][2].grid()

#alpha vs s
axes[3][3].plot(p.get_val('traj.phase0.timeseries.states:s'),
             p.get_val('traj.phase0.timeseries.states:alpha', units=None),
             'ro', label='solution')

axes[3][3].plot(sim_out.get_val('traj.phase0.timeseries.states:s'),
             sim_out.get_val('traj.phase0.timeseries.states:alpha', units=None),
             'b-', label='simulation')

axes[3][3].set_xlabel('s (m)')
axes[3][3].set_ylabel('alpha')
axes[3][3].legend()
axes[3][3].grid()

#power vs time
axes[3][0].plot(p.get_val('traj.phase0.timeseries.time'),
             p.get_val('traj.phase0.timeseries.power', units='W'),
             'ro', label='solution')

axes[3][0].plot(sim_out.get_val('traj.phase0.timeseries.time'),
             sim_out.get_val('traj.phase0.timeseries.power', units='W'),
             'b-', label='simulation')

axes[3][0].set_xlabel('t (s)')
axes[3][0].set_ylabel('power (W)')
axes[3][0].legend()
axes[3][0].grid()

#sdot vs time
axes[0][4].plot(p.get_val('traj.phase0.timeseries.time'),
             p.get_val('traj.phase0.timeseries.sdot', units='m/s'),
             'ro', label='solution')

axes[0][4].plot(sim_out.get_val('traj.phase0.timeseries.time'),
             sim_out.get_val('traj.phase0.timeseries.sdot', units='m/s'),
             'b-', label='simulation')

axes[0][4].set_xlabel('t (s)')
axes[0][4].set_ylabel('sdot (m/s)')
axes[0][4].legend()
axes[0][4].grid()

#sdot vs s
axes[1][4].plot(p.get_val('traj.phase0.timeseries.states:s'),
             p.get_val('traj.phase0.timeseries.sdot', units='m/s'),
             'ro', label='solution')

axes[1][4].plot(sim_out.get_val('traj.phase0.timeseries.states:s'),
             sim_out.get_val('traj.phase0.timeseries.sdot', units='m/s'),
             'b-', label='simulation')

axes[1][4].set_xlabel('s (m)')
axes[1][4].set_ylabel('sdot (m/s)')
axes[1][4].legend()
axes[1][4].grid()


plt.show()