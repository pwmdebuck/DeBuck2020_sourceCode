import openmdao.api as om
import numpy as np

#track curvature imports
from scipy import interpolate
from scipy import signal
from Track import Track
import tracks
from spline import *

class Curvature(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        track = tracks.Barcelona

        points = getTrackPoints(track)
        print(track.getTotalLength())
        finespline,gates,gatesd,curv,slope = getSpline(points)

        self.curv = curv
        self.trackLength = track.getTotalLength()

    def setup(self):
        nn = self.options['num_nodes']

        #constants
        self.add_input('s', val=np.zeros(nn), desc='distance along track', units='m')

        #outputs
        self.add_output('kappa', val=np.zeros(nn), desc='track centerline Curvature', units='1/m')

        # Setup partials
        # arange = np.arange(self.options['num_nodes'], dtype=int)

        # #partials
        # self.declare_partials(of='axdot', wrt='ax', rows=arange, cols=arange)
        # self.declare_partials(of='axdot', wrt='Vdot', rows=arange, cols=arange)
        # self.declare_partials(of='axdot', wrt='omega', rows=arange, cols=arange)
        # self.declare_partials(of='axdot', wrt='V', rows=arange, cols=arange)
        # self.declare_partials(of='axdot', wrt='lambda', rows=arange, cols=arange)

        # self.declare_partials(of='aydot', wrt='ay', rows=arange, cols=arange)
        # self.declare_partials(of='aydot', wrt='omega', rows=arange, cols=arange)
        # self.declare_partials(of='aydot', wrt='V', rows=arange, cols=arange)
        # self.declare_partials(of='aydot', wrt='Vdot', rows=arange, cols=arange)
        # self.declare_partials(of='aydot', wrt='lambda', rows=arange, cols=arange)
        # self.declare_partials(of='aydot', wrt='lambdadot', rows=arange, cols=arange)



    def compute(self, inputs, outputs):
        s = inputs['s']

        num_curv_points = len(self.curv)

        kappa = np.zeros(len(s))

        for i in range(len(s)):
            index = np.floor((s[i]/self.trackLength)*num_curv_points)
            index = np.minimum(index,num_curv_points-1)
            kappa[i] = self.curv[index.astype(int)]


        outputs['kappa'] = kappa

    def compute_partials(self, inputs, jacobian):
        pass
        # tau_y = inputs['tau_y']
        # tau_x = inputs['tau_x']
        # V = inputs['V']
        # lamb = inputs['lambda']
        # omega = inputs['omega']
        # Vdot = inputs['Vdot']
        # lambdadot = inputs['lambdadot']
        # ax = inputs['ax']
        # ay = inputs['ay']

        # jacobian['axdot', 'ax'] = -1/tau_x
        # jacobian['axdot', 'Vdot'] = 1/tau_x
        # jacobian['axdot', 'omega'] = (V*lamb)/tau_x
        # jacobian['axdot', 'lambda'] = (omega*V)/tau_x
        # jacobian['axdot', 'V'] = (omega*lamb)/tau_x

        # jacobian['aydot', 'ay'] = -1/tau_y
        # jacobian['aydot', 'omega'] = V/tau_y
        # jacobian['aydot', 'V'] = (omega-lambdadot)/tau_y
        # jacobian['aydot', 'lambda'] = -Vdot/tau_y
        # jacobian['aydot', 'lambdadot'] = -V/tau_y
        # jacobian['aydot', 'Vdot'] = -lamb/tau_y

        








