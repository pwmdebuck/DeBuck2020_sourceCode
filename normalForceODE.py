import openmdao.api as om
import numpy as np

class NormalForceODE(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        #constants
        self.add_input('M', val=1184.0, desc='mass', units='kg')
        self.add_input('g', val=9.8, desc='mass', units='N')
        self.add_input('a', val=1.404, desc='cg to front distance', units='m')
        self.add_input('b', val=1.356, desc='cg to rear distance', units='m')
        self.add_input('tw', val=0.807, desc='half track width', units='m')
        self.add_input('h', val=0.4, desc='cg height', units='m')
        self.add_input('chi', val=0.5, desc='roll stiffness', units=None)

        #states
        self.add_input('ax', val=np.zeros(nn), desc='longitudinal acceleration', units='m/s**2')
        self.add_input('ay', val=np.zeros(nn), desc='lateral acceleration', units='m/s**2')

        #normal load outputs
        self.add_output('N_fl', val=np.zeros(nn), desc='normal force fl', units='N')
        self.add_output('N_fr', val=np.zeros(nn), desc='normal force fr', units='N')
        self.add_output('N_rl', val=np.zeros(nn), desc='normal force rl', units='N')
        self.add_output('N_rr', val=np.zeros(nn), desc='normal force rr', units='N')

        # Setup partials
        arange = np.arange(self.options['num_nodes'], dtype=int)

        #partials
        self.declare_partials(of='N_fl', wrt='ax', rows=arange, cols=arange)
        self.declare_partials(of='N_fr', wrt='ax', rows=arange, cols=arange)
        self.declare_partials(of='N_rl', wrt='ax', rows=arange, cols=arange)
        self.declare_partials(of='N_rr', wrt='ax', rows=arange, cols=arange)

        self.declare_partials(of='N_fl', wrt='ay', rows=arange, cols=arange)
        self.declare_partials(of='N_fr', wrt='ay', rows=arange, cols=arange)
        self.declare_partials(of='N_rl', wrt='ay', rows=arange, cols=arange)
        self.declare_partials(of='N_rr', wrt='ay', rows=arange, cols=arange)



    def compute(self, inputs, outputs):
        M = inputs['M']
        g = inputs['g']
        a = inputs['a']
        b = inputs['b']
        ax = inputs['ax']
        ay = inputs['ay']
        h = inputs['h']
        chi = inputs['chi']
        tw = inputs['tw']

        outputs['N_fl'] = (M*g/2)*(b/(a+b))+(M*g/4)*((-(ax*h)/(a+b))+(ay*chi*h/tw))
        outputs['N_fr'] = (M*g/2)*(b/(a+b))+(M*g/4)*((-(ax*h)/(a+b))-(ay*chi*h/tw))
        outputs['N_rl'] = (M*g/2)*(a/(a+b))+(M*g/4)*(((ax*h)/(a+b))+(ay*(1-chi)*h/tw))
        outputs['N_rr'] = (M*g/2)*(a/(a+b))+(M*g/4)*(((ax*h)/(a+b))-(ay*(1-chi)*h/tw))

    def compute_partials(self, inputs, jacobian):
        M = inputs['M']
        g = inputs['g']
        a = inputs['a']
        b = inputs['b']
        ax = inputs['ax']
        ay = inputs['ay']
        h = inputs['h']
        chi = inputs['chi']
        tw = inputs['tw']

        jacobian['N_fl', 'ax'] = -(M*g*h)/(4*(a+b))
        jacobian['N_fr', 'ax'] = -(M*g*h)/(4*(a+b))
        jacobian['N_rl', 'ax'] = (M*g*h)/(4*(a+b))
        jacobian['N_rr', 'ax'] = (M*g*h)/(4*(a+b))

        jacobian['N_fl','ay'] = (M*g*chi*h)/(4*tw)
        jacobian['N_rl','ay'] = (M*g*(1-chi)*h)/(4*tw)
        jacobian['N_fr','ay'] = -(M*g*chi*h)/(4*tw)
        jacobian['N_rr','ay'] = -(M*g*(1-chi)*h)/(4*tw)
        








