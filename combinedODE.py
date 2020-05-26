import openmdao.api as om
import numpy as np
from carODEsimple import CarODE
from tireODE import TireODE
from normalForceODE import NormalForceODE
from accelerationODE import AccelerationODE

class CombinedODE(om.Group):

    def initialize(self):
        self.options.declare('num_nodes', types=int,
                             desc='Number of nodes to be evaluated in the RHS')

    def setup(self):
        nn = self.options['num_nodes']


        self.add_subsystem(name='normal',
                           subsys=NormalForceODE(num_nodes=nn))

        self.add_subsystem(name='tire',
                           subsys=TireODE(num_nodes=nn))

        self.add_subsystem(name='car',
                           subsys=CarODE(num_nodes=nn))

        self.add_subsystem(name='accel',
                           subsys=AccelerationODE(num_nodes=nn))

        self.connect('normal.N_fr','tire.N_fr')
        self.connect('normal.N_fl','tire.N_fl')
        self.connect('normal.N_rr','tire.N_rr')
        self.connect('normal.N_rl','tire.N_rl')

        self.connect('tire.S_fr','car.S_fr')
        self.connect('tire.S_fl','car.S_fl')
        self.connect('tire.S_rr','car.S_rr')
        self.connect('tire.S_rl','car.S_rl')

        self.connect('tire.F_fr','car.F_fr')
        self.connect('tire.F_fl','car.F_fl')
        self.connect('tire.F_rr','car.F_rr')
        self.connect('tire.F_rl','car.F_rl')

        self.connect('car.Vdot','accel.Vdot')
        self.connect('car.lambdadot','accel.lambdadot')