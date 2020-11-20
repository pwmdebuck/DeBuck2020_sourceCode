import openmdao.api as om
import numpy as np

class CarODE(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        #constants
        self.add_input('M', val=1184.0, desc='mass', units='kg')
        self.add_input('a', val=1.404, desc='cg to front distance', units='m')
        self.add_input('b', val=1.356, desc='cg to rear distance', units='m')
        self.add_input('tw', val=0.807, desc='half track width', units='m')
        self.add_input('Iz', val=1775.0, desc='yaw inertia', units='kg*m**2')
        self.add_input('rho', val=1.2, desc='air density', units='kg/m**3')
        self.add_input('kappa', val=np.zeros(nn), desc='track curvature', units='1/m')
        self.add_input('S_w', val=0.8, desc='wing planform area', units='m**2')

        #states
        self.add_input('n', val=np.zeros(nn), desc='distance perpendicular to centerline', units='m')
        self.add_input('alpha', val=np.zeros(nn), desc='angle relative to centerline', units='rad')
        self.add_input('V', val=np.zeros(nn), desc='speed', units='m/s')
        self.add_input('lambda', val=np.zeros(nn), desc='body slip angle', units='rad')
        self.add_input('omega', val=np.zeros(nn), desc='yaw rate', units='rad/s')

        #tire loads
        self.add_input('S_fl', val=np.zeros(nn), desc='longitudinal force fl', units='N')
        self.add_input('S_fr', val=np.zeros(nn), desc='longitudinal force fr', units='N')
        self.add_input('S_rl', val=np.zeros(nn), desc='longitudinal force rl', units='N')
        self.add_input('S_rr', val=np.zeros(nn), desc='longitudinal force rr', units='N')

        self.add_input('F_fl', val=np.zeros(nn), desc='lateral force fl', units='N')
        self.add_input('F_fr', val=np.zeros(nn), desc='lateral force fr', units='N')
        self.add_input('F_rl', val=np.zeros(nn), desc='lateral force rl', units='N')
        self.add_input('F_rr', val=np.zeros(nn), desc='lateral force rr', units='N')

        #controls
        self.add_input('delta', val=np.zeros(nn), desc='steering angle', units='rad')
        self.add_input('gamma', val=np.zeros(nn), desc='wing angle', units='deg')

        #outputs
        self.add_output('sdot', val=np.zeros(nn), desc='distance along track', units='m/s')
        self.add_output('ndot', val=np.zeros(nn), desc='distance perpendicular to centerline', units='m/s')
        self.add_output('alphadot', val=np.zeros(nn), desc='angle relative to centerline', units='rad/s')
        self.add_output('Vdot', val=np.zeros(nn), desc='speed', units='m/s**2')
        self.add_output('lambdadot', val=np.zeros(nn), desc='body slip angle', units='rad/s')
        self.add_output('omegadot', val=np.zeros(nn), desc='yaw rate', units='rad/s**2')
        #self.add_output('kappa', val=np.zeros(nn), desc='track curvature', units='1/m')

        #constraint (power)
        self.add_output('powerFL', val=np.zeros(nn),desc='tractive power',units='W')
        self.add_output('powerFR', val=np.zeros(nn),desc='tractive power',units='W')
        self.add_output('powerRL', val=np.zeros(nn),desc='tractive power',units='W')
        self.add_output('powerRR', val=np.zeros(nn),desc='tractive power',units='W')

        # Setup partials
        arange = np.arange(self.options['num_nodes'], dtype=int)

        #sdot
        self.declare_partials(of='sdot', wrt='V', rows=arange, cols=arange)
        self.declare_partials(of='sdot', wrt='alpha', rows=arange, cols=arange)
        self.declare_partials(of='sdot', wrt='lambda', rows=arange, cols=arange)
        self.declare_partials(of='sdot', wrt='n', rows=arange, cols=arange)
        self.declare_partials(of='sdot', wrt='kappa', rows=arange, cols=arange)

        #ndot
        self.declare_partials(of='ndot', wrt='V', rows=arange, cols=arange)
        self.declare_partials(of='ndot', wrt='alpha', rows=arange, cols=arange)
        self.declare_partials(of='ndot', wrt='lambda', rows=arange, cols=arange)

        #alphadot
        self.declare_partials(of='alphadot', wrt='V', rows=arange, cols=arange)
        self.declare_partials(of='alphadot', wrt='alpha', rows=arange, cols=arange)
        self.declare_partials(of='alphadot', wrt='lambda', rows=arange, cols=arange)
        self.declare_partials(of='alphadot', wrt='n', rows=arange, cols=arange)
        self.declare_partials(of='alphadot', wrt='omega', rows=arange, cols=arange)
        self.declare_partials(of='alphadot', wrt='kappa', rows=arange, cols=arange)

        #vdot
        self.declare_partials(of='Vdot', wrt='S_fl', rows=arange, cols=arange)
        self.declare_partials(of='Vdot', wrt='S_fr', rows=arange, cols=arange)
        self.declare_partials(of='Vdot', wrt='S_rl', rows=arange, cols=arange)
        self.declare_partials(of='Vdot', wrt='S_rr', rows=arange, cols=arange)

        self.declare_partials(of='Vdot', wrt='F_fr', rows=arange, cols=arange)
        self.declare_partials(of='Vdot', wrt='F_fl', rows=arange, cols=arange)

        self.declare_partials(of='Vdot', wrt='V', rows=arange, cols=arange)
        self.declare_partials(of='Vdot', wrt='omega', rows=arange, cols=arange)
        self.declare_partials(of='Vdot', wrt='lambda', rows=arange, cols=arange)
        self.declare_partials(of='Vdot', wrt='delta', rows=arange, cols=arange)
        self.declare_partials(of='Vdot',wrt='gamma',rows=arange,cols=arange)

        #lambdadot
        self.declare_partials(of='lambdadot', wrt='F_fl', rows=arange, cols=arange)
        self.declare_partials(of='lambdadot', wrt='F_fr', rows=arange, cols=arange)
        self.declare_partials(of='lambdadot', wrt='F_rl', rows=arange, cols=arange)
        self.declare_partials(of='lambdadot', wrt='F_rr', rows=arange, cols=arange)

        self.declare_partials(of='lambdadot', wrt='S_fr', rows=arange, cols=arange)
        self.declare_partials(of='lambdadot', wrt='S_fl', rows=arange, cols=arange)
        self.declare_partials(of='lambdadot', wrt='S_rl', rows=arange, cols=arange)
        self.declare_partials(of='lambdadot', wrt='S_rr', rows=arange, cols=arange)

        self.declare_partials(of='lambdadot', wrt='V', rows=arange, cols=arange)
        self.declare_partials(of='lambdadot', wrt='omega', rows=arange, cols=arange)
        self.declare_partials(of='lambdadot', wrt='lambda', rows=arange, cols=arange)
        self.declare_partials(of='lambdadot', wrt='delta', rows=arange, cols=arange)
        self.declare_partials(of='lambdadot', wrt='gamma', rows=arange, cols=arange)

        #omegadot
        self.declare_partials(of='omegadot', wrt='S_fl', rows=arange, cols=arange)
        self.declare_partials(of='omegadot', wrt='S_fr', rows=arange, cols=arange)
        self.declare_partials(of='omegadot', wrt='S_rl', rows=arange, cols=arange)
        self.declare_partials(of='omegadot', wrt='S_rr', rows=arange, cols=arange)

        self.declare_partials(of='omegadot', wrt='F_fr', rows=arange, cols=arange)
        self.declare_partials(of='omegadot', wrt='F_fl', rows=arange, cols=arange)
        self.declare_partials(of='omegadot', wrt='F_rl', rows=arange, cols=arange)
        self.declare_partials(of='omegadot', wrt='F_rr', rows=arange, cols=arange)

        self.declare_partials(of='powerRL', wrt='V', rows=arange, cols=arange)
        self.declare_partials(of='powerRR', wrt='V', rows=arange, cols=arange)
        self.declare_partials(of='powerFL', wrt='V', rows=arange, cols=arange)
        self.declare_partials(of='powerFR', wrt='V', rows=arange, cols=arange)
        self.declare_partials(of='powerRR', wrt='S_rr', rows=arange, cols=arange)
        self.declare_partials(of='powerRL', wrt='S_rl', rows=arange, cols=arange)
        self.declare_partials(of='powerFL', wrt='S_fl', rows=arange, cols=arange)
        self.declare_partials(of='powerFR', wrt='S_fr', rows=arange, cols=arange)

        # self.declare_partials(of='kappa', wrt='s', rows=arange, cols=arange)




    def compute(self, inputs, outputs):
        omega = inputs['omega']
        V = inputs['V']
        n = inputs['n']
        lamb = inputs['lambda']
        delta = inputs['delta']
        alpha = inputs['alpha']
        S_fl = inputs['S_fl']
        S_fr = inputs['S_fr']
        S_rl = inputs['S_rl']
        S_rr = inputs['S_rr']
        F_fl = inputs['F_fl']
        F_fr = inputs['F_fr']
        F_rl = inputs['F_rl']
        F_rr = inputs['F_rr']
        kappa = inputs['kappa']
        M = inputs['M']
        rho = inputs['rho']
        gamma = inputs['gamma']
        Iz = inputs['Iz']
        a = inputs['a']
        b = inputs['b']
        tw = inputs['tw']
        S_w = inputs['S_w']
        gamma = inputs['gamma']

        #compute kappa based on s
        # k1 = 0.01/(1+np.exp(-0.5*(s-120)))
        # k2 = 0.01/(1+np.exp(0.5*(s-180)))
        # k1 = -0.1/(1+np.exp(-0.3*(s-40)))
        # k2 = -0.1/(1+np.exp(0.3*(s-70)))
        # # kappa = np.minimum(k1,k2)

        # kappa1 = np.where(np.abs(k1) > np.abs(k2), k2, k1)

        # k1 = 0.1/(1+np.exp(-0.3*(s-110)))
        # k2 = 0.1/(1+np.exp(0.3*(s-140)))
        # kappa2 = np.where(np.abs(k1) > np.abs(k2), k2, k1)

        # kappa = np.where(np.abs(kappa1)<np.abs(kappa2),kappa2,kappa1)
        # print('s',s)


        F_all = F_fl+F_fr+F_rl+F_rr
        S_all = S_fl+S_fr+S_rl+S_rr
        F_front = F_fl+F_fr
        S_front = S_fl+S_fr

        #drag equation
        CdA = 1.055-7.588e-4*V-9.156e-6*V**2
        drag = 0.5*rho*CdA*V**2

        Cd_wing = 0.0667+0.0127*gamma
        drag_wing = 0.5*rho*Cd_wing*S_w*V**2


        outputs['sdot'] = (V*np.cos(alpha-lamb))/(1-n*kappa)
        outputs['ndot'] = V*np.sin(alpha-lamb)
        outputs['alphadot'] = omega-(kappa*V*np.cos(alpha-lamb))/(1-n*kappa)
        # Vdot = (S_fl+S_rr+S_fr+S_rl)/M - delta*(F_fr+F_fl)/M - (rho*CdA*V**2)/(2*M) - omega*V*lamb
        Vdot = S_all/M - delta*F_front/M - (drag+drag_wing)/M - omega*V*lamb
        outputs['Vdot'] = Vdot
        outputs['lambdadot'] = omega-Vdot*lamb/V - delta*S_front/(M*V) - F_all/(M*V)
        outputs['omegadot'] = (a*(F_fr+F_fl))/Iz - (b*(F_rr+F_rl))/Iz + (tw*(-S_rr+S_rl-S_fr+S_fl))/Iz
        outputs['powerFL'] = V*S_fl
        outputs['powerFR'] = V*S_fr
        outputs['powerRL'] = V*S_rl
        outputs['powerRR'] = V*S_rr

    def compute_partials(self, inputs, jacobian):
        omega = inputs['omega']
        V = inputs['V']
        n = inputs['n']
        lamb = inputs['lambda']
        delta = inputs['delta']
        alpha = inputs['alpha']
        S_fl = inputs['S_fl']
        S_fr = inputs['S_fr']
        S_rl = inputs['S_rl']
        S_rr = inputs['S_rr']
        F_fl = inputs['F_fl']
        F_fr = inputs['F_fr']
        F_rl = inputs['F_rl']
        F_rr = inputs['F_rr']
        kappa = inputs['kappa']
        M = inputs['M']
        rho = inputs['rho']
        Iz = inputs['Iz']
        a = inputs['a']
        b = inputs['b']
        tw = inputs['tw']
        S_w = inputs['S_w']
        gamma = inputs['gamma']

        F_all = F_fl+F_fr+F_rl+F_rr
        S_all = S_fl+S_fr+S_rl+S_rr
        F_front = F_fl+F_fr
        S_front = S_fl+S_fr

        #drag equation
        CdA = 1.055-7.588e-4*V-9.156e-6*V**2
        drag = 0.5*rho*CdA*V**2

        Cd_wing = 0.0667+0.0127*gamma
        drag_wing = 0.5*rho*Cd_wing*S_w*V**2

        ddrag_dv = 2.110*V-2.2764e-3*V**2-3.6624e-5*V**3

        # k1 = -0.1/(1+np.exp(-0.1*(s-40)))#+0.1/(1+np.exp(-0.1*(s-80)))
        # k2 = -0.1/(1+np.exp(0.1*(s-60)))#+0.1/(1+np.exp(0.1*(s-100)))
        # #kappa = np.minimum(k1,k2)
        # kappa = np.where(np.abs(k1) > np.abs(k2), k2, k1)

        jacobian['sdot', 'V'] = np.cos(alpha-lamb)/(1-n*kappa)
        jacobian['sdot', 'alpha'] = (V*np.sin(alpha-lamb))/(kappa*n-1)
        jacobian['sdot', 'lambda'] = -(V*np.sin(alpha-lamb))/(kappa*n-1)
        jacobian['sdot', 'n'] = (kappa*V*np.cos(alpha-lamb))/(1-kappa*n)**2
        jacobian['sdot','kappa'] = n*V*np.cos(alpha - lamb)/(1-kappa*n)**2

        jacobian['ndot','V'] = np.sin(alpha-lamb)
        jacobian['ndot','alpha'] = V*np.cos(alpha-lamb)
        jacobian['ndot','lambda'] = -V*np.cos(alpha-lamb)

        jacobian['alphadot','omega'] = 1.0
        jacobian['alphadot','V'] = -(kappa*np.cos(alpha-lamb))/(1-n*kappa)
        jacobian['alphadot','alpha'] = (kappa*V*np.sin(alpha-lamb))/(1-kappa*n)
        jacobian['alphadot','lambda'] = (kappa*V*np.sin(alpha-lamb))/(kappa*n-1)
        jacobian['alphadot','n'] = -(kappa**2*V*np.cos(alpha-lamb))/((1-kappa*n)**2)
        jacobian['alphadot','kappa'] = -(V*np.cos(alpha - lamb))/((1-kappa*n)**2)

        jacobian['Vdot','S_rr'] = 1/M
        jacobian['Vdot','S_rl'] = 1/M
        jacobian['Vdot','S_fr'] = 1/M
        jacobian['Vdot','S_fl'] = 1/M
        jacobian['Vdot','gamma'] = -(1/M)*0.5*rho*0.0127*S_w*V**2

        jacobian['Vdot','delta'] = -(F_fr+F_fl)/M
        jacobian['Vdot','F_fr'] = -delta/M
        jacobian['Vdot','F_fl'] = -delta/M

        jacobian['Vdot','V'] = -(0.5*rho*ddrag_dv+rho*Cd_wing*S_w*V)/M - omega*lamb
        jacobian['Vdot','omega'] = -V*lamb
        jacobian['Vdot','lambda'] = -omega*V

        jacobian['omegadot','F_fr'] = a/Iz
        jacobian['omegadot','F_fl'] = a/Iz
        jacobian['omegadot','F_rr'] = -b/Iz
        jacobian['omegadot','F_rl'] = -b/Iz
        jacobian['omegadot','S_rr'] = -tw/Iz
        jacobian['omegadot','S_rl'] = tw/Iz
        jacobian['omegadot','S_fr'] = -tw/Iz
        jacobian['omegadot','S_fl'] = tw/Iz

        jacobian['lambdadot','omega'] = 1+lamb**2
        jacobian['lambdadot','gamma'] = 0.5*rho*0.0127*S_w*V*lamb/M
        #jacobian['lambdadot','lambda'] = -(((S_fl+S_fr+S_rl+S_rr)/(M*V))-(delta*(F_fr+F_fl)/(M*V))-(rho*CdA*V/(2*M)))+2*lamb*omega
        #jacobian['lambdadot','lambda'] = 2*lamb*omega + ((CdA*rho*V**2)/2 - (S_fr+S_fl+S_rr+S_rl) + delta*(F_fr+F_fl))/(M*V)
        #jacobian['lambdadot','lambda'] = -S_all/M+delta*F_front/(M*V)+(drag+drag_wing)/(M*V)+2*lamb*omega
        jacobian['lambdadot','lambda'] = 2.0*lamb*omega + (F_front*delta - 1.0*S_all + 0.5275*V**2*rho - 0.0003794*V**3*rho - 0.000004578*V**4*rho + 0.03385*S_w*V**2*rho + 0.00635*S_w*V**2*gamma*rho)/(M*V)
        jacobian['lambdadot','delta'] = (lamb*(F_fr+F_fl)/(M*V))-(S_fr+S_fl)/(M*V) #changed sign
        jacobian['lambdadot','F_fr'] = (lamb*delta/(M*V))-1/(M*V)
        jacobian['lambdadot','F_fl'] = (lamb*delta/(M*V))-1/(M*V)
        jacobian['lambdadot','F_rr'] = -1/(M*V)
        jacobian['lambdadot','F_rl'] = -1/(M*V)
        jacobian['lambdadot','S_fr'] = -(lamb+delta)/(M*V) #changed sign
        jacobian['lambdadot','S_fl'] = -(lamb+delta)/(M*V) #changed sign
        jacobian['lambdadot','S_rr'] = -lamb/(M*V)
        jacobian['lambdadot','S_rl'] = -lamb/(M*V)
        # jacobian['lambdadot','V'] = (lamb*(rho*CdA*V**2+2*(S_fl+S_fr+S_rl+S_rr))+2*delta*(lamb*(F_fr+F_fl)+S_fl+S_fr)+2*(F_fl+F_fr+F_rl+F_rr))/(2*M*V**2)

        #ddrag_dv_2 = 1.055-1.5176e-3*V-2.7468e-5*V**2
        #jacobian['lambdadot','V'] = (F_all + delta*S_front + lamb*S_all - delta*F_front*lamb)/(M*V**2) + (CdA*lamb*rho)/(2*M)
        jacobian['lambdadot','V'] = (F_all + S_front*delta + S_all*lamb - 1.0*F_front*delta*lamb + 0.5275*V**2*lamb*rho - 0.0007588*V**3*lamb*rho - 0.000013734*V**4*lamb*rho + 0.03385*S_w*V**2*lamb*rho + 0.00635*S_w*V**2*gamma*lamb*rho)/(M*V**2)

        jacobian['powerRR','V'] = S_rr
        jacobian['powerRL','V'] = S_rl
        jacobian['powerFR','V'] = S_fr
        jacobian['powerFL','V'] = S_fl
        jacobian['powerFL','S_fl'] = V
        jacobian['powerFR','S_fr'] = V
        jacobian['powerRR','S_rr'] = V
        jacobian['powerRL','S_rl'] = V







