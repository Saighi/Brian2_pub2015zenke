#!/usr/bin/env python
# coding: utf-8
import pickle
from brian2 import __version__ as brian2version
from brian2 import *
import subprocess
from sys import argv
from os.path import expanduser
import random as rand
#from recovered_network import *
#import numpy as np_

sim_id = argv[-1]

set_device('cpp_standalone', build_on_run=False)

#data_dir = expanduser('~/data/hebb/sim/')
data_dir = "/mnt/data2/paul_data/"

@implementation('cpp','''
// Note that functions always need a return value at the moment
double store_spike(int i, double t, int E) {
    static std::ofstream spike_file("'''+data_dir+'''spikes'''+sim_id+'''"+std::to_string(E)+".txt");  // opens the file the first time
    spike_file << i << " " << t << std::endl;
    return 0.;  // unused
}
''')

# @check_units(i=1, t=second, result=1)
# def store_spikeE(i, t):
#     raise NotImplementedError('Use standalone mode')
@check_units(i=1, t=second,E=1, result=1)
def store_spike(i, t, E):
    raise NotImplementedError('Use standalone mode')

cpp_code_dir = data_dir + 'tmp/' + sim_id

# TODO: parallelization doesn't work out-of-the-box
# with globally shared factor H (see below); tests suggested that for
# the typical network size, parallelization with atomic clause or reducing
# for loop etc is actually slower than a single-core simulation...

#prefs.devices.cpp_standalone.openmp_threads = 4


# 0) parameter initialization
# ===========================

# membrane potential dynamics
# ===========================
# timescale + reversal potentials
tau = 20*ms
Urest = -60 * mV
Uexc = 0 * mV
Uinh = -80*mV

# excitatory currents
tauampa = 5*ms
taunmda = 100*ms
aE = 0.2
aI = 0.3

# inhibitory currents
tauI = 10*ms


# adaptation currents
#delta = 1
delta = 0.1
taua = 100*ms

# spike threshold adaptation
tauthr = 2*ms
Uthr = -50*mV
thetaspi = 50*mV

# plasticity parameters
# =====================
# short-term plasticity
U1 = 0.2
taud = 200*ms
tauf = 600*ms

# exc. STDP (long-term plasticity)
# --------------------------------
# timescales of spike traces
tauz = 20*ms
tauslow = 100*ms
A = 1e-3  # 1e-3
B = 1e-3
beta = 0.05
delt = 2e-5 # 2e-5

# consolidation dynamics
tauc = 20*60*second
P = 17
wP = 0.5

# inhibitory plasticity
# ---------------------
#tauistdp = 20*ms   # NOTE: same as tauz -> use z trace
tauH = 10*second
eta = 2e-5
gamma = 4*Hz # --> convert into correct global reference...


# further synaptic parameters
# ===========================

# connection probabilities
# ------------------------

p = 0.1
pstim=0.05

# initial weights
# ---------------

wEE0 = 0.1
wEI0 = 0.2
wIE0 = 0.15
wII0 = 0.2

wStimE = 0.5


# stimulation parameters
# ======================

# baseline firing frequency of external input
# -------------------------------------------

Fstim = 10*Hz


# calculate approximate average values for initialization
# =======================================================

# --> exp. mean values for STP
ustarE = U1*(1/tauf+gamma)/(1/tauf+U1*gamma)
xstarE = 1/(1+taud*gamma*ustarE)

# --> exp. mean values for STP
ustarStim = U1*(1/tauf+Fstim)/(1/tauf+U1*Fstim)
xstarStim = 1/(1+taud*Fstim*ustarStim)


# simulation parameters
# =====================

# network size
NE = 4096
NI = 1024

# --> effective reference and learning rate for global plasticity factor
H_ref  = tauH*gamma*NE
eta_eff = eta/H_ref

duration = float(argv[1]) * second

# for _ in range(2):


# 1) Network creation
# ===================
start_scope()


# equations governing neuronal dynamics
# -------------------------------------

eqsE = '''
dU/dt = (Urest-U)/tau + gE*(Uexc-U)/tau + (gI+ga)*(Uinh-U)/tau : volt
dgI/dt = -gI/tauI : 1
dga/dt = -ga/taua : 1
gE = aE*gampa + (1-aE)*gnmda : 1
dgampa/dt = -gampa/tauampa : 1
dgnmda/dt = (-gnmda+gampa)/taunmda : 1
dtheta/dt = (Uthr-theta)/tauthr : volt
dz/dt = -z/tauz : 1
dzslow/dt = -zslow/tauslow : 1
dh/dt = -h/tauH : 1
H : 1 (shared)
dx/dt = (1-x)/taud : 1 
du/dt = (U1-u)/tauf : 1 
'''
# FAUX


eqsI = '''
dU/dt = (Urest-U)/tau + gE*(Uexc-U)/tau + gI*(Uinh-U)/tau : volt
dgI/dt = -gI/tauI : 1
gE = aI*gampa + (1-aI)*gnmda : 1
dgampa/dt = -gampa/tauampa : 1
dgnmda/dt = (-gnmda+gampa)/taunmda : 1
dtheta/dt = (Uthr-theta)/tauthr : volt
dz/dt = -z/tauz : 1
'''


# neuron groups
# -------------

# create external input neurons
Input = NeuronGroup(NE,
                    '''
                    dx/dt = (1-x)/taud : 1 
                    du/dt = (U1-u)/tauf : 1 
                    rates = Fstim : Hz
                    dz/dt = -z/tauz : 1''',
                    threshold='rand()<rates*dt',
                    reset='''
                    z += 1
                    x += -u*x
                    u += U1*(1-u)
                    ''',
                    method='euler')
Input.x = xstarStim
Input.u = ustarStim

# excitatory neuron group
E = NeuronGroup(NE,
                eqsE,
                threshold='''U>theta''',
                reset='''theta=thetaspi
                        U=Urest
                        ga+=delta
                        zslow += 1
                        z +=1
                        x += -u*x
                        u += U1*(1-u)
                        E = 1 
                        h += 1
                        dummy_var = store_spike(i,t,E)
                        ''',
                method='euler')



# variable initialization

E.U = 'rand()*(Uthr-Urest) + Urest'
E.theta = Uthr
E.h = tauH*gamma
E.H = tauH*gamma*NE
E.x = xstarE
E.u = ustarE


# inhibitory neuron group
I = NeuronGroup(NI,
                eqsI,
                threshold='U>theta',
                reset='''theta=thetaspi
                        U=Urest
                        z += 1
                        E = 0 
                        dummy_var = store_spike(i,t,E)
                        ''',
                method='euler')

# variable initialization

I.U = 'rand()*(Uthr-Urest) + Urest'
I.theta = Uthr

# initialize conductances with expected mean values
E.gampa = tauampa*NE*(ustarStim*xstarStim*Fstim*pstim*wStimE +
                    ustarE*xstarE*gamma*wEE0*p)
E.gnmda = E.gampa
E.gI = tauI*NI*wIE0*p*gamma
E.ga = taua*delta*gamma

I.gampa = tauampa*NE*(ustarStim*xstarStim*Fstim*pstim*wStimE +
                    ustarE*xstarE*gamma*wEE0*p)
I.gnmda = I.gampa
I.gI = tauI*NI*wIE0*p*gamma


# Synapses
# ========

# shared synapse equations
# ------------------------



eq_cons = '''dwt/dt = 1/tauc*(w-wt-P*wt*(wP/2-wt)*(wP-wt)) : 1 (clock-driven)
        '''

exc_syn_eqs_model = eq_cons + '''w : 1'''

exc_syn_eqs_on_pre = '''dw = - B* z_post + delt
                                        w = clip(w + dw, 0, 5)
                                        gampa_post += w*x_pre*u_pre'''

exc_syn_eqs_on_post = '''dw = A * z_pre*zslow_post - beta*(w-wt)*z_post**3
                        w = clip(w + dw , 0, 5)'''


SInputE = Synapses(Input, E,
                model=exc_syn_eqs_model,
                on_pre=exc_syn_eqs_on_pre,
                on_post=exc_syn_eqs_on_post,
                delay=0.8*ms,
                dt=1.2*second,
                method='euler')

SInputE.connect(p=pstim)

SInputE.w = wStimE

SEE = Synapses(E, E,
            model=exc_syn_eqs_model,
            on_pre=exc_syn_eqs_on_pre,
            on_post=exc_syn_eqs_on_post,
            delay=0.8*ms,
            dt=1.2*second,
            method='euler')

SEE.connect(condition='i!=j', p=p)

SEE.w = wEE0


# Exc. synapses on inhibitory neurons exhibit STP but no other forms of
# plasticity.

SEI = Synapses(E, I,
            on_pre='''gampa_post += wEI0*x_pre*u_pre''',)

SEI.connect( p=p)


#SEI.x = 1
#SEI.u = U1


# Inh. synapses on exc. neurons show global, homeostatic plasticity to ensure
# roughly constant average firing rates.

SIE = Synapses(I, E,
            model='''w : 1''',
            on_pre='''dw = eta_eff*(H_post-H_ref)*(z_post+1)
                        w = clip(w + dw, 0, 5)
                        gI_post += w''',
            on_post='''dw = eta_eff*(H_post-H_ref)*z_pre
                        w = clip(w + dw, 0, 5) ''',
            delay=0.8*ms)

SIE.connect(p=p)
SIE.w = wIE0


# Inh. synapses on inh. neurons do not show any form of plasticity.

SII = Synapses(I, I, on_pre='gI_post += wII0')
SII.connect(condition='i!=j', p=p)



# 3) Set up recording
# ===================

# approx. number of synapses

secmargin = 1.01
nInputEupperest = int(secmargin*NE**2*pstim)
nEEupperest = int(secmargin*NE**2*p)
nIEupperest = int(secmargin*NI*NE*p)


dt_weightdist = 30. * second
dt_singleweights = 0.1 * second
nsinglew = 10
inds_singlew = int(nEEupperest/secmargin/nsinglew)*arange(nsinglew, dtype=int)


#Recording

StateMonE = StateMonitor(E, ['H'], record=range(1),dt = 10*second)
SpikeMonE = SpikeMonitor(E)

# StateMonI = StateMonitor(I, ['U','gI','gampa','gnmda','theta'],record=range(10))
SpikeMonI = SpikeMonitor(I)

# SpikeMonInput = SpikeMonitor(Input)

# StateMonEE = StateMonitor(SEE, ['w','wt','x','u'], record=inds_singlew, dt=defaultclock.dt)

weightMonInputE = StateMonitor(SInputE, ['w','wt'],
                            record=rand.sample(list(range(nInputEupperest)),nInputEupperest//2),
                            dt=dt_weightdist)
weightMonEE = StateMonitor(SEE, ['w','wt'], record=rand.sample(list(range(nEEupperest)),nEEupperest//2),
                        dt=dt_weightdist)
weightMonIE = StateMonitor(SIE, ['w'], record=rand.sample(list(range(nIEupperest)),nIEupperest//2),
                        dt=dt_weightdist)

net = Network(SpikeMonI,SpikeMonE,StateMonE,weightMonInputE,weightMonIE,weightMonEE,Input,E,I,SInputE,SII,SIE,SEI,SEE)

# 4) Create c++ executable and run simulation
# ===========================================
# for _ in range(2):

#     net.run(duration, report='stdout', report_period=10*second)

net.run(duration, report='stdout', report_period=10*second)

device.build(directory=cpp_code_dir, compile=False, run=False, debug=False)


# insertions for global variable H
# NOTE: line numbers according to .cpp file generated by brian 2.2.2.1

comment = '// the following line wass added to implement global H variable\n'
insertions = [' double* __restrict  _ptr_array_neurongroup_1_H = _array_neurongroup_1_H;\n',
              ' double H = 0.0;\n',
              '        H += h;\n',
              '    _ptr_array_neurongroup_1_H[0] = H;\n\n']

insert_at_lines = {'2.1.2' : [270, 288, 324, 336],
                   '2.2.2.1' : [102, 120, 155, 167],
                  '2.4.2' : [92,120,151,167]}
# modify E stateupdater to compute global variable
with open(cpp_code_dir+'/code_objects/neurongroup_1_stateupdater_codeobject.cpp', 'r+') as f:
    lines = f.readlines()
    offset = 0
    for line_number, text in zip(insert_at_lines[brian2version], insertions):
        lines.insert(line_number + offset, comment + text)
        offset += 1
    f.seek(0)
    f.writelines(lines)


# compile cpp code
# ----------------
subprocess.call(['cd %s; make'%(cpp_code_dir, )], shell=True)

# run device
# ----------
device.run(cpp_code_dir, True, [])


# save data
# ---------

nInputE = len(SInputE)
nEE = len(SEE)
nIE = len(SIE)


np.savez_compressed(data_dir+'%s-weights.npz'%sim_id,
                    InputE=weightMonInputE.w[:nInputE],
                    EE=weightMonEE.w[:nEE],
                    IE=weightMonIE.w[:nIE])
np.savez_compressed(data_dir+'%s-reference_weights.npz'%sim_id,
                    InputE=weightMonInputE.wt[:nInputE],
                    EE=weightMonEE.wt[:nEE])
np.savez_compressed(data_dir+'%s-H.npz'%sim_id,
                    H=StateMonE.H)
#gather(sim_id,data_dir)
