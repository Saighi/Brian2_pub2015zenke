#!/usr/bin/env python
# coding: utf-8
import pickle
from brian2 import __version__ as brian2version
from brian2 import *
import subprocess
from sys import argv
from os.path import expanduser
import random as rand
import numpy as np
#from recovered_network import *
#import numpy as np_

sim_id = argv[-1]

set_device('cpp_standalone', build_on_run=False)

#data_dir = expanduser('~/data/hebb/sim/')
data_dir = "/mnt/data2/paul_data/"

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
Urest = -70 * mV #MODIFIE
Uexc = 0 * mV
Uinh = -80*mV

# excitatory currents
tauampa = 5*ms
taunmda = 100*ms
ratio = 0.2
ampa_ratio = ratio/(ratio+1.0)
nmda_ratio = 1./(ratio+1.0)
aI = 0.3

# inhibitory currents
tauI = 10*ms


# adaptation currents
#delta = 1
delta = 0.1
taua = 100*ms

# spike threshold adaptation
tauthr = 5*ms #MODIFIE
Uthr = -50*mV
thetaspi = 100*mV  #MODIFIE

# plasticity parameters
# =====================
# short-term plasticity
U1 = 0.2
taud = 150*ms #MODIFIE
tauf = 600*ms

# exc. STDP (long-term plasticity)
# --------------------------------
# timescales of spike traces
tauz = 20*ms
tauslow = 100*ms
A = 1e-3  # 1e-3
B = 1e-3
beta = 0.05 # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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
#p = 0.05 #MODIFIE
pstim=0.05 #MODIFIE

# initial weights
# ---------------

#wEE0 = 0.1

wStimE = 0.2

# stimulation parameters
# ======================

# baseline firing frequency of external input
# -------------------------------------------

Fstim = 10*Hz #CHANGE

# calculate approximate average values for initialization
# =======================================================


# CHANGE !!!!


# --> exp. mean values for STP
#ustarE = U1*(1/tauf+gamma)/(1/tauf+U1*gamma)
#xstarE = 1/(1+taud*gamma*ustarE)
ustarE=U1 #MODIFIE
xstarE=1
# --> exp. mean values for STP
#ustarStim = U1*(1/tauf+Fstim)/(1/tauf+U1*Fstim)
#xstarStim = 1/(1+taud*Fstim*ustarStim)
#ustarStim=U1
ustarStim=U1 #MODIFIE
xstarStim=1



# simulation parameters
# =====================

# network size
NE = 200

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
dga/dt = -ga/taua : 1
dgampa/dt = -gampa/tauampa : 1
dgnmda/dt = (-gnmda+gampa)/taunmda : 1
dtheta/dt = (-theta)/tauthr : volt
gE = ampa_ratio*gampa + nmda_ratio*gnmda : 1
dU/dt = (Urest-U)/tau + gE*(Uexc-U)/tau + ga*(Uinh-U)/tau : volt
'''
datas = np.loadtxt("/users/nsr/saighi/orchestratedSaighi/src/data/one_neuron_spk_many_input")
my_array = datas[:,0]-0.0003
neurons = datas[:,1]

G = SpikeGeneratorGroup(NE, neurons,my_array*second)
#G = SpikeGeneratorGroup(1,[],[]*ms)

# neuron groups
# -------------

# create external input neurons
Input = NeuronGroup(NE,
                    '''
                    isSpiking : 1
                    rates = Fstim : Hz''',
                    threshold='isSpiking == 1',
                    reset='''
                    isSpiking = 0
                    ''',
                    method='euler')

# excitatory neuron group
E = NeuronGroup(1,
                eqsE,
                threshold='''U>Uthr+theta''',
                reset='''theta=thetaspi
                        U=Urest
                        ga+=delta
                        ''',
                method='rk4')

Input.isSpiking = 0

# variable initialization
# E.U = 'rand()*(Uthr-Urest) + Urest'

E.U = Urest
E.theta = 0
E.gampa = 0
E.gnmda = 0
E.ga = 0


# Generator to input synapses

SGI = Synapses(G,Input,on_pre = 'isSpiking_post=1')
SGI.connect(condition='i==j')


# Synapses
# ========

# shared synapse equations
# ------------------------


eq_cons = '''dwt/dt = 1/tauc*(w-wt-P*wt*(wP/2-wt)*(wP-wt)) : 1 (clock-driven)
        dx/dt = (1-x)/taud : 1 (event-driven)
        du/dt = (U1-u)/tauf : 1 (event-driven)
        dzpre/dt = -zpre/tauz : 1 (event-driven)
        dzpost/dt = -zpost/tauz : 1 (event-driven)
        dzSlowpost/dt = -zSlowpost/tauslow : 1 (event-driven)
        '''

exc_syn_eqs_model = eq_cons + '''w : 1'''

#Order of equation (gampa before) very important
exc_syn_eqs_on_pre = '''
                        gampa_post += w*x*u
                        dw = - B* zpost + delt
                        w = clip(w + dw, 0, 5)
                        x += -u*x
                        u += U1*(1-u)
                        zpre+=1
                        '''

exc_syn_eqs_on_post = '''dw = A * zpre*zSlowpost - beta*(w-wt)*zpost**3
                        w = clip(w + dw , 0, 5)
                        zpost+=1
                        zSlowpost+=1'''


SInputE = Synapses(Input, E,
                model=exc_syn_eqs_model,
                on_pre=exc_syn_eqs_on_pre,
                on_post=exc_syn_eqs_on_post,
                delay = 0.8 * ms,
                method='euler')

SInputE.connect(p=1)
SInputE.x = xstarE
SInputE.u = ustarE
#SInputE.w = 'clip(wStimE+randn()*0.2,0,1)'
SInputE.w = wStimE

# SEE = Synapses(E, E,
#             model=exc_syn_eqs_model,
#             on_pre=exc_syn_eqs_on_pre,
#             on_post=exc_syn_eqs_on_post,
#             delay=0.8*ms,
#             dt=1.2*second,
#             method='euler')


# SEE.connect(condition='i!=j', p=p)

# SEE.w = wEE0


# 3) Set up recording
# ===================

#Recording

SpikeMonE = SpikeMonitor(E)
SpikeMonG = SpikeMonitor(G)
SpikeMonInput = SpikeMonitor(Input)

StateMonInputE = StateMonitor(SInputE, ['w','wt','x','u'],record = range(NE))

# SpikeMonInput = SpikeMonitor(Input)

StateMonE = StateMonitor(E, ['U','ga','gampa','gnmda','theta'], record = [0],dt=0.1*ms)
StateMonInput = StateMonitor(Input, ['isSpiking'],record = [0],dt=0.1*ms)



net = Network(SpikeMonInput,SpikeMonG,G,StateMonE,SpikeMonE,Input,E,SInputE,SGI,StateMonInput,StateMonInputE)
net.schedule = ['start', 'groups', 'thresholds', 'synapses', 'resets', 'end']
# 4) Create c++ executable and run simulation
# ===========================================
# for _ in range(2):

#     net.run(duration, report='stdout', report_period=10*second)

net.run(duration, report='stdout', report_period=10*second)

device.build(directory=cpp_code_dir, compile=False, run=False, debug=False)


# compile cpp code
# ----------------
subprocess.call(['cd %s; make'%(cpp_code_dir, )], shell=True)

# run device
# ----------
device.run(cpp_code_dir, True, [])


np.savez_compressed(data_dir+'%s-E.npz'%sim_id,
                    gampa=StateMonE.gampa,
                    gnmda=StateMonE.gnmda,
                    theta=StateMonE.theta,
                    ga = StateMonE.ga,
                    U = StateMonE.U)

np.savez_compressed(data_dir+'%s-Input.npz'%sim_id,
                    isSpiking = StateMonInput.isSpiking
                    )

np.savez_compressed(data_dir+'%s-synapses_InputE.npz'%sim_id,
                    w=StateMonInputE.w,
                    wt=StateMonInputE.wt,
                    x=StateMonInputE.x,
                    u=StateMonInputE.u,
                    )

np.savez_compressed(data_dir+'%s-spikes_Input.npz'%sim_id,
                    t=SpikeMonInput.t,
                    i=SpikeMonInput.i
                    )

np.savez_compressed(data_dir+'%s-spikes_E.npz'%sim_id,
                    t=SpikeMonE.t,
                    i=SpikeMonE.i
                    )

np.savez_compressed(data_dir+'%s-spikes_G.npz'%sim_id,
                    t=SpikeMonG.t,
                    i=SpikeMonG.i
                    )

