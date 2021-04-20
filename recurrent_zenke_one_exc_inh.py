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
ratioE = 0.2
ampa_ratioE = ratioE/(ratioE+1.0)
nmda_ratioE = 1./(ratioE+1.0)
ratioI = 0.3
ampa_ratioI = ratioI/(ratioI+1.0)
nmda_ratioI = 1./(ratioI+1.0)

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
wEI0 = 0.2 *3
wStimE = 0.2
wIE0=0.2

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
Exc_nb = 40
H_ref  = tauH*gamma*Exc_nb #CHANGE !!
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
dgI/dt = -gI/tauI : 1
dgampa/dt = -gampa/tauampa : 1
dgnmda/dt = (-gnmda+gampa)/taunmda : 1
dtheta/dt = (-theta)/tauthr : volt
gE = ampa_ratioE*gampa + nmda_ratioE*gnmda : 1
dh/dt = -h/tauH : 1
H : 1 (shared)
dU/dt = (Urest-U)/tau + gE*(Uexc-U)/tau + (gI+ga)*(Uinh-U)/tau : volt
'''
eqsI = '''
dgI/dt = -gI/tauI : 1
dgampa/dt = -gampa/tauampa : 1
dgnmda/dt = (-gnmda+gampa)/taunmda : 1
dtheta/dt = (-theta)/tauthr : volt 
gE = ampa_ratioI*gampa + nmda_ratioI*gnmda : 1
dU/dt = (Urest-U)/tau + gE*(Uexc-U)/tau + gI*(Uinh-U)/tau : volt
'''


datas = np.loadtxt("/users/nsr/saighi/orchestratedSaighi/src/data/sim_one_ex_inh_neuron_many_input")
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
E = NeuronGroup(Exc_nb,
                eqsE,
                threshold='''U>Uthr+theta''',
                reset='''theta=thetaspi
                        U=Urest
                        ga+=delta
                        h += 1
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
E.h = tauH*gamma
E.H = tauH*gamma*Exc_nb

# inhibitory neuron group
I = NeuronGroup(200,
                eqsI,
                threshold='U>Uthr+theta',
                reset='''theta=thetaspi
                        U=Urest
                        ''',
                method='rk4')

# variable initialization

I.U =  Urest
I.theta = 0
I.gampa = 0
I.gnmda = 0
I.gI = 0


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

exc_syn_eqs_on_post = '''dw = A*zpre*zSlowpost - beta*(w-wt)*zpost**3
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


SEI = Synapses(E, I, model = '''
        dx/dt = (1-x)/taud : 1 (event-driven)
        du/dt = (U1-u)/tauf : 1 (event-driven)
        ''',
            on_pre='''gampa_post += wEI0*x*u
                        x += -u*x
                        u += U1*(1-u)
                        ''',
            delay=0.8*ms)  # Delay ? Yes
# CHANGE !

SEI.connect(p=1)

SEI.x = 1
SEI.u = U1


# Inh. synapses on exc. neurons show global, homeostatic plasticity to ensure
# roughly constant average firing rates.

SIE = Synapses(I, E,
            model='''w : 1 
                    dzpre/dt = -zpre/tauz : 1 (event-driven)
                    dzpost/dt = -zpost/tauz : 1 (event-driven)''',
            on_pre='''  gI_post += w
                        dw = eta_eff*(H_post-H_ref)*(zpost+1)
                        w = clip(w + dw, 0, 5)
                        zpost+=1
                        ''',
            on_post=''' dw = eta_eff*(H_post-H_ref)*zpre
                        w = clip(w + dw, 0, 5) 
                        zpre+=1
                        ''',
            delay=0.8*ms)

SIE.connect(p=1)
SIE.w = wIE0


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
SpikeMonI = SpikeMonitor(I)
SpikeMonG = SpikeMonitor(G)
SpikeMonInput = SpikeMonitor(Input)

StateMonInputE = StateMonitor(SInputE, ['w','wt','x','u'],record = [0])
StateMonIE = StateMonitor(SIE,['w'],record = range(NE))
# SpikeMonInput = SpikeMonitor(Input)

StateMonE = StateMonitor(E, ['U','ga','gampa','gnmda','theta',"gI"], record = [0])
StateMonI = StateMonitor(I, ['U','gampa','gnmda','theta'], record = [0])
StateMonInput = StateMonitor(Input, ['isSpiking'],record = [0])


net = Network(SIE,StateMonIE,SpikeMonI,SEI,StateMonI,I,SpikeMonInput,SpikeMonG,G,StateMonE,SpikeMonE,Input,E,SInputE,SGI,StateMonInput,StateMonInputE)
net.schedule = ['start', 'groups', 'thresholds', 'synapses', 'resets', 'end']
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
                  '2.4.2' : [92,110,151,167]}
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


np.savez_compressed(data_dir+'%s-E.npz'%sim_id,
                    gampa=StateMonE.gampa,
                    gnmda=StateMonE.gnmda,
                    theta=StateMonE.theta,
                    ga = StateMonE.ga,
                    U = StateMonE.U,
                    gI = StateMonE.gI)


np.savez_compressed(data_dir+'%s-I.npz'%sim_id,
                    gampa=StateMonI.gampa,
                    gnmda=StateMonI.gnmda,
                    theta=StateMonI.theta,
                    U = StateMonI.U)

np.savez_compressed(data_dir+'%s-Input.npz'%sim_id,
                    isSpiking = StateMonInput.isSpiking
                    )

np.savez_compressed(data_dir+'%s-synapses_InputE.npz'%sim_id,
                    w=StateMonInputE.w,
                    wt=StateMonInputE.wt,
                    x=StateMonInputE.x,
                    u=StateMonInputE.u,
                    )

np.savez_compressed(data_dir+'%s-synapses_IE.npz'%sim_id,
                    w=StateMonIE.w,
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

np.savez_compressed(data_dir+'%s-spikes_I.npz'%sim_id,
                    t=SpikeMonI.t,
                    i=SpikeMonI.i
                    )