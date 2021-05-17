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
p = 0.05 #MODIFIE
pstim=0.05 #MODIFIE

# initial weights
# ---------------

wEE0 = 0.1
wEI0 = 0.2 *3
wIE0 = 0.2
wII0 = 0.2

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
NE = 4096
NI = 1024

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

# neuron groups
# -------------

# create external input neurons
Input = NeuronGroup(NE,
                    '''
                    rates = Fstim : Hz''',
                    threshold='rand()<rates*dt',
                    method='euler')

# excitatory neuron group
E = NeuronGroup(NE,
                eqsE,
                threshold='''U>Uthr+theta''',
                reset='''theta=thetaspi
                        U=Urest
                        ga+=delta
                        h += 1
                        ''',
                method='rk4')



# variable initialization
# E.U = 'rand()*(Uthr-Urest) + Urest'

E.U = Urest
E.theta = 0
E.h = tauH*gamma
E.H = tauH*gamma*NE
E.gampa = 0
E.gnmda = 0
E.gI = 0
E.ga = 0


# inhibitory neuron group
I = NeuronGroup(NI,
                eqsI,
                threshold='U>Uthr+theta',
                reset='''theta=thetaspi
                        U=Urest
                        E = 0 
                        ''',
                method='rk4')

# variable initialization

I.U =  Urest
I.theta = 0
I.gampa = 0
I.gnmda = 0
I.gI = 0

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
                delay=0.8*ms,
                method='euler')

SInputE.connect(p=pstim)
SInputE.x = xstarE
SInputE.u = ustarE
#SInputE.w = 'clip(wStimE+randn()*0.2,0,1)'
SInputE.w = wStimE

SEE = Synapses(E, E,
            model=exc_syn_eqs_model,
            on_pre=exc_syn_eqs_on_pre,
            on_post=exc_syn_eqs_on_post,
            delay=0.8*ms,
            method='euler')


SEE.connect(condition='i!=j', p=p)

SEE.x = xstarE
SEE.u = ustarE
#SInputE.w = 'clip(wStimE+randn()*0.2,0,1)'
SEE.w = wEE0

# Exc. synapses on inhibitory neurons exhibit STP but no other forms of
# plasticity.

# Encore douteux
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

SEI.connect(p=p)

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

SIE.connect(p=p)
SIE.w = wIE0


# Inh. synapses on inh. neurons do not show any form of plasticity.

SII = Synapses(I, I, on_pre='gI_post += wII0',
            delay=0.8*ms)
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

SpikeMonE = SpikeMonitor(E)

StateMonEE = StateMonitor(SEE, ['w'],record=range(5000),dt=1*second)
SpikeMonI = SpikeMonitor(I)

# SpikeMonInput = SpikeMonitor(Input)

#StateMonE = StateMonitor(E, ['x','u'], record=range(10))
RateMonE = PopulationRateMonitor(E)
RateMonI= PopulationRateMonitor(I)


net = Network(StateMonEE,RateMonE,RateMonI,SpikeMonI,SpikeMonE,Input,E,I,SInputE,SII,SIE,SEI,SEE)

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
                  '2.4.2' : [92,110,141,150]}
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


np.savez_compressed(data_dir+'%s-rate.npz'%sim_id,
                    E=RateMonE.rate,
                    I=RateMonI.rate,
                    t=RateMonE.t )


np.savez_compressed(data_dir+'%s-wEE.npz'%sim_id,
                    w=StateMonEE.w)

np.savez_compressed(data_dir+'%s-wEEfin.npz'%sim_id,
                    w=SEE.w)
np.savez_compressed(data_dir+'%s-wStimEfin.npz'%sim_id,
                    w=SInputE.w)
np.savez_compressed(data_dir+'%s-wIEfin.npz'%sim_id,
                    w=SIE.w)
np.savez_compressed(data_dir+'%s-spkI.npz'%sim_id,
                    t=SpikeMonI.t,
                    i=SpikeMonI.i)



