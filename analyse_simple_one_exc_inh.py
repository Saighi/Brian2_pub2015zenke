# %%
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
sys.path.append(os.path.expanduser("~/auryn/tools/python/"))
from auryntools import *
import pandas as pd
import matplotlib.pyplot as plt
import os, sys
import quantities as pq
from elephant.spike_train_generation import homogeneous_poisson_process
from elephant.statistics import instantaneous_rate
import plotly.graph_objects as go
from neo.core import SpikeTrain
#%%
nb_neurons_stim= 200
spikes = []
name = "/users/nsr/saighi/orchestratedSaighi/src/data/sim_one_ex_inh_neuron_many_input"


for n in range(nb_neurons_stim):
    spiketrain = homogeneous_poisson_process(10*pq.Hz,0.1*pq.second,5*pq.second,as_array=True,refractory_period=0.002*pq.second)
    for i in np.column_stack((spiketrain,np.full(len(spiketrain),n))):
        spikes.append(list(i))

spikes = np.array(spikes)
spikes = spikes[np.argsort(spikes[:,0])]
f = open(name,'w')
for s in spikes:
    f.write(str(s[0])+" "+str(int(s[1]))+'\n')
f.close()

# %%
# instant = 0.1
# shift = 0.0005
# nb_neurons_stim= 200
# boucle = 4
# name = "/users/nsr/saighi/orchestratedSaighi/src/data/one_neuron_spk_many_input"
# spikes_stim = np.array([ [(instant+(i*shift)),i%nb_neurons_stim] for i in range(nb_neurons_stim*boucle)])
# f = open(name,'w')
# for s in spikes_stim:
#     f.write(str(s[0])+" "+str(int(s[1]))+'\n')
# f.close()

# %%
dir_end ="/mnt/data2/paul_data/"
sim_name = "one_neuron_exc_inh"
# %%
####Brian
excitatory = np.load(dir_end+sim_name+"-E.npz")
inhibitory = np.load(dir_end+sim_name+"-I.npz")
Input = np.load(dir_end+sim_name+"-Input.npz")
spikes_Input = np.load(dir_end+sim_name+"-spikes_Input.npz")
spikes_I = np.load(dir_end+sim_name+"-spikes_I.npz")
spikes_E = np.load(dir_end+sim_name+"-spikes_E.npz")
spikes_G = np.load(dir_end+sim_name+"-spikes_G.npz")
weight_InputE = np.load(dir_end+sim_name+"-synapses_InputE.npz")
weight_IE = np.load(dir_end+sim_name+"-synapses_IE.npz")
# %%
datadir = os.path.expanduser("~/data/sim_network/sim_one_ex_inh_neuron_many_input") # Set this to your data path
prefix = "rf1"

# %%
spkfiles  = "%s/%s.0.ext.spk"%(datadir,prefix)
sfo = AurynBinarySpikeView(spkfiles)
spkfiles  = "%s/%s.0.e.spk"%(datadir,prefix)
sfo_e = AurynBinarySpikeView(spkfiles)
spkfiles  = "%s/%s.0.i.spk"%(datadir,prefix)
sfo_i = AurynBinarySpikeView(spkfiles)
# %%
spikes_ext = np.array(sfo.get_spikes())
spikes_e = np.array(sfo_e.get_spikes())
spikes_i = np.array(sfo_i.get_spikes())
# %%
####Auryn
membrane_ex= pd.read_csv("%s/%s.0.e.mem"%(datadir,prefix),delimiter=' ').values
membrane_i= pd.read_csv("%s/%s.0.i.mem"%(datadir,prefix),delimiter=' ').values
sse_w= pd.read_csv("%s/%s.0.sse"%(datadir,prefix),delimiter=' ').values
sie_w= pd.read_csv("%s/%s.0.sie"%(datadir,prefix),delimiter=' ').values
sse_x_u= pd.read_csv("%s/%s.u_x"%(datadir,prefix),delimiter=' ').values
ex_g_nmda= pd.read_csv("%s/%s.0.e.g_nmda"%(datadir,prefix),delimiter=' ').values
ex_g_ampa= pd.read_csv("%s/%s.0.e.g_ampa"%(datadir,prefix),delimiter=' ').values
ex_g_gaba= pd.read_csv("%s/%s.0.e.g_gaba"%(datadir,prefix),delimiter=' ').values
inh_g_ampa= pd.read_csv("%s/%s.0.i.g_ampa"%(datadir,prefix),delimiter=' ').values


# %%
# print(spikes_Input['t'])
# print(spikes_Input['i'])
# print(spikes_ext)

# %%
### AURYN et BRIAN

plt.plot(sse_x_u[:,2])
plt.plot(weight_InputE["u"][0])
#plt.plot(sse_x_u[:,2]-weight_InputE["u"][0][:-1])
# %%
plt.plot(sse_x_u[:,1])
plt.plot(weight_InputE["x"][0])

# %%
plt.plot(sse_w[:,1])
plt.plot(weight_InputE["w"][0])
#%%
fig = go.Figure(data=go.Scatter( y=ex_g_ampa[:,1]))
fig.add_trace(go.Scatter( y=excitatory["gampa"][0]))
fig.show()
#%%
plt.plot(sie_w[:,1])
plt.plot(weight_IE["w"][0])
# %%

fig = go.Figure(data=go.Scatter( y=ex_g_gaba[:,1]))
fig.add_trace(go.Scatter( y=excitatory["gI"][0]))
fig.show()
# plt.plot(ex_g_ampa[:,1])
# plt.plot(excitatory["gampa"][0])

# %%
fig = go.Figure(data=go.Scatter( y=excitatory["U"][0]))
fig.add_trace(go.Scatter( y=membrane_ex[:,1]))
fig.show()
# plt.plot(excitatory["U"][0])
# plt.plot(membrane_ex[:,1])
# %%
# plt.plot(excitatory["gnmda"][0])
# plt.plot(ex_g_nmda[:,1])

fig = go.Figure(data=go.Scatter( y=ex_g_nmda[:,1]))
fig.add_trace(go.Scatter( y=excitatory["gnmda"][0]))
fig.show()

# %%
#values
# %%
#set(weight_InputE["w"][0])

# %%
#plt.plot(weight_InputE["w"][0][:-1]-sse_w[:,1])
# %%
plt.plot(np.array(spikes_E['t'])-np.array(spikes_e)[:,0])
## L'écart entre les spikes est proportionnel à l'écart de la plasticité
# %%
print(len(spikes_E['t']))
print(len(np.array(spikes_e)[:,0]))

# %%
fig = go.Figure(data=go.Scatter( y=inh_g_ampa[:,1]))
fig.add_trace(go.Scatter( y=inhibitory["gampa"][0]))
fig.show()

# %%
fig = go.Figure(data=go.Scatter( y=membrane_i[:,1]))
fig.add_trace(go.Scatter( y=inhibitory["U"][0]))
fig.show()
# %%
brianSpk = SpikeTrain(spikes_E['t']*pq.s, t_stop = max(spikes_E['t']))
# %%
brianRate = instantaneous_rate(brianSpk,1*pq.s)
# %%
plt.plot(brianRate)
# %%
