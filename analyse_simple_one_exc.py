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
import plotly.graph_objects as go
#%%
nb_neurons_stim= 200
spikes = []
name = "/users/nsr/saighi/orchestratedSaighi/src/data/one_neuron_spk_many_input"


for n in range(nb_neurons_stim):
    spiketrain = homogeneous_poisson_process(15*pq.Hz,0.1*pq.second,100*pq.second,as_array=True,refractory_period=0.002*pq.second)
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
sim_name = "one_neuron_1_many_input"
# %%
####AURYN
excitatory = np.load(dir_end+sim_name+"-E.npz")
Input = np.load(dir_end+sim_name+"-Input.npz")
spikes_Input = np.load(dir_end+sim_name+"-spikes_Input.npz")
spikes_E = np.load(dir_end+sim_name+"-spikes_E.npz")
spikes_G = np.load(dir_end+sim_name+"-spikes_G.npz")
weight_InputE = np.load(dir_end+sim_name+"-synapses_InputE.npz")
# %%
datadir = os.path.expanduser("~/data/sim_network/sim_one_excitatory_neuron_many_input") # Set this to your data path
prefix = "rf1"

# %%
spkfiles  = "%s/%s.0.ext.spk"%(datadir,prefix)
sfo = AurynBinarySpikeView(spkfiles)
spkfiles  = "%s/%s.0.e.spk"%(datadir,prefix)
sfo_e = AurynBinarySpikeView(spkfiles)
# %%
spikes_ext = np.array(sfo.get_spikes())
spikes_e = np.array(sfo_e.get_spikes())
# %%
####Brian
membrane_ex= pd.read_csv("%s/%s.0.e.mem"%(datadir,prefix),delimiter=' ').values
sse_w= pd.read_csv("%s/%s.0.sse"%(datadir,prefix),delimiter=' ').values
sse_x_u= pd.read_csv("%s/%s.u_x"%(datadir,prefix),delimiter=' ').values
ex_g_nmda= pd.read_csv("%s/%s.0.e.g_nmda"%(datadir,prefix),delimiter=' ').values
ex_g_ampa= pd.read_csv("%s/%s.0.e.g_ampa"%(datadir,prefix),delimiter=' ').values

# %%
# print(spikes_Input['t'])
# print(spikes_Input['i'])
# print(spikes_ext)
# %%
### AURYN et BRIAN
# %%
plt.plot(sse_x_u[:,2])
plt.plot(weight_InputE["u"][0])
#plt.plot(sse_x_u[:,2]-weight_InputE["u"][0][:-1])
# %%
plt.plot(sse_x_u[:,1])
plt.plot(weight_InputE["x"][0])
# %%
plt.plot(sse_w[:,100])
plt.plot(weight_InputE["w"][100])

# %%

fig = go.Figure(data=go.Scatter( y=ex_g_ampa[:,1][:10000]))
fig.add_trace(go.Scatter( y=excitatory["gampa"][0][:10000]))
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
expo = lambda x : np.exp(-x/0.02)
res = lambda x: beta*w*expo(x)**3
beta = 0.05
w = 0.2
init = 0.20002
# %%
values = [init]
sptr = spikes_E['t']
for t in range(1,len(sptr)-1):
    to_add = values[-1]
    for t2 in range(t):
        to_add-= res(sptr[t]-sptr[t2])
    values.append(to_add)
# %%
plt.hist(sse_w[-1,:][1:-1],alpha=0.7,bins=20)
plt.hist(weight_InputE["w"][:,-1],alpha=0.7,bins=20)

# %%
print(spikes_E['t'])
print(np.array(spikes_e)[:,0])
# %%
len(weight_InputE["w"][:,100000])
# %%
len(sse_w[100000,:][1:-1])
# %%
for t in range(int(sse_w.shape[0]/4),sse_w.shape[0],int(sse_w.shape[0]/4)):
    plt.hist(sse_w[t,1:-1],alpha=0.7,bins=15)
    plt.hist(weight_InputE["w"][:,t],alpha=0.7,bins=15)
    plt.show()

# %%
len(weight_InputE["w"][:,-1])
# %%
weight_InputE["w"][:,-1]
# %%
len(sse_w[-1,1:-1])
# %%
