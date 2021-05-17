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
from scipy import signal
from scipy.io import mmread
import seaborn as sns

# %%
time_step =0.0001
# %% nbh
dir_end ="/mnt/data2/paul_data/"
sim_name = "21Apr"
# %%
####Brian
spikesI = np.load(dir_end+sim_name+"-spkI.npz")
rates=np.load(dir_end+sim_name+"-rate.npz")
weights = np.load(dir_end+sim_name+"-wEEfin.npz")['w']
weights_ie = np.load(dir_end+sim_name+"-wIEfin.npz")['w']
weightsext = np.load(dir_end+sim_name+"-wStimEfin.npz")['w']

# %%
num_mpi_ranks = 4 # the number of sims you used in parallel
datadir = os.path.expanduser("~/mnt/data2/paul_data/Auryn_archives/sim_stady_state_brian") # Set this to your data path
prefix = "rf1"

#%%
spkfiles  = ["%s/%s.%i.e.spk"%(datadir,prefix,i) for i in range(num_mpi_ranks)]
sfo = AurynBinarySpikeView(spkfiles)

# %%
####Auryn
rateE  = np.mean([pd.read_csv("%s/%s.%i.e.prate"%(datadir,prefix,2),delimiter=' ').values for i in range(num_mpi_ranks)],axis=0)
time_axis = rateE[:,0]
rateE= rateE[:,1]
rateI  = np.mean([pd.read_csv("%s/%s.%i.i2.prate"%(datadir,prefix,i),delimiter=' ' ) for i in range(num_mpi_ranks)],axis=0)
time_axis_I = rateI[:,0]
rateI= rateI[:,1]
wmatfiles  = ["%s/rf1.%i.ee.wmat"%(datadir,i) for i in range(num_mpi_ranks)]
w = np.sum( [ mmread(wf) for wf in wmatfiles ] )
wmatfilesext  = ["%s/rf1.%i.ext.wmat"%(datadir,i) for i in range(num_mpi_ranks)]
wext = np.sum( [ mmread(wf) for wf in wmatfilesext ] )
wmatfilesie  = ["%s/rf1.%i.ie.wmat"%(datadir,i) for i in range(num_mpi_ranks)]
wie = np.sum( [ mmread(wf) for wf in wmatfilesext ] )
# %%
### AURYN et BRIAN
win = signal.windows.hann(1000)
plt.plot(time_axis,np.convolve(rateE,win,'same')/ sum(win),label = "Auryn",alpha = 0.75)
plt.plot(time_axis,np.convolve(rates["E"][:-1],win,'same')/ sum(win),label = "Brian",alpha = 0.75)
plt.legend()

# %%
win = signal.windows.hann(1000)
plt.plot(time_axis_I,rateI,label = "Auryn",alpha = 0.75)
plt.plot(time_axis,np.convolve(rates["I"][:-1],win,'same')/ sum(win),label = "Brian",alpha = 0.75)
plt.legend()
# %%
plt.hist(w.data, bins=100, log=True,alpha=0.5,label="Auryn")
plt.hist(weights,bins=100,log=True,alpha=0.5,label="Brian");
plt.legend()
plt.title("EE weight distribution")
sns.despine()
# %%
plt.hist(wext.data, bins=100, log=True,alpha=0.5,label="Auryn")
plt.hist(weightsext,bins=100,log=True,alpha=0.5,label="Brian");
plt.legend()
plt.title("Ext->E weight distribution")
sns.despine()
# %%
plt.hist(wext.data, bins=100, log=True,alpha=0.5,label="Auryn")
plt.hist(weightsext,bins=100,log=True,alpha=0.5,label="Brian");
plt.legend()
plt.title("I->E weight distribution")
sns.despine()