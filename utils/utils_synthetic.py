import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from utils.utils import *
import utils.HTC_utils as HTC
import utils.dyngdim.dyngdim as dyngdim
from utils.dyngdim.plotting import compute_dim_max, plot_results

import random

import networkx as nx

import seaborn as sns

### NETWORK GENERATION
def create_block_model(M, L, kavg, mu, seed=None):
    sizes = [M] * L
    N = M*L
    kout = mu*kavg #/2
    kin = kavg-kout

    p_in = kin / M
    p_out = kout / M / (L-1)
    
    return nx.planted_partition_graph(L, M, p_in=p_in, p_out=p_out, seed=seed)

def params_lognormal(mu, sigma):
    return np.log(mu / np.sqrt(mu**2 + sigma**2)), np.log(1 + sigma**2 / mu**2)

def generate_weights_lognormal(sigma, size):
    mu_real, sigma_real = params_lognormal(1, sigma)
    return np.random.lognormal(mu_real, sigma_real, size=size)
    #return np.random.lognormal(1, sigma, size=size)

def generate_weights_exponential(lmbd, size):
    return  np.random.exponential(scale=1/lmbd, size=size)

def create_network(name, kavg, other, info_network, weighted=True, debug=False, lmbd=12.5, seed=None):
    N = info_network['N']
    if name == 'SBM':
        M = info_network['M']
        L = info_network['L']
        mat = create_block_model(M, L, kavg, other, seed=seed)
    elif name == 'SW':
        mat = nx.watts_strogatz_graph(N, int(kavg), other, seed=seed)
    elif name == 'WW':
        mat = nx.erdos_renyi_graph(N, kavg/N, seed=seed)
        
    ### Check if connected
    if not nx.is_connected(mat):
        if debug:
            print('WARNING: network not connected.')
        return None
    
    mat = nx.to_numpy_array(mat)
    
    if name == 'WW':
        mat *= generate_weights_lognormal(sigma=1/other, size=mat.shape)
        # Symmetrize
        mat = np.triu(mat,1)
        mat += mat.T
        return mat
        
    if weighted:
        mat *= generate_weights_exponential(lmbd, size=mat.shape)
        # Symmetrize
        #mat = np.triu(mat,1)
        #mat += mat.T
        #mat *= generate_weights_lognormal(sigma=1, size=mat.shape)
    
    return mat

### PLOT
def plot_single(res, ax=None, figsize=(4,2)):
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.subplot(1,1,1)
    
    ax.plot(res[0], res[1] / N, '-o', c='k')
    
    ax.axvline(Tminus, ls='--', c='k')
    ax.axvline(Tplus, ls='--', c='k')
    ax.axhline(Tminus, ls='--', c='k')
    ax.axhline(Tplus, ls='--', c='k')
    
    ax.set_xlabel('T')
    ax.set_ylabel('A')
    
    if ax is None:
        plt.show()

def show_example(name, info_network, kavg, other, t_min, t_max, n_t,
                 simulate_HTC=True, compute_dim=True, show_network=True):
    times = np.logspace(t_min, t_max, n_t)
    
    ### Create matrix
    mat = None
    while mat is None:
        mat = create_network(name, kavg, other, info_network)
        
    if show_network:
        plt.figure(figsize=(3,3))
        nx.draw(nx.from_numpy_array(mat), node_size=5)
        plt.show()
    
    ### Simulate HTC model
    if simulate_HTC:
        print('Simulating HTC...')
        tmp_htc = HTC.run_htc(mat, r1, r2, Tmin, Tmax, nT, steps, eq_steps, runs, step_clust=0,
                  norm=True, Tdiv_log=False, display=False, hysteresis=True)
        plot_single(tmp_htc)
    
    ### Compute dimensionality
    if compute_dim:
        print('Computing dimensionality...')
        local_dimensions = dyngdim.run_local_dimension(nx.from_numpy_array(mat/mat.max()), times, n_workers=n_workers, use_spectral_gap=use_spectral_gap)    
        dim, dim_all = plot_results(times, local_dimensions, mat)
        return dim, dim_all