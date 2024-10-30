import numpy as np
import matplotlib.pyplot as plt
import os
import time

from scipy.stats import truncnorm
from scipy import signal

from tqdm.auto import tqdm
from IPython.display import clear_output

from numba import jit, prange, i8, set_num_threads
from numba.typed import List

nopython = True
parallel = True

x_init_random = 0.3
y_init_random = 0.3

# ----------------- USEFUL FUNCTIONS -----------------
def normalize(W):
    ''' Normalize each entry in a matrix by the sum of its row'''
    return W / np.sum(W, axis=1)[:,None]

def get_Trange(r1, r2, c_min=0, c_max=1.4):
    print(f'r1={r1}, r2={r2}')

    Tminus = r1 * r2 / (r1 + r2 + r1*r2)
    Tplus = r2 / (2*r2 +1)

    xplus = Tplus
    yplus = Tplus / r2
    xminus = Tminus
    yminus = Tminus / r2
    print(f'Tminus={Tminus}, Tplus={Tplus}')

    Tmin = c_min * Tminus
    Tmax = c_max * Tplus
    print(f'Tmin={Tmin}, Tmax={Tmax}')
    
    return Tmin, Tmax, Tminus, Tplus

# ---------- CLUSTER ANALYSIS ----------
@jit(nopython=nopython)
def DFSUtil(mask, temp, node, visited):
    '''
    Depth-first search
    '''
    N = mask.shape[0]
    
    # Mark the current vertex as visited
    visited[node] = True
 
    # Store the vertex to list
    temp.append(node)
 
    # Get nearest neighbours
    nn = List.empty_list(i8)
    for n in range(N):
        if (not node == n) and mask[node, n]>0:
            nn.append(n)
                
    # Repeat for all nn
    for i in nn:
        if visited[i] == False:
            # Update the list
            temp = DFSUtil(mask, temp, i, visited)
    return temp

@jit(nopython=nopython)
def myConnectedComponents(W, s):
    '''
    Method to retrieve connected components
    in an undirected graph
    '''
    N = W.shape[0]
    
    # mask adjacency matrix with active nodes
    mask = (W * s).T * s

    visited = np.zeros(N, dtype=np.bool_)
    cc = np.zeros(N)
    
    # Loop over nodes
    for v in range(N):
        if visited[v] == False:
            # if not active, skip
            if not s[v]>0:
                continue
            
            # if active and not visited, compute cluster
            temp = List.empty_list(i8)
            
            clust = DFSUtil(mask, temp, v, visited)
            cc[v] = len(clust)
    
    return -np.sort(-cc)

@jit(nopython=nopython)
def init_state(N, x0, y0):
    '''
    Initialize the state of the system
    '''
    # generate two random numbers
    n_act, n_ref = np.ceil( np.random.random(2) * np.array([x0,y0]) * N ).astype(np.int64)
    # create array
    ss = np.zeros(N)
    ss[:n_act] = 1.
    ss[-n_ref:] = -1.
    # shuffle array
    ss = np.random.choice(ss, len(ss), replace=False)
        
    return ss

@jit(nopython=nopython)
def several_init_state(runs, N, x0, y0):
    several_init = np.zeros((runs, N)).astype(np.int64)
    
    for run in prange(runs):
        several_init[run] = init_state(N, x0, y0)
        
    return several_init

'''
@jit(nopython=True)
def update_state(S, W, N, T, r1, r2, dt):
    
    probs = np.random.random(N)                 # generate probabilities
    s = (S==1).astype(np.float64)               # get active nodes
    pA = ( r1 + (1.-r1) * ( (W@s)>T ) ) * dt    # prob. to become active

    # update state vector
    newS = ( (S==0)*(probs<pA)                  # I->A with prob pA
         + (s)*-1*(probs<dt)                    # A->R with prob dt
         + (s)*(probs>=dt)                      # remain A with prob 1-dt
         + (S==-1)*(probs>=r2*dt)*-1 )          # R->I (remain R with prob 1-r2*dt)

    return newS
'''

@jit(nopython=nopython)
def update_state(S, W, N, T, r1, r2):
    probs = np.random.random(N)                 # generate probabilities
    s = (S==1).astype(np.float64)               # get active nodes
    pA = ( r1 + (1.-r1) * ( (W@s)>T ) )         # prob. to become active

    # update state vector
    newS = ( (S==0)*(probs<pA)                  # I->A
         + (s)*-1                               # A->R
         + (S==-1)*(probs>r2)*-1 )              # R->I (remain R with prob 1-r2)
    
    return newS

@jit(nopython=nopython)
def simulate_trajectory(W, T, r1, r2, steps, eq_steps, S_init):
    N = len(W)
    
    # Create empty vector to store time-series
    states = np.zeros((steps, N))
    x, y = np.zeros(steps), np.zeros(steps)
    
    # Initialize
    #S = init_state(N,0.3,0.3)
    S = S_init.copy()
    
    # Loop over equalization steps
    if eq_steps > 0:
        for _ in range(eq_steps):
            S = update_state(S, W, N, T, r1, r2)
    
    # Loop over time steps
    for i in range(steps):
        S = update_state(S, W, N, T, r1, r2)
        x[i] = np.sum(S==1)#/N
        y[i] = np.sum(S==-1)#/N
        states[i] = S
        
    return x, y, states

@jit(nopython=nopython)
def compute_clusters(states, W, step_clust):
    N, steps = states.shape[0], states.shape[1]
    
    steps = steps // step_clust
    S1, S2 = np.zeros(steps), np.zeros(steps)
    
    for i in range(steps):
        tmp = myConnectedComponents(W, states[i*step_clust])
        S1[i], S2[i] = tmp[0], tmp[1]
        
    return np.mean(S1), np.mean(S2)

@jit(nopython=nopython, parallel=parallel)
def simulate(W, T, r1, r2, steps, eq_steps, runs, step_clust, S_init):
    # Create empty array to store activity over time
    A, sigmaA = np.zeros(runs), np.zeros(runs)
    S1, S2 = np.zeros(runs), np.zeros(runs)
    final_state = np.zeros((runs, W.shape[0])).astype(np.int64)
    
    # Loop over runs
    for run in prange(runs):
        # Simulate trajectory
        traj, refr, states = simulate_trajectory(W, T, r1, r2, steps, eq_steps, S_init[run])
        A[run] = np.mean(traj)
        sigmaA[run] = np.std(traj)
        final_state[run] = states[-1]
        
        # Compute clusters
        if step_clust>0:
            S1[run], S2[run] = compute_clusters(states, W, step_clust)
    
    return np.mean(A), np.mean(sigmaA), np.mean(S1), np.mean(S2), final_state

def run_htc(W, r1, r2, Tmin, Tmax, nT, steps, eq_steps, runs, step_clust=-1,
            norm=True, Tdiv_log=False, display=False, hysteresis=False, num_threads=None):
    start = time.time()
    
    if num_threads is None:
        num_threads = runs
    set_num_threads(int(num_threads))
    
    ### Initialize variables
    if norm:
        W = normalize(W)
    N = W.shape[0]
            
    ### Define critical params
    Tminus = r1 * r2 / (r1 + r2 + r1*r2)
    Tplus = r2 / (2*r2 +1)

    xplus = Tplus
    yplus = Tplus / r2

    xminus = Tminus
    yminus = Tminus / r2
    
    ### Define Trange
    if Tdiv_log:
        Trange = np.logspace(np.log10(Tmin), np.log10(Tmax), nT, endpoint=True)
    else:
        Trange = np.linspace(Tmin, Tmax, nT, endpoint=True)
    
    if hysteresis:
        Trange = np.concatenate([Trange[:-1], Trange[::-1]])
    
    ### Initialize empty arrays
    A, sigmaA = np.zeros(len(Trange)), np.zeros(len(Trange))
    S1, S2 = np.zeros(len(Trange)), np.zeros(len(Trange))
    
    # LOOP OVER Ts
    for i,T in enumerate(Trange):
        if display:
            print(str(i+1) + '/'+ str(len(Trange)) + ' - T = ' +  str(round(T/Tplus, 2)) + ' * T+' )
        
        # if not hysteresis or first T -> generate random initial state
        # otherwise, use final step of previous iteration
        if (hysteresis==False) or (i==0):
            final_state = several_init_state(runs, N, x_init_random, y_init_random)

        A[i], sigmaA[i], S1[i], S2[i], final_state = simulate(W, T, r1, r2, steps, eq_steps, runs, step_clust, final_state.astype(np.float64))
    # END LOOP OVER Ts
    
    if display:
        print('End simulating activity. Total computation time: {:.2f}s'.format(time.time()-start))
    
    if step_clust>0:
        return (Trange, A, sigmaA, S1, S2)
    else:
        return (Trange, A, sigmaA)
    
def plot_results(res, res2=None, cs=['cornflowerblue', 'orange'], labels=['control', 'stroke'], lw=3):
    plt.figure(figsize=(10,3))

    ax1 = plt.subplot(1,2,1)
    ax2 = ax1.twinx()

    ax1.plot(res[0], res[1], c=cs[0], label=labels[0], lw=lw)
    ax2.plot(res[0], res[2], c=cs[0], ls='-.', lw=lw)
    
    if res2 is not None:
        ax1.plot(res2[0], res2[1], c=cs[1], label=labels[1], lw=lw)
        ax2.plot(res2[0], res2[2], c=cs[1], ls='-.', lw=lw)

    ax1.set_xlabel('T')
    ax1.set_ylabel('A')
    ax2.set_ylabel(r'$\sigma$(A)')

    ax1 = plt.subplot(1,2,2)
    ax2 = ax1.twinx()

    ax1.plot(res[0], res[3], c=cs[0], label=labels[0], lw=lw)
    ax2.plot(res[0], res[4], c=cs[0], ls='-.', lw=lw)
    
    if res2 is not None:
        ax1.plot(res2[0], res2[3], c=cs[1], label=labels[1], lw=lw)
        ax2.plot(res2[0], res2[4], c=cs[1], ls='-.', lw=lw)

    ax1.set_xlabel('T') 
    ax1.set_ylabel('S1')
    ax2.set_ylabel('S2')
    
    ax1.legend()
    plt.tight_layout()
    plt.show()