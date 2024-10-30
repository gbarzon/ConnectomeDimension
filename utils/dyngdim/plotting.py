"""plotting functions"""
import os

import matplotlib.colors as pltc
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def plot_all_sources(relative_dimensions):
    """plot relative dimensionf computed from all sources"""
    relative_dimensions = relative_dimensions + np.diag(len(relative_dimensions) * [np.nan])

    plt.figure()
    plt.imshow(relative_dimensions, cmap=plt.get_cmap("coolwarm"))
    plt.colorbar(label="Rwlative dimension")


def plot_single_source(
    results,
    ds=None,
    folder="./",
    target_nodes=None,
    with_trajectories=False,
    figsize=(5, 4),
):
    """plot the relative dimensions"""
    if ds is None:
        ds = [1, 2, 3]
    plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 2, height_ratios=[0.2, 1], width_ratios=[1, 0.05])

    gs.update(wspace=0.05)
    gs.update(hspace=0.00)

    ax1 = plt.subplot(gs[0, 0])
    results["peak_times"][np.isnan(results["peak_times"])] = results["times"][-1]
    plt.hist(
        np.log10(results["peak_times"]),
        bins=max(50, int(0.02 * len(results["peak_times"]))),
        density=False,
        log=True,
        range=(
            np.log10(results["times"][0] * 0.9),
            np.log10(results["times"][-1] * 1.1),
        ),
        color="0.5",
    )
    ax1.set_xlim(np.log10(results["times"][0] * 0.9), np.log10(results["times"][-1] * 1.1))
    ax1.set_xticks([])

    ax2 = plt.subplot(gs[1, 0])
    if with_trajectories:
        for traj in results["node_trajectories"].T:
            ax2.plot(results["times"], traj, lw=0.5, c="0.8", zorder=-1)

    cmap = plt.get_cmap("coolwarm")

    nan_id = np.isnan(results["relative_dimensions"])
    relative_dimension = results["relative_dimensions"].copy()

    dim_min = np.nanpercentile(relative_dimension, 2)
    dim_max = np.nanpercentile(relative_dimension, 98)
    ax2.scatter(
        results["peak_times"][nan_id],
        results["peak_amplitudes"][nan_id],
        c="k",
        s=50,
    )
    sc = ax2.scatter(
        results["peak_times"],
        results["peak_amplitudes"],
        c=relative_dimension,
        s=50,
        cmap=cmap,
    )

    if target_nodes is not None:
        plt.scatter(
            results["peak_times"][target_nodes],
            results["peak_amplitudes"][target_nodes],
            s=50,
            color="r",
        )
    plt.xscale("log")
    plt.yscale("log")

    def f(d):
        f = (
            results["times"] ** (-d / 2.0)
            * np.exp(-d / 2.0)
            / (4.0 * results["diffusion_coefficient"] * np.pi) ** (0.5 * d)
        )
        return f

    for d in ds:
        ax2.plot(results["times"], f(d), "--", lw=2, label=r"$d_{rel} =$" + str(d))
    norm = pltc.Normalize(vmin=0.0, vmax=1)
    ax2.plot(
        results["times"],
        f(dim_min),
        "-",
        lw=1,
        label=r"$d_{rel} =$" + str(np.round(dim_min, 2)),
        c=cmap(norm(0)),
    )
    ax2.plot(
        results["times"],
        f(dim_max),
        "-",
        lw=1,
        label=r"$d_{rel} =$" + str(np.round(dim_max, 2)),
        c=cmap(norm(1)),
    )

    ax2.set_xlim(results["times"][0] * 0.9, results["times"][-1] * 1.1)
    ax2.set_ylim(
        np.nanmin(results["peak_amplitudes"]) * 0.9,
        np.nanmax(results["peak_amplitudes"]) * 1.1,
    )
    ax1.set_xticks([])

    ax_cb = plt.subplot(gs[1, 1])
    cbar = plt.colorbar(sc, cax=ax_cb)
    cbar.set_label(r"${\rm Relative\,\, dimension}\,\, d_{rel}$")

    ax2.set_xlabel(r"$\rm Peak\,\,  time\, \,  (units\, \,  of\, \,  \lambda_2)$")
    ax2.set_ylabel(r"$\rm Peak\,\, amplitude$")
    ax2.legend()

    #plt.savefig(folder + "/relative_dimension.svg", bbox_inches="tight")
    plt.show()

def plot_local_dimensions(
    graph, local_dimension, time_horizon=None, pos=None, ax=None,
    to_save=None, figsize=(5, 4), cmap = "Spectral_r", dpi=None
):
    """plot local dimensions"""

    cmap = plt.get_cmap(cmap)
    
    if pos is None:
        pos = nx.spring_layout(graph)

    vmin = np.nanmin(local_dimension)
    vmax = np.nanmax(local_dimension)

    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.subplot()

    node_size = local_dimension / np.max(local_dimension) * 50
    node_order = np.argsort(node_size)

    for n in node_order:
        nodes = nx.draw_networkx_nodes(
                graph,
                pos=pos,
                nodelist=[n],
                node_size=node_size[n],
                cmap=cmap,
                node_color=[local_dimension[n]],
                vmin=vmin,
                vmax=vmax,
                ax=ax
        )

    #plt.colorbar(nodes, label="Local Dimension")
    
    weights = np.array([graph[i][j]["weight"] for i, j in graph.edges])
    nx.draw_networkx_edges(graph, pos=pos, alpha=0.5, width=weights / np.max(weights), ax=ax)

    if time_horizon is not None:
        plt.title("Time Horizon {:.2e}".format(time_horizon), fontsize=14)
        
    #plt.axis('off')
    ax.axis('off')
    
    if to_save is not None:
        plt.savefig(to_save, bbox_inches='tight', dpi=dpi)
    
    if ax is None:
        plt.show()
                 
                 
'''
    
    
def plot_local_dimensions(
    graph, local_dimension, times, pos=None, folder="./outputs/local_dimension_figs", figsize=(5, 4)
):
    """plot local dimensions"""

    #if not os.path.isdir(folder):
    #    os.mkdir(folder)

    if pos is None:
        pos = nx.spring_layout(graph)

    #plt.figure()

    vmin = np.nanmin(local_dimension)
    vmax = np.nanmax(local_dimension)

    for time_index, time_horizon in enumerate(times):
        plt.figure(figsize=figsize)

        node_size = local_dimension[time_index, :] / np.max(local_dimension[time_index, :]) * 20

        cmap = plt.cm.coolwarm

        node_order = np.argsort(node_size)

        for n in node_order:
            nodes = nx.draw_networkx_nodes(
                graph,
                pos=pos,
                nodelist=[n],
                node_size=node_size[n],
                cmap=cmap,
                node_color=[local_dimension[time_index, n]],
                vmin=vmin,
                vmax=vmax,
            )

        plt.colorbar(nodes, label="Local Dimension")

        weights = np.array([graph[i][j]["weight"] for i, j in graph.edges])
        nx.draw_networkx_edges(graph, pos=pos, alpha=0.5, width=weights / np.max(weights))

        plt.suptitle("Time Horizon {:.2e}".format(time_horizon), fontsize=14)
        plt.show()
        #plt.savefig(folder + "/local_dimension_{}.svg".format(time_index))
        #plt.close()
'''

def compute_dim_max(local_dimensions, cutoff=.1):
    ok = np.isnan(local_dimensions).sum(axis=1)<int(cutoff*local_dimensions.shape[1])
    dim_avg = np.nanmean(local_dimensions, axis=1)
    dim = np.max(dim_avg[ok])
    imax = np.where(dim_avg == dim)[0][0]
    dim_all = local_dimensions[imax]
    
    '''
    imax = np.nanargmax(np.mean(local_dimensions, axis=1))
    #tmax = times[imax]
    dim = np.mean(local_dimensions, axis=1)[imax]
    #dim = np.nanmean(local_dimensions, axis=1)[imax]
    dim_all = local_dimensions[imax]
    '''
    
    return dim, dim_all

def plot_results(times, local_dimensions, mat, figsize=(16,4), cutoff=.1):
    plt.figure(figsize=figsize)

    plt.subplot(1,3,1)

    plt.pcolor(times, range(mat.shape[0]), local_dimensions.T, shading='auto')

    plt.xlabel(r'Scale, $\tau$')
    plt.ylabel('Node')
    plt.xscale('log')

    cbar = plt.colorbar()
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('Local dimension, D', rotation=270)

    plt.subplot(1,3,2)

    plt.plot(times, np.mean(local_dimensions, axis=1), 'o-')
    plt.plot(times, np.nanmean(local_dimensions, axis=1), 'o-', c='violet', zorder=-1)
    
    '''
    if np.isnan(np.mean(local_dimensions, axis=1)).sum() < len(times):
        imax = np.nanargmax(np.mean(local_dimensions, axis=1))
        tmax = times[imax]
        dim = np.mean(local_dimensions, axis=1)[imax]
        dim_all = local_dimensions[imax]
    
    else:
        print('WARNING: some nodes are unreachable at every times...')
    '''

    ok = np.isnan(local_dimensions).sum(axis=1)<int(cutoff*mat.shape[0])
    dim_avg = np.nanmean(local_dimensions, axis=1)
    dim = np.max(dim_avg[ok])
    imax = np.where(dim_avg == dim)[0][0]
    dim_all = local_dimensions[imax]
        
    plt.plot(times[imax], np.nanmean(local_dimensions, axis=1)[imax], 'o', c='red', label=r'D$_{max}$='+str(np.round(dim,2)))

    plt.xlabel(r'Scale, $\tau$')
    plt.ylabel(r'Global dimension, D($\tau$)')
    plt.xscale('log')
    plt.xlim(times[0], times[-1])
    plt.legend()

    plt.subplot(1,3,3)
    plt.plot(times, np.isnan(local_dimensions).sum(axis=1) / mat.shape[0], 'o-')
    plt.xlabel(r'Scale, $\tau$')
    plt.ylabel(r'Unreacheble nodes')
    plt.xscale('log')
    plt.xlim(times[0], times[-1])
    
    plt.tight_layout()
    plt.show()
    
    return dim, dim_all