import numpy as np
import pandas as pd
import os

import networkx as nx

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap
import seaborn as sns


### DEFINE FOLDERS
data_folder = '/data/barzon/arm2/'

folder_control = data_folder + 'control/'
folder_stroke = data_folder + 'stroke/'
folder_templates = data_folder + 'templates/'
folder_mask = data_folder + 'mask/'

### DEFINE GLOBAL VARIABLES
parcellations = np.array([100, 200, 500, 1000])
dict_control = {'name': 'control', 'sessions': [1, 2], 'parcs': parcellations}
dict_stroke = {'name': 'stroke', 'sessions': [1, 2, 3], 'parcs': parcellations}
dict_all = [dict_control, dict_stroke]

thr_types = [None, 'local', 'mask']

###  DEFINE UTILS FOR PLOTTING
def load_matplotlib_local_fonts():
    from matplotlib import font_manager
    
    font_path = '/home/barzon/Avenir.ttc'
    
    font_manager.fontManager.addfont(font_path)
    prop = font_manager.FontProperties(fname=font_path)

    #  Set it as default matplotlib font
    plt.rcParams.update({
        'font.sans-serif': prop.get_name(),
    })
    
load_matplotlib_local_fonts()
plt.rcParams.update({'font.size': 16})
#my_cmap = ListedColormap(sns.color_palette('mako_r', as_cmap=True).as_hex())
my_cmap = sns.color_palette('mako_r', as_cmap=True)
plt.rcParams['font.size'] = 18
plt.rcParams['figure.figsize'] = [6, 3.5]
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.linewidth'] = 1.2

########################################

class data_loader:
    '''
    WARNING:
    - medial wall are present only in the connectomes, not in the Schaefer templates
    - in data_loader, return None if the matrix is not connected
    
    TODO:
    - add column 'ctx', 'subctx' to roi dataset
    - add column with RSN to roi dataset
    
    INFO:
    - thr can be: [None, 'local', 'mask']
    '''
    
    def __init__(self, which, ses, parc, thr, include_subctx,
                 include_all=False,
                 percentile_intra=0.8, percentile_inter=0.85, percentile_subctx=0.9):
        
        if not thr in thr_types:
            raise Exception(f'Wrong threshold type. Should be: {thr_types}')
        
        ### Store variables
        self.which = which
        self.ses = ses
        self.parc = parc
        self.thr = thr
        
        self.include_subctx = include_subctx
        self.include_all = include_all
        
        self.percentile_intra = percentile_intra
        self.percentile_inter = percentile_inter
        self.percentile_subctx = percentile_subctx
        
        ### Load additional infos
        self.define_indexes()
        self.get_names()
        
        ### Print infos
        self.print_infos()
        
        ### Load mask
        if thr == 'mask':
            self.load_mask()
        
    def print_infos(self):
        print(f'######## {self.which} - ses 0{self.ses} - parc {self.parc} - total {self.idx_max} - thr {self.thr} - subctx {self.include_subctx} ########')
        
    def define_indexes(self):
        '''
        Define indexes of nodes to be removed
        '''
        ### Create temporary indexes to remove
        self.idx_cerebellum = np.arange(14, 48)
        self.idx_medial_wall = np.array([1, self.parc//2+1+1]) + self.idx_cerebellum[-1]
        self.idx_subctx = np.arange(14)
        self.old_idx_subctx = np.arange(14)
        
        if not self.include_all:
            self.to_remove = np.concatenate([self.idx_cerebellum, self.idx_medial_wall])
            #self.idx_cerebellum, self.idx_medial_wall = [], []
            
            if not self.include_subctx:
                self.to_remove = np.concatenate([self.to_remove, self.idx_subctx])
                self.idx_subctx = []
                
        else:
            self.to_remove = []
            
        self.idx_all = np.concatenate([['subctx']*len(self.idx_subctx), ['ctx'] * self.parc])
                
    def get_rois(self):
        ### Load templates
        template = f'Schaefer2018_{int(self.parc)}Parcels_7Networks_order_FSLMNI152_1mm.Centroid_RAS.csv'
        cortical = pd.read_csv(folder_templates + template).drop(columns='ROI Label')
        
        # Add column RSN
        cortical['RSN'] = [name.split('_')[2] for name in cortical['ROI Name'].values]
        
        if not self.include_subctx:
            return cortical
        
        ### Load subctx
        subctx = pd.read_csv(folder_templates + 'lut_subcortical-cerebellum_mics.csv').drop(columns=['R', 'G', 'B', 'A', 'roi', 'mics', 'side'])
        
        # Remove cerebellum
        subctx = subctx.drop(self.idx_cerebellum)
        # Add column RSN
        subctx['RSN'] = [None] * len(self.idx_subctx)
        # Rename columns
        subctx = subctx.set_axis(cortical.columns, axis=1)
        
        # Merge
        rois = pd.concat([subctx, cortical], ignore_index=True)
        # Add column type
        rois['type'] = self.idx_all
            
        return rois
    
    def get_rois_mni(self):
        '''
        WARNING: only ctx, subctx not included
        '''
        ### Load templates
        template = f'Schaefer2018_{int(self.parc)}Parcels_7Networks_order_FSLMNI152_1mm.Centroid_MNI.xls'
        cortical = pd.read_excel(folder_templates + template)#.drop(columns='ROI Label')
        
        # Add column RSN
        cortical['RSN'] = [name.split('_')[2] for name in cortical['Name'].values]
            
        return cortical
    
    def get_names(self):
        ### Define names
        t_ses = '-0' + str(self.ses) + '_space'
        t_parc = '-' + str(self.parc) + '_desc'
        self.full_names = [file for file in os.listdir(data_folder+self.which) if (t_ses in file and t_parc in file)]
        self.names = [file.split('_')[0] for file in self.full_names]
        
        # Store idxs for iterator
        self.idx_max = len(self.names)
        if self.idx_max == 0:
            raise Exception('No names loaded. Check if session and parcellations are correct.')
    
    def load_mask(self):
        try:
            fname = f'mask_ses_{self.ses}_N_{self.parc}_intra_{self.percentile_intra}_inter_{self.percentile_inter}_subctx_{self.percentile_subctx}.txt'
            self.mask = np.loadtxt('/data/barzon/arm2/mask/'+fname)
                
            # Remove subctx
            if not self.include_subctx:
                self.mask = np.delete(np.delete(self.mask, self.old_idx_subctx, axis=0), self.old_idx_subctx, axis=1)
        except:
            raise Exception('Unable to load mask.')
        
    def load_matrix(self, idx, keep_none=False):
        print(f'- Loading {self.names[idx]}')
        # Load data
        data = np.loadtxt(data_folder+self.which+'/'+self.full_names[idx])
        
        # Remove idxs
        if len(self.to_remove)>0:
            data = np.delete(np.delete(data, self.to_remove, axis=0), self.to_remove, axis=1)
            
        # Remove diagonal
        data -= np.diag(np.diag(data))
        # Add lower triangular
        data += data.T
        
        # Threshold
        if self.thr is not None:
            if self.thr == 'local':
                # Percentile thresholding
                data = threshold_local(data, len(self.idx_subctx), self.percentile_intra, self.percentile_inter, self.percentile_subctx)
            elif self.thr == 'mask':
                # Load mask
                data *= self.mask
            
        # Check if connected
        if not nx.is_connected(nx.from_numpy_array(data)):
            print('WARNING: matrix not connected.')
            if not keep_none:
                return None
            
        return data
    
    def __iter__(self):
        self.idx_iter = 0
        return self

    def __next__(self):
        if self.idx_iter < self.idx_max:
            self.idx_iter += 1
            return self.load_matrix(self.idx_iter-1)
        raise StopIteration
        
########################################

def plot_connectome(data, logscale=True, cmap='viridis', ax=None):
    vmin = data[data>0].max()
    vmax = data[data>0].min()

    if ax is None:
        plt.figure(figsize=(3,3))
        ax = plt.subplot()
        #TODO: change plt to ax
        
    if logscale:
        im = plt.imshow(data, norm=LogNorm(vmin=vmin, vmax=vmax), cmap=cmap, aspect='auto')
    else:
        im = plt.imshow(data, cmap=cmap, aspect='auto')
    plt.colorbar(im,fraction=0.046, pad=0.04)
    plt.show()

########################################

def generate_mask_indexes(num_ctx, num_subctx):
    #num_ctx = num_nodes - num_subctx
    num_nodes = num_ctx + num_subctx
    
    # Create mask for subctx
    if num_subctx==0:
        mask_subctx = None
    else:
        mask_subctx = np.zeros((num_nodes,num_nodes), dtype=bool)
        mask_subctx[:num_subctx, :] = True
        mask_subctx[:, :num_subctx] = True
        np.fill_diagonal(mask_subctx, False)
    
    # Create a mask for intraconnections within each hemisphere
    left_mask = np.zeros((num_nodes,num_nodes), dtype=bool)
    left_mask[num_subctx:num_subctx+num_ctx//2, num_subctx:num_subctx+num_ctx//2] = True
    np.fill_diagonal(left_mask, False)

    right_mask = np.zeros((num_nodes,num_nodes), dtype=bool)
    right_mask[num_subctx+num_ctx//2:, num_subctx+num_ctx//2:] = True
    np.fill_diagonal(right_mask, False)

    mask_intra = np.logical_or(left_mask, right_mask)

    # Create a mask for interconnections between hemispheres
    if num_subctx==0:
        mask_inter = np.logical_not(mask_intra)
    else:
        mask_inter = np.logical_not(np.logical_or(mask_intra, mask_subctx))
    np.fill_diagonal(mask_inter, False)
    
    return mask_intra, mask_inter, mask_subctx

def threshold_local(mat, num_subctx, percentile_intra, percentile_inter, percentile_subctx):
    num_nodes = mat.shape[0]
    num_ctx = num_nodes - num_subctx
    mat_thr = mat.copy()
    
    ### Get mask
    mask_intra, mask_inter, mask_subctx = generate_mask_indexes(num_ctx, num_subctx)

    ### Compute thr values
    thr_intra = np.quantile(mat[mask_intra], percentile_intra)
    thr_inter = np.quantile(mat[mask_inter], percentile_inter)
    if num_subctx>0:
        thr_subctx = np.quantile(mat[mask_subctx], percentile_subctx)

    #print('thr_intra:', thr_intra, ', thr_inter:', thr_inter)

    ### Threshold
    mat_thr[mask_intra & (mat<=thr_intra)] = 0
    mat_thr[mask_inter & (mat<=thr_inter)] = 0
    if num_subctx>0:
        mat_thr[mask_subctx & (mat<=thr_subctx)] = 0
    
    return mat_thr

def sparsify(mat, threshold=2e-5):
    # Sparsify matrix
    #print(mat.nonzero()[0].size)

    mask = mat<threshold
    mat[mask] = 0
    
    is_conn = nx.is_connected(nx.from_numpy_array(mat))

    #print(mat.nonzero()[0].size)
    if not is_conn:
        print('WARNING: not connected.')
    
    return mat, is_conn

########################################

def load_result(which, var_name, thr, include_subctx, only_matched=False):
    folder = 'results/'+which+'/'
    
    # Init empty list
    ress = []
        
    for tmp_dict in dict_all:
        for parc in tmp_dict['parcs']:
            for sess in tmp_dict['sessions']:
                try:
                    # Load names
                    data = data_loader(which=tmp_dict['name'], ses=sess, parc=parc, thr=thr, include_subctx=include_subctx)
                
                    # Load results
                    for (tmp_full_name, tmp_name) in zip(data.full_names, data.names):
                        try:
                            tmp_res = np.loadtxt(folder+tmp_full_name)
                            ress.append([tmp_dict['name'], sess, parc, tmp_name, np.nanmean(tmp_res), tmp_res.T])
                        except:
                            continue
                except:
                    print('No data found.')
    # Create pandas dataset
    return pd.DataFrame(ress, columns=['name', 'session', 'parc', 'sub', var_name+'_avg', var_name+'_all'])

########################################

def compute_dim_max(local_dimensions):
    imax = np.nanargmax(np.mean(local_dimensions, axis=1))
    #tmax = times[imax]
    dim = np.mean(local_dimensions, axis=1)[imax]
    dim_all = local_dimensions[imax]
    
    return dim, dim_all

def plot_results(times, local_dimensions, mat, figsize=(16,4), cutoff=10):
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
    
    if np.isnan(np.mean(local_dimensions, axis=1)).sum() < len(times):
        imax = np.nanargmax(np.mean(local_dimensions, axis=1))
        tmax = times[imax]
        dim = np.mean(local_dimensions, axis=1)[imax]
        dim_all = local_dimensions[imax]
    
    else:
        print('WARNING: some nodes are unreachable at every times...')
        ok = np.isnan(local_dimensions).sum(axis=1)<cutoff
        dim = np.max(np.nanmean(local_dimensions, axis=1)[ok])
        imax = np.where(np.nanmean(local_dimensions, axis=1)[ok]==dim)[0][0]
        
    plt.plot(times[imax], np.nanmean(local_dimensions, axis=1)[imax], 'o', c='red', label=r'D$_{max}$='+str(np.round(dim,2)))

    plt.xlabel(r'Scale, $\tau$')
    plt.ylabel(r'Global dimension, D($\tau$)')
    plt.xscale('log')
    plt.legend()

    plt.subplot(1,3,3)
    plt.plot(times, np.isnan(local_dimensions).sum(axis=1) / mat.shape[0], 'o-')
    plt.xlabel(r'Scale, $\tau$')
    plt.ylabel(r'Fraction of unreacheble nodes')
    plt.xscale('log')
    plt.tight_layout()
    plt.show()
    
    return dim, dim_all