import numpy as np
import pandas as pd
import scipy.stats as ss
import matplotlib.pyplot as plt
import seaborn as sns
from brainspace.plotting import plot_hemispheres
from brainspace.utils.parcellation import map_to_labels
import hcp_utils as hcp
from brainsmash.mapgen.stats import spearmanr
from brainsmash.mapgen.stats import pearsonr
from brainsmash.mapgen.stats import nonparp
from brainspace.null_models import SampledSurrogateMaps
from scipy.sparse.csgraph import dijkstra
from brainspace.datasets import load_conte69
from scipy.interpolate import make_interp_spline
from scipy.io import loadmat
from matplotlib.colors import ListedColormap
import math
import pingouin
lh, rh = load_conte69()

def spin_spearman(x,y):
  # x, y should be one array 
  n_surrogate_datasets = 1000

  # Note: number samples must be greater than number neighbors
  num_samples = 40
  num_neighbors = 20
  distance= dijkstra(np.array(pd.read_csv('../data/LeftParcelGeodesicDistmat.txt',
                                          header=None, delimiter=' ')), directed=False)
  distance_idx_sorted = np.argsort(distance, axis=1)
  ssm = SampledSurrogateMaps(ns=num_samples, knn=num_samples, random_state=0,resample=True)
  ssm.fit(distance, distance_idx_sorted)
  x_surrogates = ssm.randomize(x, n_rep=n_surrogate_datasets)
  surrogate_corrs = spearmanr(y, x_surrogates).flatten()
  r_stat = spearmanr(x, y)[0]
  p = nonparp(r_stat, surrogate_corrs)
  return p


def spin_pearson(x,y):
  # x, y should be one array 
  n_surrogate_datasets = 1000

  # Note: number samples must be greater than number neighbors
  num_samples = 40
  num_neighbors = 20
  distance= dijkstra(np.array(pd.read_csv('../data/LeftParcelGeodesicDistmat.txt',
                                          header=None, delimiter=' ')), directed=False)
  distance_idx_sorted = np.argsort(distance, axis=1)
  ssm = SampledSurrogateMaps(ns=num_samples, knn=num_samples, random_state=0,resample=True)
  ssm.fit(distance, distance_idx_sorted)
  x_surrogates = ssm.randomize(x, n_rep=n_surrogate_datasets)
  surrogate_corrs = pearsonr(y, x_surrogates).flatten()
  r_stat = pearsonr(x, y)[0]
  p = nonparp(r_stat, surrogate_corrs)
  return p

class spin_pearson_pair:
  def __init__(self,x,y1,y2):
    n_surrogate_datasets = 1000

  # Note: number samples must be greater than number neighbors
    num_samples = 40
    num_neighbors = 20
    distance = dijkstra(np.array(pd.read_csv('../data/LeftParcelGeodesicDistmat.txt',
                        header=None, delimiter=' ')), directed=False)
    distance_idx_sorted = np.argsort(distance, axis=1)
    ssm = SampledSurrogateMaps(ns=num_samples, knn=num_samples, random_state=0,resample=True)
    ssm.fit(distance, distance_idx_sorted)
    x_surrogates = ssm.randomize(x, n_rep=n_surrogate_datasets)
    self.r1_permut = pearsonr(y1, x_surrogates).flatten()
    self.r2_permut = pearsonr(y2, x_surrogates).flatten()
    r_stat1 = pearsonr(x, y1)[0]
    r_stat2 = pearsonr(x, y2)[0]
    self.spin_p1 = nonparp(r_stat1, self.r1_permut)
    self.spin_p2 = nonparp(r_stat2, self.r2_permut)

rng = np.random.default_rng()
def my_statistic(x, y):
  return ss.pearsonr(x, y)[0]
def res(x,y):
  a = ss.bootstrap((x, y), my_statistic, vectorized=False, paired=True,
                  random_state=rng)
  return a

def fdr(p_vals):
    from scipy.stats import rankdata
    ranked_p_values = rankdata(p_vals)
    fdr = p_vals * len(p_vals) / ranked_p_values
    fdr[fdr > 1] = 1
    return fdr

def se(data):
    error = np.std(data, ddof=1) / np.sqrt(np.size(data))
    return error

def cohen_d(d1,d2):
	# calculate the size of samples
  n1, n2 = len(d1), len(d2)
	# calculate the variance of the samples
  s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
	# calculate the pooled standard deviation
  s = math.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
	# calculate the means of the samples
  u1, u2 = np.mean(d1), np.mean(d2)
	# calculate the effect size
  return (u1 - u2) / s

phen = pd.read_csv('../abide_demo_sort.csv')
datasite =['NYU','Pitt','USM', 'NYU_II','TCD']

def sub_corr(path, group, grad):
  corr = [None] * 5
  for i in range(5):
    sub = np.array(phen['ID'][phen['group']==group][phen['site']==datasite[i]]).astype(str)
    n = len(sub)
    a = np.zeros((n,n)) 
    for m in range(n):
      for j in range(n):
        x = np.array(pd.read_csv(path + sub[m]+'.csv', header=None))[:,grad]
        y = np.array(pd.read_csv(path + sub[j]+'.csv', header=None))[:,grad] 
        a[m][j] = ss.pearsonr(x,y)[0]
    corr[i] = a
  return corr

def compare(asd, control, filename):
    a = [None] * 5
    tmp1 = [None] * 5
    tmp2 = [None] * 5
    for i in range(5):
      tmp1[i] = np.triu(asd[i], k=1).flatten()
      tmp1[i][tmp1[i]==0]=np.nan
      tmp2[i] = np.triu(control[i], k=1).flatten()
      tmp2[i][tmp2[i]==0]=np.nan
      a[i] = ss.ttest_ind(tmp1[i], tmp2[i], nan_policy='omit')
    b = ss.ttest_ind(np.concatenate((tmp1)),
                     np.concatenate((tmp2)), nan_policy='omit')
    t_p = np.vstack((a,b))
    index = ['NYU_I', 'Pitt', 'USM', 'NYU_II', 'TCD', 'Total']
    cols = ['t','p']
    df = pd.DataFrame(t_p)
    df.columns = cols
    df.index = index
    df.to_csv(filename)
    return print('every_site:', a, '\n', 
                 'total:', b)

def permut(actual, cal, m): # x,y,n_permutations
  test_stat = ss.pearsonr(actual, cal)[0]
  perm = np.array([np.random.permutation(actual) for _ in range(m)])
  naive_corrs = [ss.pearsonr(perm[i], cal)[0] for i in range(m)]
  p = np.sum(np.abs(naive_corrs) > abs(test_stat)) / m
  return p

mmp = np.genfromtxt('../data/glasser.csv')
lh_mmp = mmp[:32492]
lh_mmp_ = lh_mmp + 180
lh_mmp_[lh_mmp_==180] = 0
mmp_ll = np.concatenate((lh_mmp,lh_mmp_))
rh_mmp = mmp[32492:]
rh_mmp[rh_mmp==180]=0
mmp_lr = np.concatenate((lh_mmp,rh_mmp))
mmp_labels = [hcp.mmp['labels'][n] for n in range(181)][1:]
mmp_labels = np.array([mmp_labels[i][2:] for i in range(180)])

def plot_surface_ll(data, 
                    size,
                    cmap,
                    color_range,
                    filename):
  plot_hemispheres(lh, lh, array_name = data, nan_color = (1,1,1,0.01),size = size,
                   cmap = cmap, color_bar = True, color_range=color_range,
                   interactive = False, zoom = 1.5, embed_nb = True, transparent_bg=True,
                   screenshot=True, filename=filename)
  fig = plot_hemispheres(lh, lh, array_name = data, nan_color = (1,1,1,0.01),size = size,
                         cmap = cmap, color_bar = True, color_range=color_range,
                         interactive = False, zoom = 1.5, embed_nb = True)
  return fig

def plot_surface_lr(data, 
                    size,
                    cmap,
                    color_range,
                    filename):
  plot_hemispheres(lh, rh, array_name = data, nan_color = (1,1,1,0.01),size = size,
                   cmap = cmap, color_bar = True, color_range=color_range,
                   interactive = False, zoom = 1.5, embed_nb = True, transparent_bg=True,
                   screenshot=True, filename=filename)
  fig = plot_hemispheres(lh, rh, array_name = data, nan_color = (1,1,1,0.01),size = size,
                         cmap = cmap, color_bar = True, color_range=color_range,
                         interactive = False, zoom = 1.5, embed_nb = True)
  return fig

def plot_corr_bar(data, filename):
  sns.set_context('paper', font_scale = 3)
  fig, ax = plt.subplots(2,3, figsize=(18,12))
  ax = ax.ravel()
  data_site = ['NYU_I','Pitt','USM', 'NYU_II','TCD']
  corr = data.copy()
  for i in range(5):
    corr[i].sort(axis=1)
    corr[i].sort(axis=0)
    ax[i].imshow(np.delete(corr[i],-1,axis=1), interpolation = 'bilinear', cmap = 'Spectral_r', vmin=-0.5, vmax=0.5, alpha=1)
    ax[i].set_xticks([])
    ax[i].set_yticks([])
    #ax[i].set_ylabel('subject', fontsize=20)
    #ax[i].set_xlabel('subject', fontsize=20)
    ax[i].set_title(data_site[i], pad=15)

  y = [corr[0].mean(), corr[1].mean(), corr[2].mean(), corr[3].mean(), corr[4].mean()]
  ste = [se(corr[0]), se(corr[1]), se(corr[2]), se(corr[3]), se(corr[4])]
  ax[5].errorbar(data_site, y, ste, linestyle='None', marker=' ',
                 elinewidth=3, capsize=6, capthick=2, ms=10)
  ax[5].bar(data_site, y)
  ax[5].set_ylabel('$\it{r}$', fontsize=40)
  ax[5].set_yticks([0,0.15])
  ax[5].tick_params(axis='x', labelsize=20)
  ax[5].tick_params(axis='y', labelsize=20)
  fig.tight_layout()
  fig.savefig('../figures/'+filename+'.png', dpi=300, transparent=True)
  total_mean = np.concatenate(([corr[i].flatten() for i in range(5)])).mean()
  total_se = se(np.concatenate(([corr[i].flatten() for i in range(5)])))
  mean_se = np.vstack((np.array([y,ste]).T,[total_mean, total_se]))
  index = ['NYU_I', 'Pitt', 'USM', 'NYU_II', 'TCD', 'Total']
  cols = ['mean','sd']
  df = pd.DataFrame(mean_se)
  df.columns = cols
  df.index = index
  df.to_csv('../results/compare_mean_se_'+filename+'.csv')
  print('separate:', [y, ste], 'total:',[total_mean, total_se])

rgb = np.array(list(hcp.ca_network['rgba'].values())[1:])

def plot_network_age(mean, ste, num, filename):
  sns.set_context("paper", font_scale = 3)
  netname = ['Vis1', 'Vis2', 'SMN', 'CON', 'DAN', 'Lan.', 'FPN', 'Aud.', 'DMN', 'PMN', 'VMN', 'OAN']
  title = ['Control adults', 'Control adolescents','Controls children', 
           'ASD adults', 'ASD adolescents', 'ASD children']
  fig, ax = plt.subplots(2,3, figsize=(18,12))
  ax = ax.ravel()
  for i in range(6):
    rank = mean[i][:,num].argsort()
    ax[i].barh(np.array(netname)[rank], mean[i][:,num][rank], xerr=ste[i][:,num][rank], alpha=1, align='center',
               color=rgb[rank], error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2))
    ax[i].margins(0.02)
    ax[i].set_title(title[i], pad=10)
    ax[i].set_xticks([-0.02,0,0.02])
    ax[i].spines['right'].set_visible(False)
    ax[i].spines['top'].set_visible(False)

  fig.tight_layout()
  fig.savefig(filename,  dpi=300, transparent=True)
 
def plot_mean_save_sd(mean, sd, num, name):
  data = mean[num]
  np.savetxt('../results/mean_'+name+'.csv', data)
  np.savetxt('../results/sd_'+name+'.csv', sd[num])
  group_level = [None] * 3
  mask = mmp_ll != 0
  for i in range(3):
    group_level[i] = map_to_labels(np.concatenate((data[:,i],data[:,i])), mmp_ll, mask=mask)
    group_level[i][group_level[i]==0]=np.nan

  plot_hemispheres(lh, lh, array_name = group_level, nan_color = (1,1,1,0.01),size = (1200, 600),
                   cmap = 'RdBu', color_bar = True, color_range=(-0.03,0.03),
                   interactive = False, zoom = 1.5, embed_nb = True, transparent_bg=True,
                   screenshot=True, filename='../figures/mean_'+name+'.png')
  a = plot_hemispheres(lh, lh, array_name = group_level, nan_color = (1,1,1,0.01),size = (1200, 600),
                       cmap = 'RdBu', color_bar = True, color_range=(-0.03,0.03),
                       interactive = False, zoom = 1.5, embed_nb = True)

  return a

def plot_interact(mean, ste, network, filename, GSR=False):
  # significant in the interaction
  # so plot this network along each gradient
  data = [mean[2][network][0], mean[5][network][0],
          mean[1][network][0], mean[4][network][0],
          mean[0][network][0], mean[3][network][0],
          mean[2][network][1], mean[5][network][1],
          mean[1][network][1], mean[4][network][1],
          mean[0][network][1], mean[3][network][1],
          mean[2][network][2], mean[5][network][2],
          mean[1][network][2], mean[4][network][2],
          mean[0][network][2], mean[3][network][2]]
  yerr = [ste[2][network][0], ste[5][network][0],
          ste[1][network][0], ste[4][network][0],
          ste[0][network][0], ste[3][network][0],
          ste[2][network][1], ste[5][network][1],
          ste[1][network][1], ste[4][network][1],
          ste[0][network][1], ste[3][network][1],
          ste[2][network][2], ste[5][network][2],
          ste[1][network][2], ste[4][network][2],
          ste[0][network][2], ste[3][network][2]]
  point = [0,0.5,2,2.5,4,4.5,8,8.5,10,10.5,12,12.5,16,16.5,18,18.5,20,20.5]
  a = ['Vis1', 'Vis2', 'SMN', 'CON', 'DAN', 'Lan', 'FPN', 'Aud', 'DMN', 'PMN', 'VMN', 'OAN']
  sns.set_context("paper", font_scale = 3)
  fig, ax = plt.subplots(figsize=(12,5))
  ax.bar(point, data, yerr=yerr, color=[0,0,0,0], edgecolor=[1,1,1,0],
         align='center', error_kw=dict(ecolor=[0,0,0,0.8], lw=1.5, capsize=5, capthick=1.5))
  #order = [0,3,1,4,2,5]
  for i in range(3):
    plt.plot(point[1::2][i*3:(i+1)*3], data[1::2][i*3:(i+1)*3],
                color=[0,0,0,0.8], ls='--', lw=2)
    plt.plot(point[::2][i*3:(i+1)*3], data[::2][i*3:(i+1)*3],
                color=[0,0,0,0.8], ls='-', lw=2)
    color_net = np.array([rgb[network]]*3)
    color_net[:,3] = [0,0.5,1]
    ax.scatter(point[1::2][i*3:(i+1)*3], data[1::2][i*3:(i+1)*3], marker='*', edgecolors=[0,0,0,1],
                color=color_net, s=[200,400,600], label='ASD')
    ax.scatter(point[::2][i*3:(i+1)*3], data[::2][i*3:(i+1)*3], marker='o', edgecolors=[0,0,0,1],
                color=color_net, s=[100,200,300], label='Controls')           
  plt.xticks([2.25,10.25,18.25],['G1','G2','G3'], fontsize=36) 
  plt.yticks([-0.02,0,0.02],['-0.02','0','0.02'], fontsize=28)
  #ax.legend(labels=['ASD', 'Controls'], loc=9, frameon=False, ncol=2)
  #plt.text(7.0, 0.02, 'Children', fontsize=22)
  #plt.text(9.6, 0.02, 'Adols', fontsize=22)
  #plt.text(11.4, 0.02, 'Adults', fontsize=22)
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)
  ax.tick_params(bottom=True, left=True)
  ax.grid(False)

  fig.tight_layout()
  if GSR==False:
    plt.savefig('../figures/interacter_'+str(a[network])+'_'+filename+'.png', dpi=300, transparent=True)
  else:
    plt.savefig('../figures/GSR/interacter_'+str(a[network])+'_'+filename+'.png', dpi=300, transparent=True)  

terms=np.array(['action','affective',"attention","auditory","autobiographical_memory","cognitive_control",
                "emotion","episodic_memory","eye_movement",'face','inhibition','language',
                'motor','multisensory','pain','reading','reward','semantics',
                'social_cognition','verbal','visual','visual_perception','visuospatial','working_memory'])

func_vertex = loadmat('../../evolution/Sofie/functional_maps.mat')['funcmap']
func_mmp = np.zeros((24,360))
for i in range(24):
    for j in range(360):
        func_mmp[i][j] = np.nanmean(func_vertex[i][mmp_lr==j+1])

cmap = ListedColormap(np.vstack((list(hcp.ca_network['rgba'].values())))[1:])
ca = np.array(pd.read_csv('../data/ca_glasser_network.csv',header=None))[:,0].astype(float)
color = ca[:180]

def h2_asy(asy, heri_hcp, se_hcp):
  rank = asy[:,7][asy[:,7] != 1].argsort()[::-1]
  sig = np.where(asy[:,7] != 1)[0]
  sig_rank = sig[rank]
  width = 0.2
  x = np.array(range(sig_rank.shape[0]))
  y = np.vstack((heri_hcp)).copy()
  y[y==0] = 0.00001
  se=np.vstack((se_hcp)).copy()
  se[se.astype(str)=='nan'] = 0.00001
  sns.set_context("paper", font_scale = 2.5)
  fig, ax = plt.subplots(figsize=(sig_rank.shape[0],6))
  ax.bar(x - width, y[0][sig[rank]], yerr=se[0][sig[rank]], width=width, label='G1', color='gray', alpha=1,
         error_kw=dict(ecolor='gray', lw=1, capsize=3, capthick=0, lolims=True))
  ax.bar(x,         y[1][sig[rank]], yerr=se[1][sig[rank]], width=width, label='G2', color='gray', alpha=0.7,
         error_kw=dict(ecolor='gray', lw=1, capsize=3, capthick=0, lolims=True))
  ax.bar(x + width, y[2][sig[rank]], yerr=se[2][sig[rank]], width=width, label='G3', color='gray', alpha=0.4,
         error_kw=dict(ecolor='gray', lw=1, capsize=3, capthick=0, lolims=True))
  ax.margins(0.02)
  ax.set_ylim(0,0.4)
  ax.set_yticks([0,0.2,0.4])
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)
  ax.set_ylabel('Heritability')
  ax.set_xlabel('Parcel labels')
  # ax.legend(loc='upper center', markerscale=0.7, scatterpoints=1, fontsize=15)
  ax.set_xticks(range(sig_rank.shape[0]))
  ax.set_xticklabels(mmp_labels[sig[rank]])
  ax.tick_params(axis='both', labelsize=15)
  fig.tight_layout()

def parcel_term(asy, filename):
  color_alpha = asy[:,3][asy[:,7] != 1]
  sig = np.where(asy[:,7] != 1)[0]
  x = np.array(range(24))
  y = func_mmp[:,sig]
  y_num = y.shape[1]
  Y_ = [None] * y_num
  for i in range(y_num):
    X_Y_Spline = make_interp_spline(x, y[:,i])
    X_ = np.linspace(x.min(), x.max(), 500)
    Y_[i] = X_Y_Spline(X_)
  sns.set_context("paper", font_scale = 3)
  fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,6), gridspec_kw={'width_ratios': [4, 1]})
  a=0
  for y_n in range(y_num): 
    ax1.plot(X_, Y_[y_n], lw=3, alpha = color_alpha[y_n]*0.2,
             color=[0,90/256,216/256,1])
    y_tmp = y[:,y_n].copy()
    y_tmp[y_tmp<3.1] = 0
    a += y_tmp
  b = np.where(a != 0)[0]
  for text in range(len(b)):
    text_size = pow((a[b[text]])*30, 0.5)
    if text_size > 20:
      text_size = 20
    ax2.text(-0.1, 0.9-text/len(b), terms[b[text]], fontsize = text_size,
            transform=ax2.transAxes)
  ax1.spines['right'].set_visible(False)
  ax1.spines['top'].set_visible(False)
  ax1.spines['bottom'].set_visible(False)
  ax2.axis('off')
  ax1.get_xaxis().set_visible(False)
  ax1.set_ylabel('Z-stat', fontsize=20)
  ax1.set_yticks([-6,0,6,12,18])
  ax1.tick_params(axis='both', labelsize=15)
  fig.tight_layout()
  fig.savefig(filename, dpi=300, transparent=True)

def test_r_displot(data):
  sns.set_context("paper", font_scale = 2.5)
  fig, ax = plt.subplots(1, figsize=(8,6))
  L1 = ['0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0']
  for i in range(10):
    sns.distplot(np.array(data)[i][:,0], hist = False, label='L1_ratio='+L1[i], ax=ax)
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)
  ax.set_xlabel('Pearson $\it{r}$')
  plt.legend(loc='upper left', markerscale=0.7, scatterpoints=1, fontsize=10)
  fig.tight_layout()

def select_importance(sig_parcel, importance):
    sel = [None] * 3
    imp = [None] * 3
    sel[0] = sig_parcel[sig_parcel<181]
    imp[0] = importance[sig_parcel<181]
    sel[1] = sig_parcel[np.logical_and(sig_parcel>180,sig_parcel<361)]
    imp[1] = importance[np.logical_and(sig_parcel>180,sig_parcel<361)]
    sel[2] = sig_parcel[sig_parcel>360]
    imp[2] = importance[sig_parcel>360]
    sel_imp = np.zeros((3,180))
    sel_imp[0][sel[0]-1] =imp[0]
    sel_imp[1][sel[1]-181] =imp[1]
    sel_imp[2][sel[2]-361] =imp[2]
    sel_imp[sel_imp == 0] = 0.000001
    return sel_imp

def test_train_fit(test_r_p, train_r_p):
  sns.set_context("paper", font_scale = 1.5)
  fig, ax = plt.subplots(2,5, figsize=(25, 10))
  ax = ax.ravel()
  test_r = np.array(test_r_p).T[0].T
  train_r = np.array(train_r_p).T[0].T
  #diff = (test_r - train_r)/(test_r + train_r)
  for i in range(10):
    rank = test_r[i].argsort()
    #ax[i].plot(range(101), diff[i][rank], lw=3, color='grey')
    ax[i].plot(range(101), test_r[i][rank], lw=3)
    ax[i].plot(range(101), train_r[i][rank], lw=3)
    ax[i].set_title('L1 ratio='+str((i+1)/10), pad=10)
    ax[i].axvline(x=50, color='red', ls='--', alpha = 0.8, lw=3)
  fig.tight_layout()

def select_model(train_r_p, test_r_p, model_num=51): 
  train_r = np.array(train_r_p).T[0].T
  test_r = np.array(test_r_p).T[0].T
  select = np.array([np.where(ss.rankdata(test_r[i])==model_num)[0] for i in range(10)])
  test_sel = np.array([test_r[l1][model] for l1, model in enumerate(select)])
  train_sel = np.array([train_r[l1][model] for l1, model in enumerate(select)])
  return np.array([select[:,0], train_sel[:,0], test_sel[:,0]])

network_name = np.array(['Vis1', 'Vis2', 'SMN', 'CON', 'DAN', 'Lan', 'FPN', 'Aud', 'DMN', 'PMN', 'VMN', 'OAN'])   

def ca_radar_3g(data, width, filename):
  import plotly.graph_objects as go
  import plotly.offline as pyo
  sns.set_context("paper", font_scale = 3)
  network_name = ['Vis1', 'Vis2', 'SMN', 'CON', 'DAN', 'Lan', 'FPN', 'Aud', 'DMN', 'PMN', 'VMN', 'OAN']
  network_name = [*network_name, network_name[0]]
  g1 = data[:,0]
  g2 = data[:,1]
  g3 = data[:,2]
  g1 = [*g1, g1[0]]
  g2 = [*g2, g2[0]]
  g3 = [*g3, g3[0]]
  fig = go.Figure(
      data=[go.Scatterpolar(r=g1, theta=network_name, fill=None, name='G1'),
            go.Scatterpolar(r=g2, theta=network_name, fill=None, name='G2'),
            go.Scatterpolar(r=g3, theta=network_name, fill=None, name='G3')],
      layout=go.Layout(polar={'radialaxis': {'visible': False}},
                       showlegend=True)
                  )
  fig.update_layout(font=dict(size=20))
  pyo.plot(fig, filename = filename,image='svg', auto_play=False, 
           image_width = width, image_height = width)

def dist_plot(data1, data2, bin):
  sns.set_context("paper", font_scale = 3)
  bin_n = len(bin[0])
  title = ['Bin_' + str(i+1) for i in range(bin_n)]
  t_p = [[ss.ttest_rel(data1[:,n][bin[n][i]], data2[:,n][bin[n][i]]) for i in range(bin_n)] for n in range(3)]
  p_str = np.zeros((3,9)).astype(str)
  for n in range(3):
    for i in range(9):
      if t_p[n][i][1] < 0.001:
        p_str[n][i] = '<0.001'
      else:
        p_str[n][i] = '=' + '%.3f'%t_p[n][i][1]

  t = [['%.3f'%np.array(t_p)[n][i][0] for i in range(bin_n)] for n in range(3)]

  fig, ax = plt.subplots(3, bin_n, figsize=(6*bin_n,15))
  ax = ax.ravel()
  for n in range(3):
    for i in range(bin_n):
      sns.distplot(data1[:,n][bin[n][i]], kde_kws=dict(linewidth=5),
                   ax=ax[i+bin_n*n], hist=False, color = 'teal',fit_kws={'color':False})
      sns.distplot(data2[:,n][bin[n][i]], kde_kws=dict(linewidth=5),
                   ax=ax[i+bin_n*n], hist=False, color ='maroon', fit_kws={'color':False})
      l1 = ax[i+bin_n*n].lines[0]
      l2 = ax[i+bin_n*n].lines[1]
      x1 = l1.get_xydata()[:,0]
      y1 = l1.get_xydata()[:,1]
      x2 = l2.get_xydata()[:,0]
      y2 = l2.get_xydata()[:,1]
      #ax[i+bin_n*n].fill_between(x1,y1, color='teal', alpha=0.2)
      #ax[i+bin_n*n].fill_between(x2,y2, color='maroon', alpha=0.2)
      ax[i+bin_n*n].set_yticks([0,2],[])
      ax[i+bin_n*n].set_yticklabels([])
      ax[i+bin_n*n].set_xlabel('$\it{t}$='+t[n][i]+', $\it{P}$'+p_str[n][i], 
                           fontsize=24)
      ax[i+bin_n*n].set_xticks([-1, 0, 1],[])
      #ax[i+bin_n*n].text(0.2, 0.6, '$\it{t}$='+t[n][i]+', $\it{P}$'+p_str[n][i], 
      #               fontsize=26, c='black', transform=ax[i+5*n].transAxes)
      ax[i+bin_n*n].set_ylabel(' ', fontsize=24)
      ax[i+bin_n*n].spines['right'].set_visible(False)
      ax[i+bin_n*n].spines['top'].set_visible(False)
      ax[i+bin_n*n].spines['left'].set_visible(False)
  fig.tight_layout()

def plot_t_single(t_g123, filename, thres=4):
  rank = t_g123.argsort()
  sns.set_context("paper", font_scale = 3.5)
  fig, ax = plt.subplots(figsize=(5,3))
  ax.bar(network_name[rank], t_g123[rank], color=rgb[rank])
  ax.set_xticks([])
  ax.set_yticks([0,thres])
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)
  ax.spines['bottom'].set_visible(False)
  fig.tight_layout()
  fig.savefig(filename, dpi=300, transparent=True)

def plot_t_multi(t_g1, t_g2, t_g3, t_g123, filename):
  rank = t_g123.argsort()
  sns.set_context("paper", font_scale = 3)
  label_loc = np.linspace(start=0, stop=5.8, num=12)
  fig = plt.figure(figsize=(8, 8))
  ax = fig.add_subplot(projection='polar')
  plt.yticks(np.arange(-0.1, 0.1, 0.05))
  plt.bar(label_loc+0.1, t_g1[rank], width=0.1, bottom=0, alpha=1,
          color = rgb[rank])
  plt.bar(label_loc, t_g2[rank], width=0.15, bottom=0, alpha=0.5,
          color = rgb[rank])
  plt.bar(label_loc-0.1, t_g3[rank], width=0.15, bottom=0, alpha=0.3,
          color = rgb[rank])   
  a = network_name[rank]
  lines, labels = plt.thetagrids(np.degrees(label_loc), labels = a)
  plt.yticks([0],[' '])
  plt.ylim(-4,4)
  for spine in ax.spines.values():
      spine.set_edgecolor('None')
  plt.savefig(filename, dpi=300, transparent=True)

#correct if the population S.D. is expected to be equal for the two groups.
def cohen_d(x,y):
    nx = len(x)
    ny = len(y)
    stat = pingouin.compute_effsize(x,y)
    ci = pingouin.compute_esci(stat, nx, ny)
    return np.array([stat, ci[0],ci[1]]).round(2)
