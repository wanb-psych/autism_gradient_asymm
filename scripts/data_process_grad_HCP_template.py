import numpy as np
import nibabel as nib
import os
import shutil
import pandas as pd
from brainspace.gradient import GradientMaps
from scipy import stats

path = '../results/fc/'
try:
    os.remove(path+'full/'+'.DS_Store')
    os.remove(path+'LL/'+'.DS_Store')
    os.remove(path+'RR/'+'.DS_Store')
    os.remove(path+'LR/'+'.DS_Store')
    os.remove(path+'RL/'+'.DS_Store')
except:    
    pass

try:
  shutil.rmtree('../results/grad/', ignore_errors=False, onerror=None)
except:
  pass

try:
  os.mkdir('../results/grad/')
  os.mkdir('../results/grad/LL/')
  os.mkdir('../results/grad/RR/')
  os.mkdir('../results/grad/LR/')
  os.mkdir('../results/grad/RL/')
  os.mkdir('../results/grad/intra/')
  os.mkdir('../results/grad/inter/')
  os.mkdir('../results/grad/network/')
  os.mkdir('../results/grad/network/intra/')
  os.mkdir('../results/grad/network/inter/')
except:
  pass

path_list = os.listdir(path+'full/')
path_list.sort()

### mean FC matrix
n = len(path_list)
matrix_fc_LL = [None] * n
total_fc_LL = np.zeros((180,180))

for i in range(n):
  matrix_fc_LL[i] = np.array(pd.read_csv(path+'LL/'+path_list[i], header=None))
  total_fc_LL += matrix_fc_LL[i]

mean_fc_LL = total_fc_LL/n
np.savetxt(path+'LL_groupmean.csv', mean_fc_LL, delimiter = ',')

### group gradients
gm = GradientMaps(approach='dm', kernel='normalized_angle',n_components=10,random_state=0)
LL = np.array(pd.read_csv('../data/LL_groupFC_HCP.csv', header=None))
gm.fit(LL)
group_grad_LL = gm.gradients_

path_add = '../results/grad/'

np.savetxt('../results/grad/group_grad_LL.csv', 
             group_grad_LL, delimiter = ',')
np.savetxt('../results/grad/group_grad_LL_lambdas.csv', 
             gm.lambdas_, delimiter = ',')

# individual gradients
for i in path_list:
  align = GradientMaps(n_components=10, random_state=0, approach='dm', 
                       kernel='normalized_angle', alignment='procrustes')  
  fc_LL = np.array(pd.read_csv('../results/fc/LL/'+i, header=None))
  align.fit(fc_LL,reference=group_grad_LL)
  grad_LL = align.aligned_
  np.savetxt(path_add+'LL/'+i, grad_LL, delimiter = ',')
  align = GradientMaps(n_components=10, random_state=0, approach='dm', 
                       kernel='normalized_angle', alignment='procrustes')
  fc_RR = np.array(pd.read_csv('../results/fc/RR/'+i, header=None))
  align.fit(fc_RR,reference=group_grad_LL)
  grad_RR = align.aligned_
  np.savetxt(path_add+'RR/'+i, grad_RR, delimiter = ',')
  align = GradientMaps(n_components=10, random_state=0, approach='dm', 
                       kernel='normalized_angle', alignment='procrustes')
  fc_LR = np.array(pd.read_csv('../results/fc/LR/'+i, header=None))
  align.fit(fc_LR,reference=group_grad_LL)
  grad_LR = align.aligned_
  np.savetxt(path_add+'LR/'+i, grad_LR, delimiter = ',')
  align = GradientMaps(n_components=10, random_state=0, approach='dm', 
                       kernel='normalized_angle', alignment='procrustes')
  fc_RL = np.array(pd.read_csv('../results/fc/RL/'+i, header=None))
  align.fit(fc_RL,reference=group_grad_LL)
  grad_RL = align.aligned_
  np.savetxt(path_add+'RL/'+i, grad_RL, delimiter = ',')
  print('finish   ' + i)

# correct individual gradient if the correlation is negative
for dir in path_list:
  # LL
  df = np.array(pd.read_csv(path_add+'LL/'+dir,header=None))
  r = [None] * 10
  corrected = [None]*10
  for i in range(10):
    r[i] = stats.pearsonr(group_grad_LL[:,i],df[:,i])    
    if r[i][0] > 0:
      corrected[i]=df[:,i]
    else:
      corrected[i]=-1*df[:,i]
  corrected_ll = np.array(corrected).T
  np.savetxt(path_add+'LL/'+dir, corrected_ll, delimiter = ',')
  
  # RR
  df = np.array(pd.read_csv(path_add+'RR/'+dir,header=None))
  r = [None] * 10
  corrected = [None]*10
  for i in range(10):
    r[i] = stats.pearsonr(group_grad_LL[:,i],df[:,i])    
    if r[i][0] > 0:
      corrected[i]=df[:,i]
    else:
      corrected[i]=-1*df[:,i]
  corrected_rr = np.array(corrected).T
  np.savetxt(path_add+'RR/'+dir, corrected_rr, delimiter = ',')
    
  # LR
  df = np.array(pd.read_csv(path_add+'LR/'+dir,header=None))
  r = [None] * 10   
  corrected = [None]*10
  for i in range(10):
    r[i] = stats.pearsonr(group_grad_LL[:,i],df[:,i])    
    if r[i][0] > 0:
      corrected[i]=df[:,i]
    else:
      corrected[i]=-1*df[:,i]
  corrected_lr = np.array(corrected).T
  np.savetxt(path_add+'LR/'+dir, corrected_lr, delimiter = ',')
    
  # RL
  df = np.array(pd.read_csv(path_add+'RL/'+dir,header=None))
  r = [None] * 10
  corrected = [None]*10
  for i in range(10):
    r[i] = stats.pearsonr(group_grad_LL[:,i],df[:,i])    
    if r[i][0] > 0:
      corrected[i]=df[:,i]
    else:
      corrected[i]=-1*df[:,i]
  corrected_rl = np.array(corrected).T
  np.savetxt(path_add+'RL/'+dir, corrected_rl, delimiter = ',')

  # RR-LL, RL-RL
  AI_llrr = corrected_ll - corrected_rr
  AI_lrrl = corrected_lr - corrected_rl
  np.savetxt(path_add+'intra/'+dir, AI_llrr, delimiter = ',')
  np.savetxt(path_add+'inter/'+dir, AI_lrrl, delimiter = ',')
  print('finish   ' + dir)

# ca network
ca_l = np.array(pd.read_csv('../data/ca_glasser_network.csv',header=None))[:,0][:180]
ca_r = np.array(pd.read_csv('../data/ca_glasser_network.csv',header=None))[:,0][180:]

for n in range(len(path_list)):
  ll = np.array(pd.read_csv(path_add+'LL/'+path_list[n],header=None))
  rr = np.array(pd.read_csv(path_add+'RR/'+path_list[n],header=None))
  lr = np.array(pd.read_csv(path_add+'LR/'+path_list[n],header=None))
  rl = np.array(pd.read_csv(path_add+'RL/'+path_list[n],header=None))
  intra = [None] * 3  
  for i in range(3):
    intra[i] = [np.mean(ll[:,i][ca_l==1])-np.mean(rr[:,i][ca_r==1]),
                np.mean(ll[:,i][ca_l==2])-np.mean(rr[:,i][ca_r==2]),
                np.mean(ll[:,i][ca_l==3])-np.mean(rr[:,i][ca_r==3]),
                np.mean(ll[:,i][ca_l==4])-np.mean(rr[:,i][ca_r==4]),
                np.mean(ll[:,i][ca_l==5])-np.mean(rr[:,i][ca_r==5]),
                np.mean(ll[:,i][ca_l==6])-np.mean(rr[:,i][ca_r==6]),
                np.mean(ll[:,i][ca_l==7])-np.mean(rr[:,i][ca_r==7]),
                np.mean(ll[:,i][ca_l==8])-np.mean(rr[:,i][ca_r==8]),
                np.mean(ll[:,i][ca_l==9])-np.mean(rr[:,i][ca_r==9]),
                np.mean(ll[:,i][ca_l==10])-np.mean(rr[:,i][ca_r==10]),
                np.mean(ll[:,i][ca_l==11])-np.mean(rr[:,i][ca_r==11]),
                np.mean(ll[:,i][ca_l==12])-np.mean(rr[:,i][ca_r==12])]
  np.savetxt(path_add+'network/intra/'+path_list[n], np.array(intra).T, delimiter = ',')

  inter = [None] * 3  
  for i in range(3):
    inter[i] = [np.mean(lr[:,i][ca_l==1])-np.mean(rl[:,i][ca_r==1]),
                np.mean(lr[:,i][ca_l==2])-np.mean(rl[:,i][ca_r==2]),
                np.mean(lr[:,i][ca_l==3])-np.mean(rl[:,i][ca_r==3]),
                np.mean(lr[:,i][ca_l==4])-np.mean(rl[:,i][ca_r==4]),
                np.mean(lr[:,i][ca_l==5])-np.mean(rl[:,i][ca_r==5]),
                np.mean(lr[:,i][ca_l==6])-np.mean(rl[:,i][ca_r==6]),
                np.mean(lr[:,i][ca_l==7])-np.mean(rl[:,i][ca_r==7]),
                np.mean(lr[:,i][ca_l==8])-np.mean(rl[:,i][ca_r==8]),
                np.mean(lr[:,i][ca_l==9])-np.mean(rl[:,i][ca_r==9]),
                np.mean(lr[:,i][ca_l==10])-np.mean(rl[:,i][ca_r==10]),
                np.mean(lr[:,i][ca_l==11])-np.mean(rl[:,i][ca_r==11]),
                np.mean(lr[:,i][ca_l==12])-np.mean(rl[:,i][ca_r==12])]
  
  np.savetxt(path_add+'network/inter/'+path_list[n], np.array(inter).T, delimiter = ',')
  print('finish......'+path_list[n])







