import shutil
import numpy as np
import nibabel as nib
import os
import pandas as pd
from brainstat.stats.terms import FixedEffect
from brainstat.stats.SLM import SLM

try:
  shutil.rmtree('../results/GSR/fc/', ignore_errors=False, onerror=None)
except:
  pass
  
try:
  os.mkdir('../results/GSR/fc/')
  os.mkdir('../results/GSR/fc/full/')
  os.mkdir('../results/GSR/fc/LL/')
  os.mkdir('../results/GSR/fc/RR/')
  os.mkdir('../results/GSR/fc/LR/')
  os.mkdir('../results/GSR/fc/RL/')
  os.mkdir('../results/GSR/fc/intra/')
  os.mkdir('../results/GSR/fc/inter/')
except:
  pass

### ABIDE I
df = pd.read_csv('../data/abideI_clean.csv')
dir_num = np.array(df['ID']).astype(str)
file_name = [None]*len(dir_num)
for i in range(len(dir_num)):
  file_name[i] = np.array(df['Site']).astype(str)[i]+'_00' + np.array(df['OrgID']).astype(str)[i]

mmp = nib.load('../data/cifti.HCP_MMP_P210.CorticalAreas_dil_Colors.10k_fs_LR.dlabel.nii').get_fdata()
lh_mmp = mmp[0][:10242]
rh_mmp = mmp[0][10242:]+180
rh_mmp[rh_mmp==180]=0
mmp_lr = np.concatenate((lh_mmp,rh_mmp))

medial_wall_l = nib.load('../data/lh.atlasroi.10k_fs_LR.shape.gii').agg_data()
medial_wall_r = nib.load('../data/rh.atlasroi.10k_fs_LR.shape.gii').agg_data()
medial_wall = np.concatenate((medial_wall_l,medial_wall_r))

mmp_lr_clean = mmp_lr[medial_wall==1] #Glasser atlas 10k fs_LR without medial wall

path = '../../data/data_autism/1_fc/ABIDE-I/'
for i in range(len(dir_num)):
  file = path+dir_num[i]+'/hcp_processed/'+file_name[i]+'_func.dtseries.nii'  
  if os.path.exists(file):
    img = nib.load(file).get_fdata()
    print('executing sub.'+dir_num[i]+'......')
    img_mmp = np.zeros((img.shape[0],360))
    img_cortex=img[:,0:18722] # mmp_lr_clean.shape is 18722, the number of cortex nodes without medial wall
    for m in range(img.shape[0]):
      for n in range(360):
        img_mmp[m][n]=np.mean(img_cortex[m][mmp_lr_clean==n+1])
    global_mean=np.zeros(img_mmp.shape[0])
    for j in range(img_mmp.shape[0]):
      global_mean[j] = img_mmp[j][img_mmp[j]>=0].mean() # mean across the parcels where bold signal>0
    global_mean = np.nan_to_num(global_mean)
    term = FixedEffect(global_mean,'mean') # default intercept is 1
    model = SLM(term, contrast=term.mean)
    model.fit(img_mmp)
    gsr = np.array([global_mean*model.coef[1][i] for i in range(360)]).T
    ts_gsr = img_mmp - gsr 
    corr_matrix = np.corrcoef(ts_gsr.T)
    corr_matrix[corr_matrix>0.99999] = 1
    fc = np.arctanh(corr_matrix)
    fc[fc == np.inf] = 0
    np.savetxt('../results/GSR/fc/full/'+dir_num[i]+'.csv', fc, delimiter=',')
    fc_LL = fc[0:180,0:180]
    fc_RR = fc[180:360,180:360]
    fc_LR = fc[0:180,180:360]
    fc_RL = fc[180:360,0:180]
    fc_LLRR = fc_LL - fc_RR
    fc_LRRL = fc_LR - fc_RL
    np.savetxt('../results/GSR/fc/LL/'+dir_num[i]+'.csv', fc_LL, delimiter=',')
    np.savetxt('../results/GSR/fc/RR/'+dir_num[i]+'.csv', fc_RR, delimiter=',')
    np.savetxt('../results/GSR/fc/LR/'+dir_num[i]+'.csv', fc_LR, delimiter=',')
    np.savetxt('../results/GSR/fc/RL/'+dir_num[i]+'.csv', fc_RL, delimiter=',')
    np.savetxt('../results/GSR/fc/intra/'+dir_num[i]+'.csv', fc_LLRR, delimiter=',')
    np.savetxt('../results/GSR/fc/inter/'+dir_num[i]+'.csv', fc_LRRL, delimiter=',')


### ABIDE II
df = pd.read_csv('../data/abideII_clean.csv')
dir_num = np.array(df['ID']).astype(str)
path = '../../data/data_autism/1_fc/ABIDE-II/'

for i in range(len(dir_num)):
  file_name = '%.3s'%np.array(df['Site'])[i] +'_'+dir_num[i]
  file = path+dir_num[i]+'/hcp_processed/'+file_name+'_func.dtseries.nii'  
  if os.path.exists(file):
    img = nib.load(file).get_fdata()
    print('executing sub.'+dir_num[i]+'......')
    img_mmp = np.zeros((img.shape[0],360))
    img_cortex=img[:,0:18722] #18722 is the number of cortex nodes without medial wall
    for m in range(img.shape[0]):
      for n in range(360):
        img_mmp[m][n]=np.mean(img_cortex[m][mmp_lr_clean==n+1])
    global_mean=np.zeros(img_mmp.shape[0])
    for j in range(img_mmp.shape[0]):
      global_mean[j] = img_mmp[j][img_mmp[j]>0].mean()    
    global_mean = np.nan_to_num(global_mean)
    term = FixedEffect(global_mean,'mean') # default intercept is 1
    model = SLM(term, contrast=term.mean)
    model.fit(img_mmp)
    gsr = np.array([global_mean*model.coef[1][i] for i in range(360)]).T
    ts_gsr = img_mmp - gsr
    corr_matrix = np.corrcoef(ts_gsr.T)
    corr_matrix[corr_matrix>0.99999] = 1
    fc = np.arctanh(corr_matrix)
    fc[fc == np.inf] = 0
    np.savetxt('../results/GSR/fc/full/'+dir_num[i]+'.csv', fc, delimiter=',')
    fc_LL = fc[0:180,0:180]
    fc_RR = fc[180:360,180:360]
    fc_LR = fc[0:180,180:360]
    fc_RL = fc[180:360,0:180]
    fc_LLRR = fc_LL - fc_RR
    fc_LRRL = fc_LR - fc_RL
    np.savetxt('../results/GSR/fc/LL/'+dir_num[i]+'.csv', fc_LL, delimiter=',')
    np.savetxt('../results/GSR/fc/RR/'+dir_num[i]+'.csv', fc_RR, delimiter=',')
    np.savetxt('../results/GSR/fc/LR/'+dir_num[i]+'.csv', fc_LR, delimiter=',')
    np.savetxt('../results/GSR/fc/RL/'+dir_num[i]+'.csv', fc_RL, delimiter=',')
    np.savetxt('../results/GSR/fc/intra/'+dir_num[i]+'.csv', fc_LLRR, delimiter=',')
    np.savetxt('../results/GSR/fc/inter/'+dir_num[i]+'.csv', fc_LRRL, delimiter=',')
    
    
# Gradients
from brainspace.gradient import GradientMaps
from scipy import stats

path = '../results/GSR/fc/'
try:
    os.remove(path+'full/'+'.DS_Store')
    os.remove(path+'LL/'+'.DS_Store')
    os.remove(path+'RR/'+'.DS_Store')
    os.remove(path+'LR/'+'.DS_Store')
    os.remove(path+'RL/'+'.DS_Store')
except:    
    pass

try:
  shutil.rmtree('../results/GSR/grad/', ignore_errors=False, onerror=None)
except:
  pass

try:
  os.mkdir('../results/GSR/grad/')
  os.mkdir('../results/GSR/grad/LL/')
  os.mkdir('../results/GSR/grad/RR/')
  os.mkdir('../results/GSR/grad/LR/')
  os.mkdir('../results/GSR/grad/RL/')
  os.mkdir('../results/GSR/grad/intra/')
  os.mkdir('../results/GSR/grad/inter/')
  os.mkdir('../results/GSR/grad/network/')
  os.mkdir('../results/GSR/grad/network/intra/')
  os.mkdir('../results/GSR/grad/network/inter/')
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

path_add = '../results/GSR/grad/'

np.savetxt(path_add+'group_grad_LL.csv', 
             group_grad_LL, delimiter = ',')
np.savetxt(path_add+'group_grad_LL_lambdas.csv', 
             gm.lambdas_, delimiter = ',')

# individual gradients
for i in path_list:
  align = GradientMaps(n_components=10, random_state=0, approach='dm', 
                       kernel='normalized_angle', alignment='procrustes')  
  fc_LL = np.array(pd.read_csv('../results/GSR/fc/LL/'+i, header=None))
  align.fit(fc_LL,reference=group_grad_LL)
  grad_LL = align.aligned_
  np.savetxt(path_add+'LL/'+i, grad_LL, delimiter = ',')
  align = GradientMaps(n_components=10, random_state=0, approach='dm', 
                       kernel='normalized_angle', alignment='procrustes')
  fc_RR = np.array(pd.read_csv('../results/GSR/fc/RR/'+i, header=None))
  align.fit(fc_RR,reference=group_grad_LL)
  grad_RR = align.aligned_
  np.savetxt(path_add+'RR/'+i, grad_RR, delimiter = ',')
  align = GradientMaps(n_components=10, random_state=0, approach='dm', 
                       kernel='normalized_angle', alignment='procrustes')
  fc_LR = np.array(pd.read_csv('../results/GSR/fc/LR/'+i, header=None))
  align.fit(fc_LR,reference=group_grad_LL)
  grad_LR = align.aligned_
  np.savetxt(path_add+'LR/'+i, grad_LR, delimiter = ',')
  align = GradientMaps(n_components=10, random_state=0, approach='dm', 
                       kernel='normalized_angle', alignment='procrustes')
  fc_RL = np.array(pd.read_csv('../results/GSR/fc/RL/'+i, header=None))
  align.fit(fc_RL,reference=group_grad_LL)
  grad_RL = align.aligned_
  np.savetxt(path_add+'RL/'+i, grad_RL, delimiter = ',')
  print('finish   ' + i)

# correct individual gradient if the correlation is negative
for dir in path_list:
  #LL
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
