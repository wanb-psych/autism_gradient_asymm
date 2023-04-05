import shutil
import numpy as np
import nibabel as nib
import os
import pandas as pd

try:
  shutil.rmtree('../results/fc/', ignore_errors=False, onerror=None)
except:
  pass
  
try:
  os.mkdir('../results/fc/')
  os.mkdir('../results/fc/full/')
  os.mkdir('../results/fc/LL/')
  os.mkdir('../results/fc/RR/')
  os.mkdir('../results/fc/LR/')
  os.mkdir('../results/fc/RL/')
  os.mkdir('../results/fc/intra/')
  os.mkdir('../results/fc/inter/')
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

mmp_lr_clean = mmp_lr[medial_wall==1] # Glasser atlas 10k fs_LR without medial wall

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
    corr_matrix = np.corrcoef(img_mmp.T)
    corr_matrix[corr_matrix>0.99999] = 1
    fc = np.arctanh(corr_matrix)
    fc[fc == np.inf] = 0
    np.savetxt('../results/fc/full/'+dir_num[i]+'.csv', fc, delimiter=',')
    fc_LL = fc[0:180,0:180]
    fc_RR = fc[180:360,180:360]
    fc_LR = fc[0:180,180:360]
    fc_RL = fc[180:360,0:180]
    fc_LLRR = fc_LL - fc_RR
    fc_LRRL = fc_LR - fc_RL
    np.savetxt('../results/fc/LL/'+dir_num[i]+'.csv', fc_LL, delimiter=',')
    np.savetxt('../results/fc/RR/'+dir_num[i]+'.csv', fc_RR, delimiter=',')
    np.savetxt('../results/fc/LR/'+dir_num[i]+'.csv', fc_LR, delimiter=',')
    np.savetxt('../results/fc/RL/'+dir_num[i]+'.csv', fc_RL, delimiter=',')
    np.savetxt('../results/fc/intra/'+dir_num[i]+'.csv', fc_LLRR, delimiter=',')
    np.savetxt('../results/fc/inter/'+dir_num[i]+'.csv', fc_LRRL, delimiter=',')

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
    img_cortex=img[:,0:18722] 
    for m in range(img.shape[0]):
      for n in range(360):
        img_mmp[m][n]=np.mean(img_cortex[m][mmp_lr_clean==n+1])
    corr_matrix = np.corrcoef(img_mmp.T)
    corr_matrix[corr_matrix>0.99999] = 1
    fc = np.arctanh(corr_matrix)
    fc[fc == np.inf] = 0
    np.savetxt('../results/fc/full/'+dir_num[i]+'.csv', fc, delimiter=',')
    fc_LL = fc[0:180,0:180]
    fc_RR = fc[180:360,180:360]
    fc_LR = fc[0:180,180:360]
    fc_RL = fc[180:360,0:180]
    fc_LLRR = fc_LL - fc_RR
    fc_LRRL = fc_LR - fc_RL
    np.savetxt('../results/fc/LL/'+dir_num[i]+'.csv', fc_LL, delimiter=',')
    np.savetxt('../results/fc/RR/'+dir_num[i]+'.csv', fc_RR, delimiter=',')
    np.savetxt('../results/fc/LR/'+dir_num[i]+'.csv', fc_LR, delimiter=',')
    np.savetxt('../results/fc/RL/'+dir_num[i]+'.csv', fc_RL, delimiter=',')
    np.savetxt('../results/fc/intra/'+dir_num[i]+'.csv', fc_LLRR, delimiter=',')
    np.savetxt('../results/fc/inter/'+dir_num[i]+'.csv', fc_LRRL, delimiter=',')
