import shutil
import numpy as np
import nibabel as nib
import os
import pandas as pd


### ABIDE I
df = pd.read_csv('../DONT/abideI_clean.csv')
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

path = '../../../p_02378/data/data_autism/1_fc/ABIDE-I/'
for i in range(len(dir_num)):
  file = path+dir_num[i]+'/hcp_processed/'+file_name[i]+'_func.dtseries.nii'  
  if os.path.exists(file):
    img = nib.load(file).get_fdata()
    print('executing sub.'+dir_num[i]+'......'+str(img.shape[0]))
   
df = pd.read_csv('../DONT/abideII_clean.csv')
dir_num = np.array(df['ID']).astype(str)
path = '../../../p_02378/data/data_autism/1_fc/ABIDE-II/'
for i in range(len(dir_num)):
  file_name = '%.3s'%np.array(df['Site'])[i] +'_'+dir_num[i]
  file = path+dir_num[i]+'/hcp_processed/'+file_name+'_func.dtseries.nii'  
  if os.path.exists(file):
    img = nib.load(file).get_fdata()
    print('executing sub.'+dir_num[i]+'......'+str(img.shape[0]))

