import numpy as np
import pandas as pd
import os

df1 = pd.read_csv('../data/abideI_clean.csv')
df2 = pd.read_csv('../data/abideII_clean.csv') 

abide1 = pd.DataFrame([df1['ID'],df1['Group'],df1['Site'],df1['Age'], df1['IQ'], df1['ADOS_Comm'],
                       df1['ADOS_Soc'],df1['ADOE_Rep'], df1['Func_MeanFD']]).T
abide2 = pd.DataFrame([df2['ID'],df2['Group'],df2['Site'],df2['Age'], df2['FIQ'], df2['ADOS_G_COMM'],
                       df2['ADOS_G_SOCIAL'],df2['ADOS_G_STEREO_BEHAV'], df2['MeanFD_Jenkinson']]).T

a = df1['Age'][df1['Age']<40]
adult = a[a>17.9999]
kid = a[a<12]
adol = a[a>11.9999]
adol = adol[adol<18]

b = df2['Age'][df2['Age']<40]
adult2 = b[b>17.9999]
kid2 = b[b<12]
adol2 = b[b>11.9999]
adol2 = adol2[adol2<18]

abide1_adult = abide1.iloc[adult.index]
abide2_adult = abide2.iloc[adult2.index]
abide1_adol = abide1.iloc[adol.index]
abide2_adol = abide2.iloc[adol2.index]
abide1_kid = abide1.iloc[kid.index]
abide2_kid = abide2.iloc[kid2.index]

df = np.concatenate((abide1_adult,abide2_adult,abide1_adol,abide2_adol,abide1_kid, abide2_kid))
abide = pd.DataFrame()
abide['ID'] = df[:,0]
abide['group'] = df[:,1]
abide['site'] = df[:,2]
abide['age'] = df[:,3]
abide['FIQ'] = df[:,4]
abide['ados_comm'] = df[:,5]
abide['ados_social'] = df[:,6]
abide['ados_rrb'] = df[:,7]
abide['meanFD'] = df[:,8]

abide_sort =  abide.sort_values(by='ID')
abide_sort.to_csv('../abide_demo_sort.csv', index = None)