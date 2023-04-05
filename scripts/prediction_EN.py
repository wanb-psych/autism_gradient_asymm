'''
This is a python script for running supervised machine learning to predict ADOS score

data input : foldname
'''

import numpy as np
import pandas as pd
import scipy.stats as ss
import sklearn.linear_model as slm
from sklearn.model_selection import train_test_split
from neuroCombat import neuroCombat
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('../abide_demo_sort.csv')
site = np.array(df['site'])
site[site=='NYU'] = 1
site[site=='Pitt'] = 2
site[site=='USM'] = 3
site[site=='NYU_II'] = 4
site[site=='TCD'] = 5
df['site'] = site
df['site'] = pd.to_numeric(df['site'])
df['ados'] = df['ados_comm'] + df['ados_rrb'] + df['ados_social']

def harmony(data):
  cov = pd.DataFrame()
  cov['site'] = data[:,0]
  cov['age'] = data[:,1]
  data_feature = data[:,2:]
  data_combat = neuroCombat(dat=data_feature.T, mean_only= True,
                            covars=cov,
                            batch_col='site',
                            continuous_cols=['age'])['data']
  return data_combat.T

id = np.array(df['ID'][df['group']=='ASD'])
ados = np.array(df['ados'][df['group']=='ASD'])
comm = np.array(df['ados_comm'][df['group']=='ASD'])
rrb = np.array(df['ados_rrb'][df['group']=='ASD'])
social = np.array(df['ados_social'][df['group']=='ASD'])
age = np.array(df['age'][df['group']=='ASD'])
site = np.array(df['site'][df['group']=='ASD'], dtype=int)

id = id[comm != -999]
ados = ados[comm != -999]
social = social[comm != -999]
rrb = rrb[comm != -999]
site = site[comm != -999]
age = age[comm != -999]
comm = comm[comm != -999]

def model_elasticnet(m,y_name): # l1_ratio
  dic = {}
  for i in range(sample): 
    lr = slm.ElasticNetCV(alphas=[0.0001, 0.001, 0.01, 0.1, 1], l1_ratio=m, cv=5)
    x_train_harmony = harmony(x_train[i])
    x_test_harmony = harmony(x_test[i])
    model = lr.fit(x_train_harmony, y_train[i])
    y_pred_train = model.predict(x_train_harmony)
    y_pred_test = model.predict(x_test_harmony)
    corr_train = ss.pearsonr(y_pred_train, y_train[i])
    corr_test = ss.pearsonr(y_pred_test, y_test[i])
    a={}
    a['importance'] = model.coef_
    a['intercept'] = model.intercept_
    a['y_train'] = y_train[i]
    a['y_test'] = y_test[i]
    a['y_pred_train'] = y_pred_train
    a['y_pred_test'] = y_pred_test
    a['predict_train_r_p'] = np.array(corr_train)
    a['predict_test_r_p'] = np.array(corr_test)
    dic['train_test_'+str(i+1)] = a
    print('finish model......', str(i+1))
  np.save('../results/prediction_EN/'+ fn + y_name + '_l1ratio_' + str(m)+ '.npy', dic)
  return print('feature_' + '_l1ratio_' + str(m)+'   finished')

## hemisphere
foldname=['intra/', 'inter/']
for fn in foldname:
  feature = [None] * len(id)
  for i in range(len(id)):
    feature[i] = np.genfromtxt('../results/grad/'+fn+str(id[i])+'.csv', 
                               delimiter=',')[:,:3]

  x = np.vstack((np.array(feature).T[0],np.array(feature).T[1], np.array(feature).T[2])).T
  x = np.concatenate((site.reshape(site.shape[0],1), age.reshape(site.shape[0],1),x), axis=1)
  print(x.shape)
  # IMPORT sample train_test iterations
  sample = 100
  #random_state = np.random.randint(0, 1000, sample)
  y = [ados, comm, social, rrb]
  y_names = ['ados','comm', 'social','rrb']
  for y_i in range(4):
    x_train = [None] * sample
    y_train = [None] * sample
    x_test = [None] * sample
    y_test = [None] * sample
    for sam in range(sample):
      x_train[sam], x_test[sam], y_train[sam], y_test[sam] = train_test_split(
          x, y[y_i], test_size=0.2, random_state=sam)
    regulation = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for j in regulation:
      model_elasticnet(m=j, y_name = y_names[y_i])

#end
