# How to run the codes

## Activate the enviornment  
- `$ conda activate autism`  
- `$ cd [working directory]/autism/`

## Clean the data according to the requirements
- Here, I deleted the subjects with bad quality of MRI
- And remove IQ<70 or without IQ information
- Keep all the boys
- Age from 5-40 years
- Head motion with mean_FD<0.3mm
- Finally there are 5 datasites survived including 283 subjects.

## Prepare the phenotype dataframe  
- ID, site, group, age, FIQ, ADOS_social, ADOS_communication, ADOS_rrb, mean_FD  
`$ python scripts/data_sort.py`  
The **output** here is: 'abide_demo_sort.csv' 

## Process the fMRI data from time series to FC to gradients  
- **FC**   
**Input**: '../data/data_autism/1_fc/'  
`$ python scripts/data_process_fc.py`  
**Output**: 'results/fc/'    
- **Gradients**  
**Input**: 'results/fc/' 
`$ python scripts/data_process_grad_HCP_template.py`  
**Output**: 'results/grad/'

## Statics  
`$ jupyter-lab`  
  1. Demographics  
  **click** the *'scripts/vis_basic_stas.ipython'* (Table S1)
  2. Comparisons between ASD and controls  
  **click** the *'scripts/vis_main.ipython'* (Figures 1, 3, 4, S1, S2, S8, S9 and Tables S2, S4, S5, S6)  
  During this period, inter-subject correlations have done with running:  
  `$ python scripts/spin_permutation` 
  3. Machine learning prediction  
  `$ python scripts/prediction.py`  
  **Output**: 'results/prediction/'  
  then in ipython notebook, **click** the *'scripts/vis_prediction.ipynb'* (Figures 2, S3, S4, S5, S6, S7 and Table S3)
  4. Enrichment analyses  
  **click** the *'scripts/vis_enrichment.ipynb'* (Figures 5, S10)
  5. Global signal regression  
  **Input:** '../data/data_autism/1_fc/'  
  `$ python scripts/data_process_GSR.py`  
  **Output**: 'results/GSR/'# autism_gradient_asymm
