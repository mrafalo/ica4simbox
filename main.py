import os

if os.path.exists("C:/Program Files/NVIDIA/CUDA/v11.8/bin"):
    os.add_dll_directory("C:/Program Files/NVIDIA/CUDA/v11.8/bin")

if os.path.exists("C:/Program Files/NVIDIA/CUDNN/v8.6.0/bin"):
    os.add_dll_directory("C:/Program Files/NVIDIA/CUDNN/v8.6.0/bin")

    
import numpy as np
import pandas as pd
import work
import work.models as m
import work.data as d
#import work.visualize as v
import yaml    
import importlib
from datetime import datetime
import tensorflow as tf
import random
import utils
import utils.custom_logger as cl
import glob
import time
from collections import ChainMap
logger = cl.get_logger()
from sklearn import metrics

with open(r'config.yaml') as file:
    cfg = yaml.load(file, Loader=yaml.FullLoader)
    SEED = cfg['SEED']
    EPOCHS = cfg['EPOCHS']
    SAMPLE_SIZE = cfg['SAMPLE_SIZE']
    ITERATIONS = cfg['ITERATIONS']
    MODEL_CONFIG_FILE = cfg['MODEL_CONFIG_FILE']
    QUALITY_MEASURES = cfg['QUALITY_MEASURES']



def create_folders():
    
    os.makedirs('results/models', exist_ok=True)
    os.makedirs('results/ica', exist_ok=True)
    os.makedirs('results/divergence', exist_ok=True)
    os.makedirs('results/model_predictions', exist_ok=True)
    
def ica_results(_mask, _quality_measures):
    
    # _mask = "results/divergence/divergence*"
    # _quality_measures =  {'auc':'max'}

    div_files = glob.glob(_mask)
    
    iters = 0
    matched = 0
    res = pd.DataFrame(columns = ['divergence_filename', 'ica_filename', 'beta', 'gausian', 'uniform', 'cauchy', 'improvement_ratio' ] )
    
    t = 0
   
    for f in div_files:
        iters = iters + 1
        df_div = pd.read_csv(f, sep=';')
        
        source_summary_filename = df_div.loc[:,'source_summary_filename'][0]
        component_columns = [x for x in list(df_div.columns) if x not in ('beta', 'gausian', 'uniform', 'cauchy', 'source_filename', 'source_summary_filename')]
            
        min_values_per_column = df_div[component_columns].min()
        div_component_min = min_values_per_column.idxmin()
    
        beta = df_div.loc[df_div[div_component_min]==np.min(min_values_per_column),'beta'].values[0]
        gausian = df_div.loc[df_div[div_component_min]==np.min(min_values_per_column),'gausian'].values[0]
        uniform = df_div.loc[df_div[div_component_min]==np.min(min_values_per_column),'uniform'].values[0]
        cauchy = df_div.loc[df_div[div_component_min]==np.min(min_values_per_column),'cauchy'].values[0]
        
        df_ica = pd.read_csv(source_summary_filename, sep=';')
        model_columns = [x for x in list(df_ica.columns) if x not in ('scenario', 'measure', 'source_filename')]
        number_of_components = len(model_columns)
        
        for q in _quality_measures:
            
            base = df_ica.loc[(df_ica.scenario == 'base') & (df_ica['measure'] == q), model_columns].values
            components = df_ica.loc[(df_ica.scenario != 'base') & (df_ica['measure'] == q),model_columns].values
            
            ica_measure_reductions = []
            for i in range(number_of_components):
                measure_reduction = components[i,:] - base
     
                improvement_cnt = np.sum(measure_reduction>0)

                ica_measure_reductions.append(np.mean(measure_reduction))
                ica_component = "c_"+str(i+1)
                
                
                if  _quality_measures[q] == 'min':
                    if np.mean(measure_reduction) < t:
                        print(beta, q, ica_component, 'mean red:', str(np.round(np.mean(measure_reduction),3)), 'max red:', str(np.round(np.max(measure_reduction),3)))
                        
                       
                        if div_component_min == ica_component:
                          matched  = matched + 1
                        
                else:
                    if np.max(measure_reduction) > t:

                        if div_component_min == ica_component:
                          matched  = matched + 1
                          if beta>0:
                              print(round(gausian,2), round(uniform,2), round(cauchy,2), beta, div_component_min, np.round(np.max(measure_reduction),3),improvement_cnt )
        
            ica_component_max = "c_"+str(np.argmax(ica_measure_reductions)+1)
            ica_component_min = "c_"+str(np.argmin(ica_measure_reductions)+1)

    print("result for: " + q + " " + str(matched) + "/" + str(len(div_files) * len(_quality_measures)) + " matches!")


            
def compute_ica_iterator(_mask, _dest_folder):

    logger.info("ica compute start for mask " + _mask)
    
    np.random.seed(m.SEED)
    tf.keras.utils.set_random_seed(m.SEED)
    random.seed(m.SEED)    
    
    prediction_files = glob.glob(_mask)
    i = 0
    curr_date = datetime.now().strftime("%Y%m%d_%H%M")
    for f in prediction_files:
        logger.info("ica compute for file: " + f)
        i = i + 1
        res_mse, res_ica = m.compute_ica(f, QUALITY_MEASURES)
        res_mse['source_filename'] = f
        res_ica['source_filename'] = f
        res_mse.to_csv(_dest_folder+"/ica_summary_" + str(i) + "_"+ curr_date + ".csv", mode='w', header=True, index=False, sep=";")
        res_ica.to_csv(_dest_folder+ "/ica_detail_" + str(i) + "_"+ curr_date + ".csv", mode='w', header=True, index=False, sep=";")

    logger.info("ica compute finished; all OK!")

def compute_divergence_iterator(_mask, _dest_folder, _number_of_components):
  
    logger.info("divergence compute start for mask " + _mask)

    list_files = glob.glob(_mask )
    i = 0
    curr_date = datetime.now().strftime("%Y%m%d_%H%M")
    for f in list_files:
        logger.info("divergence compute for file: " + f)
        i = i + 1
        res = m.compute_divergence(f, _number_of_components)
        
        k = f.split(sep='_')    
    
        res['source_filename'] = f
        res['source_summary_filename'] = "results/ica/ica_summary_" + k[2] + "_" + k[3] + "_" + k[4]
        
        res.to_csv(_dest_folder+ "/divergence_detail_" + str(i) + "_"+ curr_date + ".csv", mode='w', header=True, index=False, sep=";")

    logger.info("divergence compute finished; all OK!")

    
def model_training_iterator(_dest):
    
    logger.info("training starting... epochs: " + str(EPOCHS))
     
    np.random.seed(m.SEED)
    tf.keras.utils.set_random_seed(m.SEED)
    random.seed(m.SEED)
    start = time.time()

    for i in range(ITERATIONS):
        logger.info("training iteration: " + str(i+1) + "/" + str(ITERATIONS))
        
        X_train, y_train, X_test, y_test = d.get_data();
    
        res_preds, res_summary = m.train_models(X_train, y_train, X_test, y_test,  QUALITY_MEASURES)
         
        curr_date = datetime.now().strftime("%Y%m%d_%H%M")
        predictions_file = _dest + "/models_predictions_" + str(i+1) + "_" + curr_date + ".csv"
        res_preds.to_csv(predictions_file, mode='w', header=True, index=False, sep=";")
        res_summary.to_csv(_dest + "/models_summary_" + str(i+1) + "_" + curr_date + ".csv", mode='w', header=True, index=False, sep=";")
        
        logger.info("done... iteration: " + str(i+1) + "/" + str(ITERATIONS))
        
        
    stop = time.time()
    
    elapsed_sec = stop-start
    logger.info("training finished!, elapsed: " + str(elapsed_sec//60) + " minutes")


def main():
    start = time.time()
    create_folders()
    
    logger.info("starting...")
    model_cfg = pd.read_csv(MODEL_CONFIG_FILE, sep=";")
    model_cnt = model_cfg['model_name'].count()
    
    #model_training_iterator("results/model_predictions")
    #compute_ica_iterator("results/model_predictions/models_predictions*", "results/ica/")
    compute_divergence_iterator("results/ica/ica_detail*", "results/divergence/", model_cnt)
    
    stop = time.time()
    elapsed_sec = stop-start
    logger.info("training finished!, elapsed: " + str(elapsed_sec//60) + " minutes")
        

#d.prepare_data()

main()
