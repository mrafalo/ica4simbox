import os
if os.path.exists("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.0/bin"):
    os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.0/bin")

if os.path.exists("C:/Program Files/NVIDIA/CUDNN/v8.9.7/bin"):
    os.add_dll_directory("C:/Program Files/NVIDIA/CUDNN/v8.9.7/bin")
    
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

with open(r'config.yaml') as file:
    cfg = yaml.load(file, Loader=yaml.FullLoader)
    SEED = cfg['SEED']
    EPOCHS = cfg['EPOCHS']
    SAMPLE_SIZE = cfg['SAMPLE_SIZE']
    ITERATIONS = cfg['ITERATIONS']
    MODEL_CONFIG_FILE = cfg['MODEL_CONFIG_FILE']
    QUALITY_MEASURES = cfg['QUALITY_MEASURES']


def ica_results(_mask, _reduction_threshold, _quality_measures):
    
    _mask = "results/divergence/divergence*"
    
    div_files = glob.glob(_mask)
    matched = 0
    iters = 0

    for f in div_files:
        iters = iters + 1
        
      #  print('divergencefile:', f)
        df_div = pd.read_csv(f, sep=';')
        source_summary_filename = df_div.loc[:,'source_summary_filename'][0]
       # print('source_summary_filename:', source_summary_filename)
        
        component_columns = [x for x in list(df_div.columns) if x not in ('beta', 'gausian', 'uniform', 'cauchy', 'source_filename', 'source_summary_filename')]
            
        min_values_per_column = df_div[component_columns].min()

        div_component = min_values_per_column.idxmin()
        
        #print("divergence component: ", div_component)

        df_ica = pd.read_csv(source_summary_filename, sep=';')
        model_columns = [x for x in list(df_ica.columns) if x not in ('scenario', 'measure', 'source_filename')]
        number_of_components = len(model_columns)
        
        for q in _quality_measures:
            base = df_ica.loc[(df_ica.scenario == 'base') & (df_ica['measure'] == q), model_columns].values
            components = df_ica.loc[(df_ica.scenario != 'base') & (df_ica['measure'] == q),model_columns].values
            
            for i in range(number_of_components):
                measure_reduction_prc = (components[i,:] - base) / base
           
                if np.mean(measure_reduction_prc) < _reduction_threshold:
                    ica_component = "c_"+str(i+1)
                    if div_component == ica_component:
                        matched = matched + 1
                    
    
    print("result: " + str(matched) + "/" + str(iters) + " matches!")


            
def compute_ica_iterator(_mask, _dest_folder):
    #_mask = "results/model_predictions/models_predictions*"
   
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
    # _mask = "results/test2/ica_detail*"
    
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
    logger.info("starting...")
    model_cfg = pd.read_csv(MODEL_CONFIG_FILE, sep=";")
    model_cnt = model_cfg['model_name'].count()
    
    #model_training_iterator("results/model_predictions")
    compute_ica_iterator("results/model_predictions/models_predictions*", "results/ica/")
    compute_divergence_iterator("results/ica/ica_detail*", "results/divergence/", model_cnt)
    
    stop = time.time()
    elapsed_sec = stop-start
    logger.info("training finished!, elapsed: " + str(elapsed_sec//60) + " minutes")
        
# importlib.reload(work.models)
# importlib.reload(work.data)
# importlib.reload(work.visualize)

ica_results("divergence/divergence*", -0.01, ['auc'])

#v.plot_heat_map_ica('results/test2\ica_mse_result_40_20240304_2210.csv', 'plots/heat6.png')
    
main()


# from sklearn.tree import DecisionTreeClassifier
# from sklearn import metrics
# X_train, y_train, X_test, y_test = d.get_data();
# from sklearn.tree import export_text

# # Create Decision Tree classifer object
# clf = DecisionTreeClassifier()

# # Train Decision Tree Classifer
# clf = clf.fit(X_train,y_train)

# #Predict the response for test dataset
# y_pred = clf.predict(X_test)
# print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# tree_rules = export_text(clf, feature_names=list(X_train.columns))

# print(tree_rules)