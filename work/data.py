import pandas as pd
import numpy as np
import os
import yaml
from sklearn.model_selection import train_test_split
import utils.custom_logger as cl
import glob

logger = cl.get_logger()

with open(r'config.yaml') as file:
    cfg = yaml.load(file, Loader=yaml.FullLoader)
    CDR_FILE = cfg['CDR_FILE']
    SIMBOX_DATA_FILE = cfg['SIMBOX_DATA_FILE']
    SEED = cfg['SEED']

def load_ica_results(_mask):

    path_pattern = 'results/' + _mask

    csv_files = glob.glob(path_pattern)
    df = pd.concat((pd.read_csv(file, sep=';') for file in csv_files), ignore_index=True)

    
    return df


def prepare_data():
    df = load_data_file(CDR_FILE)
    
    result1 = df.groupby(['caller_id','caller_imei']).agg({
        'bts': 'nunique',
        'called_id': 'count',
        'duration_sec': 'mean',
        'roaming': 'sum',
        'target': 'max'}).reset_index()
    
    result1.columns = ['caller_id', 'caller_imei', 'bts_cnt', 'outgoing_cnt', 'mean_duration_sec', 'roaming_cnt', 'target']
    
    result3 = df.groupby(['called_id']).agg({'caller_id': 'count'}).reset_index()   
    result3.columns = ['caller_id', 'incoming_cnt']
    
    # Step 1: Count how many caller_id are associated with each imei
    imei_caller_count = df.groupby('caller_imei')['caller_id'].nunique().reset_index(name='caller_count')
    
    res = pd.merge(result1, imei_caller_count, on='caller_imei', how='left')
    # res = pd.merge(res, result2, on='caller_id', how='left')
    res = pd.merge(res, result3, on='caller_id', how='left')

    res.to_csv(SIMBOX_DATA_FILE, mode='w', header=True, sep=';', index=False)

    return res

def get_data():
    
    logger.info("loading data: " + SIMBOX_DATA_FILE)
    
    df = load_data_file(SIMBOX_DATA_FILE)

    df.dropna(inplace=True) 
    
    X = df.drop(["target", "caller_id", "caller_imei"], axis=1)
    y = df['target']
    
    for c in X.columns:
        t = X[c].dtype
        if not (t in ['float64', 'int64']):
            X.drop(c, axis=1, inplace=True)
             
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
   
    res_X_train = X_train
    res_y_train = y_train
    res_X_test = X_test
    res_y_test = y_test    
    
    logger.info("dataset size: " + str(len(df)) + " train dataset size: " + str(len(res_X_train)))
    
    return res_X_train, res_y_train, res_X_test, res_y_test


def load_data_file(_data_file):
    df = pd.read_csv(_data_file, sep=';')
    
    return df



