
#from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from tensorflow import keras
from tensorflow.keras import layers
from keras.callbacks import EarlyStopping
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import yaml
from sklearn.decomposition import FastICA
from keras.models import Sequential
from keras.layers import Dense, Input
import matplotlib.pyplot as plt
import utils
import utils.custom_logger as cl
import work.data as d
from datetime import datetime
logger = cl.get_logger()

with open(r'config.yaml') as file:
    cfg = yaml.load(file, Loader=yaml.FullLoader)
    #TELCO_FILE = cfg['TELCO_FILE']
    SEED = cfg['SEED']
    EPOCHS = cfg['EPOCHS']
    SAMPLE_SIZE = cfg['SAMPLE_SIZE']
    ITERATIONS = cfg['ITERATIONS']
    MODEL_CONFIG_FILE = cfg['MODEL_CONFIG_FILE']


def gausian(_n):
    mu = 0      # Mean of the distribution
    sigma = 1   # Standard deviation of the distribution
    
    return np.random.normal(mu, sigma, _n)

def cauchy(_n):
    mu = 0    # Location parameter
    gamma = 1  # Scale parameter
    standard_samples = np.random.standard_cauchy(_n)

    return mu + gamma * standard_samples

def uniform(_n):
    
    low = 0    # Lower bound of the distribution
    high = 1   # Upper bound of the distribution

    return np.random.uniform(low, high, _n)

def print_summary(_x):
    print(" min:", str(round(np.min(_x),2)), 
          " max:", str(round(np.max(_x),2)), 
          " mean:", str(round(np.mean(_x),2)))

def standarize(_x):
    res = (_x - np.mean(_x)) / np.std(_x)
    res = res - np.min(res) + 1
    
    return res
    
def compute_D(_y, _z, _beta):
   
    _y = standarize(_y)
    _z = standarize(_z)

    
    if _beta > 0:
        res = sum((_y * (pow(_y, _beta) - pow(_z, _beta)) / _beta) - ((pow(_y, _beta+1) - pow(_z, _beta+1)) / (_beta+1)))
        return res
    
    if _beta == 0:
        res = sum(_y * np.log(_y/_z) - _y + _z)
        return res
 
    if _beta == -1:
        res = sum(np.log(_z/_y) - _y/_z - 1)
        return res

def find_cutoff(target, predicted):
    fpr, tpr, t = metrics.roc_curve(target, predicted)
    tnr = 1 - fpr
    g = np.sqrt(tpr*tnr)
    pos = np.argmax(g)

    return t[pos]    

    
def compute_J(_y, _beta, _gausian_ratio, _cauchy_ratio=0):

    n = len(_y)
    b_gausian = _gausian_ratio;
    b_uniform = 1 - _gausian_ratio - _cauchy_ratio
    b_cauchy = _cauchy_ratio; 


    gausian_part =  b_gausian * np.log(compute_D(_y, gausian(n), _beta) / compute_D(gausian(n), _y, _beta))
    uniform_part = b_uniform * np.log(compute_D(_y, uniform(n), _beta) / compute_D(uniform(n), _y, _beta))
    cauchy_part = b_cauchy * np.log(compute_D(_y, cauchy(n), _beta) / compute_D(cauchy(n), _y, _beta))

    res = pow(pow(gausian_part,2) + pow(uniform_part,2) + pow(cauchy_part,2), 0.5)
    #res =  min(gausian_part,uniform_part, cauchy_part)
    return res
    
    
def ica_model(_X):
    ica2 = FastICA()
    y = ica2.fit_transform(_X) 
    #w = ica.mixing_
    #w = ica.components_
    return y

def mape_score(_y_test, _y_pred):
    
    return sum(abs((_y_test - _y_pred) / _y_test)) / len(_y_test);

def mse_score(_y_test, _y_pred):
    
    return sum(pow(_y_test - _y_pred,2)) / len(_y_test);

def model_random_forest(_X_train, _y_train, _X_test, _y_test):
    m1 = RandomForestRegressor(n_estimators=10)
    m1 = m1.fit(_X_train, _y_train)
    mse, mape, r2, preds = model_preditor(m1, _X_test, _y_test)
    return mse, mape, r2, preds


def model_nn_custom(_layers, _sizes, _activations, _input_size, _last_activation):
    
    model = Sequential()
    model.add(Input(shape=(_input_size,)))
    
    for i in range(_layers):
        model.add(Dense(_sizes[i], activation=_activations[i]))
    
    model.add(Dense(1, activation=_last_activation))
    
    return model

def model_fiter(_model_name, _model,_X_train, _y_train, _X_test, _y_test, _quality_measures, _loss,  _scale=False):
    
    if _scale:
        sc = StandardScaler()
        _X_train = sc.fit_transform(_X_train)
        _X_test = sc.transform(_X_test)
            
        
    _model.compile(optimizer='adam', loss=_loss)
    es = EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True)
    
    _model.fit(_X_train, _y_train, epochs=EPOCHS, batch_size=32, validation_data=(_X_test, _y_test),  callbacks=[es], verbose=False)
    
    res_quality_measures, preds = model_preditor(_model_name, _model, _X_test, _y_test, _quality_measures)
    return res_quality_measures, preds


def model_preditor(_model_name, _model, _X_test, _y_test, _quality_measures):
    m1_predict = _model.predict(_X_test, verbose=False)
    
    if len(m1_predict.shape) > 1:
        m1_predict = m1_predict[:,0]
    
    res = {}
    if 'mse' in _quality_measures:
        res['mse'] = metrics.mean_squared_error(_y_test, m1_predict)

    if 'mape' in _quality_measures:
        res['mape'] = mape_score(_y_test, m1_predict)
        
    if 'r2' in _quality_measures:
        res['r2'] = metrics.r2_score(_y_test, m1_predict)

    if 'auc' in _quality_measures:
        res['auc'] = metrics.roc_auc_score(_y_test, m1_predict)
   
    if 'threshold' in _quality_measures:
        res['threshold'] = find_cutoff(_y_test,m1_predict)

    t = find_cutoff(_y_test,m1_predict)
    m1_predict_binary = [1 if x >= t else 0 for x in m1_predict]
    conf_matrix = np.round(metrics.confusion_matrix(_y_test, m1_predict_binary),2)
    

    if 'sensitivity' in _quality_measures:
        res['sensitivity'] = np.round(metrics.recall_score(_y_test, m1_predict_binary),2)
   
    if 'specificity' in _quality_measures:
        res['specificity'] = np.round(conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1]),2)
    
    if 'precision' in _quality_measures:
        res['precision'] = np.round(metrics.precision_score(_y_test, m1_predict_binary),2)
    
    if 'f1' in _quality_measures:
        res['f1'] = np.round(metrics.f1_score(_y_test, m1_predict_binary),2)
 

    fpr, tpr, thresholds = metrics.roc_curve(_y_test, m1_predict)

    roc_df = pd.DataFrame({
        'False Positive Rate': fpr,
        'True Positive Rate': tpr,
        'Thresholds': thresholds
    })
    
    
    curr_date = datetime.now().strftime("%Y%m%d_%H%M")
    roc_filename ='results/models/roc_' + _model_name + "_" + curr_date + ".csv"
    res['roc_filename'] = roc_filename
    roc_df.to_csv(roc_filename, sep=';')

    return res, m1_predict

  
def train_models(_X_train, _y_train, _X_test, _y_test, _quality_measures):
     
    res = pd.DataFrame(columns = ['model'] + _quality_measures)
    
    df_preds = pd.DataFrame({'y_actual': _y_test})
   
    model_cfg = pd.read_csv(MODEL_CONFIG_FILE, sep=";")

    for _,c in model_cfg.iterrows():
        logger.info("model " + c['model_name'] + " training start dataset size: " + str(len(_X_train)))
        
        sizes = [int(num) for num in c['sizes'].split(',')]
        activations = [ss for ss in c['activations'].split(',')]
        m1 = model_nn_custom(c['layers'], sizes, activations, len(_X_train.columns),  c['last_activation'])
        
        res_quality_measures, preds = model_fiter(c['model_name'], m1,_X_train, _y_train, _X_test, _y_test, _quality_measures, c['loss'], True)
    
        res_quality_measures['model'] = c['model_name']
       
        res = pd.concat([res, pd.DataFrame([res_quality_measures])], ignore_index=True)    
        df_preds[c['model_name']] = preds
        
        logger.info("model " + c['model_name'] + " training finished...")
                
    return df_preds, res


def compute_mse(_number_of_components, _y_actual, _model_results, _components_results):
    
    mses = np.zeros((_number_of_components + 1,_number_of_components + 1))
    
    for j in range(_number_of_components):
        mse = mse_score(_y_actual, _model_results.T[j])
        mses[0,j] = mse

    for i in range(_number_of_components):
        for j in range(_number_of_components):
    
            mse = mse_score(_y_actual, _components_results[:,i*_number_of_components + j])
         
            mses[i+1,j] = mse
            
    base = mses[0,0:_number_of_components]

    for i in range(_number_of_components):
       
        mse_reduction_prc = (mses[i+1,0:_number_of_components] - base) / base
        
        mses[i+1,_number_of_components] = np.mean(mse_reduction_prc)
    
    return mses 

    
def compute_divergence(_filename, _number_of_components):
    
    # _filename = 'results/ica/ica_detail_result_1_20240403_0950.csv'
    # _number_of_components= 5
    
    df = pd.read_csv(_filename, sep=';')
    components_results = df.iloc[:, 1 + _number_of_components:1 + 2*_number_of_components].values
    
    res_detail = pd.DataFrame(columns = list(['beta', 'gausian', 'uniform', 'cauchy']) + list(["c_" + str(k+1) for k in range(_number_of_components)]))    

    for b in [0, 1, 2, 3]:
        for r in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
            vals = {}
            for i in range(_number_of_components):
                J = compute_J(components_results[:, i], b, r, 1)
                vals['c_' + str(i+1)] = J

            
            new_row = {'beta': b, 'gausian': r, 'uniform': 1-r, 'cauchy': 0}
            new_row = {**new_row, **vals}
            new_row = pd.DataFrame(new_row, index=[0])
            res_detail = pd.concat([res_detail, new_row])

  
    return res_detail
    
def compute_ica(_predictions_file, _quality_measures):
   
    # _predictions_file = "results\model_predictions\models_predictions_1_20240402_1227.csv"
    # _quality_measures = ['mse', 'auc']
    
    df = pd.read_csv(_predictions_file, sep=';')
    
    y_actual = df['y_actual'].values
    x = df.drop('y_actual', axis=1).values
    x_full = df.values
    
    number_of_components = x.shape[1]
    number_of_cases = x.shape[0]
 
    ica = FastICA()
    
    y = ica.fit_transform(x) 
    
    number_of_measures = len(_quality_measures)
    measures = np.zeros((number_of_measures * number_of_components + number_of_measures, number_of_components))

    measure_names = []    
    scenarios = [] 
    
    ica_components = np.concatenate((x_full, y), axis=1)
    ica_columns = ["c_" + str(k+1) for k in range(0, number_of_components)]
    xp_colnames = []
    
    pos = 0
    for q in _quality_measures:
        for j in range(number_of_components):
            
            t = find_cutoff(y_actual, x.T[j])
            m1_predict_binary = [1 if x >= t else 0 for x in x.T[j]]
            conf_matrix = np.round(metrics.confusion_matrix(y_actual, m1_predict_binary),2)
            
            if q == 'mse': 
                measures[pos,j] = mse_score(y_actual, x.T[j])
           
            if q == 'mape': 
                measures[pos,j] = mape_score(y_actual, x.T[j])
            
            if q == 'auc': 
                measures[pos,j] = metrics.roc_auc_score(y_actual, x.T[j])
           
            if q == 'sensitivity':
                measures[pos,j] = np.round(metrics.recall_score(y_actual, m1_predict_binary),2)
           
            if q == 'specificity':
                measures[pos,j] = np.round(conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1]),2)
            
            if q == 'precision':
                measures[pos,j] = np.round(metrics.precision_score(y_actual, m1_predict_binary),2)
            
            if q == 'f1':
                measures[pos,j] = np.round(metrics.f1_score(y_actual, m1_predict_binary),2)
        
        scenarios.append('base')
        measure_names.append(q)
        pos = pos + 1


    for i in range(number_of_components):
        z = y.copy()
        z[:,i] = 0
        
        xp = ica.inverse_transform(z)
        xp_colnames = xp_colnames + ['c_' + str(i+1) + "_xp_" + str(k+1) for k in range(0, number_of_components)]
        ica_components = np.concatenate((ica_components, xp), axis=1)

        for q in _quality_measures:
            for j in range(number_of_components):
                t = find_cutoff(y_actual, xp[:,j])
                m1_predict_binary = [1 if x >= t else 0 for x in xp[:,j]]
                conf_matrix = np.round(metrics.confusion_matrix(y_actual, m1_predict_binary),2)
            
                if q == 'mse': 
                    measures[pos,j] = mse_score(y_actual, xp[:,j])
                if q == 'mape': 
                    measures[pos,j] = mape_score(y_actual, xp[:,j])
                if q == 'auc': 
                    measures[pos,j] = metrics.roc_auc_score(y_actual, xp[:,j])
                    
                if q == 'sensitivity':
                    measures[pos,j] = np.round(metrics.recall_score(y_actual, m1_predict_binary),2)
               
                if q == 'specificity':
                    measures[pos,j] = np.round(conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1]),2)
                
                if q == 'precision':
                    measures[pos,j] = np.round(metrics.precision_score(y_actual, m1_predict_binary),2)
                
                if q == 'f1':
                    measures[pos,j] = np.round(metrics.f1_score(y_actual, m1_predict_binary),2)
                
            
            scenarios.append('excluded_c_' + str(i+1))
            measure_names.append(q)
            pos = pos + 1

           
    res_ica = pd.DataFrame(ica_components, columns = list(df.columns)+ica_columns+xp_colnames)
        
    res_quality_measures = pd.DataFrame(measures, columns=df.drop('y_actual', axis=1).columns)
  
    
    tmp1 = pd.DataFrame(scenarios, columns=['scenario'])  
    tmp2 = pd.DataFrame(measure_names, columns=['measure'])   
    res_quality_measures = pd.concat([tmp1, tmp2, res_quality_measures], ignore_index=True, axis=1)
    
    res_quality_measures.columns =  list(tmp1.columns) + list(tmp2.columns) + list(df.drop('y_actual', axis=1).columns)

    res_quality_measures = res_quality_measures.sort_values(by=['measure', 'scenario'], ascending=[True, True])
    
    #res_quality_measures.to_csv('tmp.csv', sep=';')
    return res_quality_measures, res_ica
