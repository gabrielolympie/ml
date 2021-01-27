import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import os
import random
from copy import deepcopy
import _pickle as pickle
import gc
from multiprocess import Pool
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import KBinsDiscretizer

from tensorflow.keras.optimizers import Adam, SGD
def save(file,name, folder = ""):
    if folder != "":
        outfile = open('./'+folder+'/'+name+'.pickle', 'wb')
    else:
        outfile = open(name+'.pickle', 'wb')
    pickle.dump(file, outfile, protocol=4)
    outfile.close
    
def load(name, folder = ""):
    
    if folder != "":
        outfile = open('./'+folder+'/'+name+'.pickle', 'rb')
    else:
        outfile = open(name+'.pickle', 'rb')
    file = pickle.load(outfile)
    outfile.close
    return file

class Discretiser:
    def __init__(self, nbins):
        self.nbins = nbins-1
        self.map_to = np.arange(self.nbins)/self.nbins
        
    def fit(self, X):
        ## X is a one dimension np array
        self.map_from = np.quantile(X, self.map_to)
        
    def transform(self, X):
        X1 = (np.interp(X, self.map_from, self.map_to, left=0, right=1, period=None) * self.nbins).astype(int)
        return X1
    
from tf_transformers2 import *
from tensorflow.keras.layers import Input, Dense, Dropout, TimeDistributed, LSTM
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

tokenizer = load('tokenizer')
dico_utags, dico_gtags, dico_parts, dico_tags = load('dico_tags')
dico_question = load('dico_questions_mean')
dico_cluster = load('dico_cluster')
#timestamp_enc, elapsed_enc,lag_time_enc, qmean_enc = load('discrete_encoders')
#w2v = load('w2v_ids_embeddings')

time_windows = ['all']
time_windows += ['pos_first_' + str(i) for i in [10,20]]
time_windows += ['pos_last_' + str(i) for i in [10,20]]
time_windows += ['tim_first_' + str(i) for i in [1,6,24,168]]
time_windows += ['tim_last_' + str(i) for i in [1,6,24,168]]

same_contents = ['all', 'content', 'parts', 'cluster']
all_contents = [] #['parts_' + str(i) for i in range(1,8)]# + ['cluster_' + str(i) for i in range(20)]

content_types = ['question', 'lectures']

kpis = {
    'question' : {
        'correctness' : ['mean', 'std', 'slope'],
        'timestamp_hour' : ['mean_diff', 'max_diff', 'mean_diff'],
        'prior_question_had_explanation' : ['mean', 'std'], 
#         'position' : ['mean_diff', 'max_diff', 'mean_diff'], 
        'question_mean' : ['mean', 'std', 'slope'],
        'elapsed_time_seconds'  : ['max', 'mean', 'slope'],
    },

    'lectures' : {
        'timestamp_hour' : ['max_diff', 'min_diff', 'mean_diff'],
        'position' : ['mean'], 
    }
}


def map_dict(ids, dico):
    def replace_dico(x):
        try:
            return dico[x]
        except:
            return 0
    return np.array(list(map(replace_dico,ids)))

def map_question_type(ids):
    def mapping(x):
        if x[0] == 'l':
            return 0
        elif x[0] == 'q':
            return 1
        else:
            return -1
    return np.array(list(map(mapping,ids)))

def apply_tokenizer(ids):
    return np.array(tokenizer.texts_to_sequences(ids)).reshape(len(ids))

def safe_divide(a, b):
    c = deepcopy(b)
    c[c == 0] = 1e9
    return a/c


def prepare(user_dico):
    user = deepcopy(user_dico)
    user['exercise_id_num'] = apply_tokenizer(user['exercise_id'])
    user['exercise_type'] = map_question_type(user['exercise_id'])
    user['parts'] = map_dict(user['exercise_id'], dico_parts)
    user['cluster'] = map_dict(user['exercise_id'], dico_cluster)
#     user['tags'] =  map_dict(user['exercise_id'], dico_tags)
    user['position'] = np.arange(len(user['exercise_id_num']))
    user['timestamp_seconds'] = user['timestamp']/(1000)
    user['timestamp_hour'] = user['timestamp']/(1000*3600)
    user['timestamp_minutes'] = user['timestamp']/(1000*60)
    user['timestamp_days'] = user['timestamp']/(1000*3600*24)
    user['question_mean'] = map_dict(user['exercise_id'], dico_question)
    user['elapsed_time_seconds'] = user['elapsed_time']/1000
    return user

def get_current_past(user, i):
    current = {
        'exercise_id_num'    : user['exercise_id_num'][i],
        'exercise_type'   : user['exercise_type'][i],
        'position'   : user['position'][i],
        'parts'    : user['parts'][i],
        'cluster'    : user['cluster'][i],
        'timestamp_hour'   : user['timestamp_hour'][i],
        'timestamp_seconds'   : user['timestamp_seconds'][i],
        'question_mean'    : user['question_mean'][i],
        'correctness' : user['correctness'][i],
    }
    past  =  {elt : user[elt][:i] for elt in user}
    return current, past

def get_time_windows_mask(current, past, strat):
    ## the array will be either time in hours, or time 
    if strat == 'all':
        return past['position'] >= 0
    else:
        parsed_strat = strat.split('_')
        t = parsed_strat[0]
        s = parsed_strat[1]
        n = float(parsed_strat[2])

        if t == 'pos':
            array = past['position']
            c = current['position']
        else:
            array = past['timestamp_hour']
            c = current['timestamp_hour']

        if s == 'first':
            m = 0
            M = min(n, c)
        else:
            M = c
            m = c - n
        return (array >= m) & (array < M)

def apply_mask(past, mask):
    return {elt : past[elt][mask] for elt in past}

def get_same_content_mask(current, past, strat):
    if strat == 'content':
        return past['exercise_id_num'] == current['exercise_id_num']
    elif strat == 'part':
        return past['parts'] == current['parts']
    elif strat == 'all':
        return past['position'] >=0
    else:
        return past['cluster'] == current['cluster']

def get_content_mask(past, strat):
    strat = strat.split('_')
    t = strat[0]
    n = int(strat[1])
    if t == 'parts':
        return past['parts'] == n
    else:
        return past['cluster'] == n

def get_content_type_mask(past, strat):
    if strat == 'question':
        return past['exercise_type'] == 1
    else:
        return past['exercise_type'] == 0
    
def get_kpis(current, past, key, metric):
    if key != 'correctness' and key != 'elapsed_time_seconds' and key != 'prior_question_had_explanation':
        c = current[key]
    array = past[key]
    
    if len(array) == 0:
        return -10
    else:
        if metric == 'sum':
            return array.sum()
        elif metric == 'mean':
            return array.mean()
        elif metric == 'min':
            return array.min()
        elif metric == 'max':
            return array.max()
        elif metric == 'std':
            return array.std()
        elif metric == 'max_diff':
            return (c - array).max()
        elif metric == 'min_diff':
            return (c - array).min()
        elif metric == 'mean_diff':
            return (c - array).mean()
        elif metric == 'slope' and len(array)>=2:
            l = len(array)//2
            return array[-l:].mean() - array[:l].mean()
        else:
            return -10

def hmean(a, b):
    if a + b == 0:
        return -10
    else:
        return 2*a*b/(a+b)



def get_features(current, past):
    final = deepcopy(current)
    for st_cont in same_contents:
        m_c = get_same_content_mask(current, past, st_cont)
        past_c = apply_mask(past, m_c)
        for st_time in time_windows:
            m_t = get_time_windows_mask(current, past_c, st_time) 
            past_t = apply_mask(past_c, m_t)
            for st_type in content_types:
                m_ty = get_content_type_mask(past_t, st_type)
                past_ty = apply_mask(past_t, m_ty)
                for field in kpis[st_type]:
                    for metric in  kpis[st_type][field]:
                        final[st_cont+'-'+st_time+'-'+st_type+'-'+field+'-'+metric] = get_kpis(current, past_ty, field, metric)

    for st_cont in all_contents:
        m_c = get_content_mask(past, st_cont)
        past_c = apply_mask(past, m_c)
        for st_time in time_windows:
            m_t = get_time_windows_mask(current, past_c, st_time) 
            past_t = apply_mask(past_c, m_t)  
            for st_type in content_types:
                m_ty = get_content_type_mask(past_t, st_type)
                past_ty = apply_mask(past_t, m_ty)
                for field in kpis[st_type]:
                    for metric in  kpis[st_type][field]:
                        final[st_cont+'-'+st_time+'-'+st_type+'-'+field+'-'+metric] = get_kpis(current, past_ty, field, metric)
    
    qm = final['question_mean'] ## question Mean
    um = final['all-all-question-correctness-mean']  # user mean
    cm = final['content-all-question-correctness-mean']  # user mean on content
    keys = list(final.keys())
    for elt in keys:
        if 'correctness' in elt:
            if 'mean' in elt:
                final[elt+'-question_hmean'] = hmean(qm, final[elt])
                final[elt+'-user_hmean'] = hmean(um, final[elt])
                final[elt+'-user_content_hmean'] = hmean(cm, final[elt])
    return final