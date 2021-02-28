import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import _pickle as pickle
import gc
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
import optuna
import os
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from hyperopt import fmin, tpe, hp
import sys

# import tensorflow as tf
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

pd.set_option('display.max_columns', 150)
pd.set_option('display.max_rows', 150)
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision
gpus = tf.config.experimental.list_physical_devices('GPU')

tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)]
            )
logical_gpus = tf.config.experimental.list_logical_devices('GPU')
print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")


policy = mixed_precision.Policy('mixed_float16')
# policy = mixed_precision.Policy('float32')
mixed_precision.set_policy(policy)

def utility_metric(date,weights, resp, action):
    import numpy as np
    p = []
    for i in np.unique(date):
        wi = weights[date == i]
        ri = resp[date == i]
        ai = action[date == i]
        pi = np.sum(wi * ri * ai)
        p.append(pi)
    p = np.array(p)
    
    nt = np.unique(date).shape[0]
#     print(nt)
    sp = np.sum(p)
    normp = np.sqrt(np.sum(np.square(p)))
    t = (sp / normp) * np.sqrt(250/nt)
    u = min(max(t,0), 6) * sp
    return u
    

def build_model(parameters):
    inputs = tf.keras.Input(shape = (131,))
    if parameters['norm']:
        x = tf.keras.layers.experimental.preprocessing.Normalization()(inputs)
    else:
        x = inputs
        
    for block in range(parameters['n_blocks']):
        for n in range(parameters['n_dense_per_block']):
            x = tf.keras.layers.Dense(parameters['dense_shape'][block], name = 'block_'+str(block)+'_dense_'+str(n))(x)
        if parameters['normalization'][block]:
            x = tf.keras.layers.BatchNormalization(name =  'block_'+str(block)+'_batch_norm')(x)
        x = tf.keras.activations.relu(x)
        tf.keras.layers.Dropout(parameters['dropouts'][block], name =  'block_'+str(block)+'_dropout')(x)
    x = tf.keras.layers.Dense(1, activation = 'sigmoid', name = 'classification_head')(x)
    model = tf.keras.Model(inputs, x)
    model.compile(  loss = 'binary_crossentropy',
                    optimizer = tf.keras.optimizers.Adam(parameters['lr']),
                    metrics = ['accuracy', 'AUC'])
    return model

def get_fold(fold_number, X_train, y_train, date_train, weights_train):
    filters = (date_train >= 44*fold_number)&(date_train < 44*(fold_number+1))
    filters = np.invert(filters)
    Xt, Xv, Yt, Yv = X_train[filters], X_train[np.invert(filters)], y_train[filters], y_train[np.invert(filters)]
    datet, datev, weightst, weightsv = date_train[filters], date_train[np.invert(filters)], weights_train[filters], weights_train[np.invert(filters)]
    sw = abs((Yt * weightst)) + 1
    yt, yv = (Yt > 0)*1, (Yv > 0)*1
    return Xt, Xv, yt, yv, Yt, Yv, datet, datev, weightst, weightsv, sw

def train(model, parameters, Xt, Xv, yt, yv, sw):
    sample_weight = sw if parameters['use_sample_weights'] else None
    
    epochs = 2
    early = EarlyStopping(monitor='val_auc', min_delta=0.0001, patience=8, verbose=1, 
                                                mode='max', restore_best_weights=True)

    reduce = ReduceLROnPlateau(monitor='val_auc', factor=0.1, patience=3, verbose=1, 
                               mode='max', min_delta=0.0001, cooldown=0, min_lr=0)
    callbacks =[early, reduce]
    
    history = model.fit(Xt, yt, validation_data = (Xv, yv), 
                  batch_size=parameters['batch_size'], epochs=epochs, callbacks = callbacks,
                           sample_weight = sample_weight, verbose = 2)
    sc = np.max(history.history['val_auc'])
    return model, sc

def make_experiment(fold_number, n_trials = 100):
    try:
        os.mkdir('./time_cv_ensembling/'+str(fold_number))
    except:
        1
    
    print("Loading Data")
    (X_train, X_test, y_train, y_test, date_train, date_test, weights_train, weights_test) = load('splitted_dataset')
    
    X_train = X_train.values
    X_test = X_test.values
    y_test_cat = (y_test > 0)*1
    
    print("Loading Fold")
    Xt, Xv, yt, yv, Yt, Yv, datet, datev, weightst, weightsv, sw = get_fold(fold_number, X_train, y_train, date_train, weights_train)
    del X_train
    del y_train
    gc.collect()
    
    print(Xt.shape, yt.shape, Yt.shape,datet.shape, weightst.shape, sw.shape)
    print(Xv.shape, yv.shape, Yv.shape,datev.shape, weightsv.shape)
    print(X_test.shape, y_test_cat.shape, y_test.shape,date_test.shape, weights_test.shape)
    
    print("Launching study")
    study = optuna.create_study(direction = 'maximize')
    study.optimize(get_objective((Xt, Xv, yt, yv, Yt, Yv, datet, datev, weightst, weightsv, sw, X_test, y_test, y_test_cat,date_test, weights_test, fold_number)), n_trials= n_trials)
    print("Study ended")

def get_objective(data):
    def objective(trial, data = data):
        assert data is not None , "Please inject some datas in the objective function"
        Xt, Xv, yt, yv, Yt, Yv, datet, datev, weightst, weightsv, sw, X_test, y_test, y_test_cat,date_test, weights_test, fold_number = data
        name = trial.suggest_int('name', 100000, 999999)
        ## Parameters
        n_blocks = trial.suggest_int('n_block', 2, 3)
        n_dense_per_block = trial.suggest_int('n_dense_per_block', 1, 2)

        dense_shape = []
        dropouts = []
        normalization = []

        for i in range(n_blocks):
            dense_shape.append(trial.suggest_categorical('dense_block_'+str(i), [64,128,256, 512, 1024]))
            dropouts.append(trial.suggest_uniform('dropout_block_'+str(i),0,0.4))
            normalization.append(trial.suggest_categorical('norm_block_'+str(i), [True, False])) 
        batch_size = trial.suggest_categorical("batch_size", [128, 256,512, 1024, 2048])
        lr = trial.suggest_categorical("lr", [0.01,0.001, 0.0001])
        norm = trial.suggest_categorical("norm", [True, False])
        use_sample_weights = trial.suggest_categorical("sample_weights", [True, False])

        parameters = {
            "name" : name,
            "n_blocks" : n_blocks,
            "n_dense_per_block" : n_dense_per_block,
            "dense_shape" : dense_shape,
            "dropouts" : dropouts,
            "normalization" : normalization,
            "batch_size" : batch_size,  
            'lr' : lr,
            "use_sample_weights" : use_sample_weights,
            "norm" : norm, 
        }

        ## Model building and training
        print('Model training, go grab a coffee')
        print(parameters)
        model = build_model(parameters)
        model, val_auc = train(model, parameters, Xt, Xv, yt, yv, sw)

        print("Model trained")
        ## Evaluation on val set
        print("Evaluation")
        parameters['val_auc'] = val_auc
        print("Val auc : " + str(val_auc))
        pred = model.predict(Xv, batch_size = parameters['batch_size'])[:,0]
        
        space = hp.normal('x', 0.5, 0.02)
        def f(x):
            action = (pred>x)*1
            utility = utility_metric(datev,weightsv, Yv, action)
            return -utility
        
        best = fmin(
            fn=f,  # "Loss" function to minimize
            space=space,  # Hyperparameter space
            algo=tpe.suggest,  # Tree-structured Parzen Estimator (TPE)
            max_evals=100  # Perform 1000 trials
        )

        parameters['val_treshold'] = best['x']
        action = (pred >= best['x'])*1
        val_utility = utility_metric(datev , weightsv , Yv , action)
        parameters['val_utility'] = val_utility
        print("Val_utility : " + str(val_utility))

        ## Evaluation on test set
        pred = model.predict(X_test, batch_size = parameters['batch_size'])[:,0]
        test_auc = roc_auc_score(y_test_cat, pred)
        print("Test Auc : " + str(test_auc))
        parameters['test_auc'] = test_auc
                
        space = hp.normal('x', 0.5, 0.02)
        def f(x):
            action = (pred>x)*1
            utility = utility_metric(date_test,weights_test, y_test, action)
            return -utility
        best = fmin(
                fn=f,  # "Loss" function to minimize
                space=space,  # Hyperparameter space
                algo=tpe.suggest,  # Tree-structured Parzen Estimator (TPE)
                max_evals=100  # Perform 1000 trials
            )
#         action = (pred >= study_test.best_params['x'])*1
        action = (pred >= best['x'])*1
        parameters['test_treshold'] = best['x']
        test_utility = utility_metric(date_test , weights_test , y_test , action)
        parameters['test_utility'] = test_utility
        print('Test utility : '+ str(test_utility))
        ## Parameters and model savings
        print("Saving")
        try:
            os.mkdir('./time_cv_ensembling/'+str(fold_number)+'/trial_'+str(name))
        except:
            1

        save(parameters, './time_cv_ensembling/'+str(fold_number)+'/trial_'+str(name)+'/parameters')
        model.save('./time_cv_ensembling/'+str(fold_number)+'/trial_'+str(name)+'/model')

        print("Next model")
        print('\n')
        return val_utility
    return objective

if __name__ == "__main__":
    print(sys.argv)
    fold_number = int(sys.argv[1])
#    print(fold_number)
    make_experiment(fold_number, n_trials = 2)