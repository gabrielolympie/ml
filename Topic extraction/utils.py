import numpy as np
import _pickle as pickle
import random
import os
def save(file,name, folder = ""):
    if folder != "":
        outfile = open('./'+folder+'/'+name+'.pickle', 'wb')
    else:
        outfile = open(name+'.pickle', 'wb')
    pickle.dump(file, outfile)
    outfile.close
    
def load(name, folder = ""):
    if folder != "":
        outfile = open('./'+folder+'/'+name+'.pickle', 'rb')
    else:
        outfile = open(name+'.pickle', 'rb')
    file = pickle.load(outfile)
    outfile.close
    return file




def clean_name(x):
    x = x.replace('/', '_')
    for j in '.?!:,':
        x = x.replace(j, '')
    x = x.replace(' ', '_')
    return x

def save_batch(df, name, batch_size):
    size = df.shape[0]
    df1 = df.sample(n = size)
    n_batch = int(size / batch_size) + 1
    
    dir1 = './batch/'+clean_name(name)
    print(dir1)
    try:
        os.mkdir(dir1)
    except:
        1
    
    for i in tqdm(range(n_batch), leave = False):
        d = df1.iloc[batch_size*i : batch_size * (i+1)]['text'].values
        save(d, clean_name(name) +'_'+str(i), dir1[2:])
        
    return d