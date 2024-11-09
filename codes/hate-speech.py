import pandas as pd
import numpy as np
import os
import shutil
import pickle
import re
import gc
import cv2 as cv
from tqdm import tqdm
from multiprocessing import Process

def load_metadata(path='datasets/hate-speech/MMHS150K_GT.json'):
    metadata = pd.read_json(path).T
    metadata['id'] = metadata['tweet_url'].apply(lambda x: x.split('/')[-1])
    metadata.set_index('id',drop=True,inplace=True)
    return metadata

def majority_vote(keys):
    ANNOT = {x.split(' - ')[1]:x.split(' - ')[0] for x in '0 - NotHate, 1 - Racist, 2 - Sexist, 3 - Homophobe, 4 - Religion, 5 - OtherHate'.split(', ')}
    return ANNOT[sorted({k:' '.join(keys).count(k) for k in keys}.items(),key=lambda x:  x[1],reverse=True)[0][0]]

def clean_tweet(twt):
    twt = re.sub(r'rt |@[\w\W]+? |https[a-z0-9:/.]+|[^a-z ]','',twt)
    twt = ' '.join(re.findall(r'[a-z]{3,}',twt))
    return twt 

def data_pipeline(df: pd.DataFrame,filename: str):
    idxs = df.index.to_numpy()
    Y = df['labels_str'].apply(lambda x: majority_vote(x)).to_numpy()
    txts = df['tweet_text'].apply(lambda x: clean_tweet(x)).to_numpy()
    
    for i, idx in tqdm(enumerate(idxs)):
        label = Y[i]
        txt = txts[i]
        img_source = f'{idx}.jpg'
        
        destiny_img = f'datasets/hate-speech/img/{filename}/{str(label)}'
        destiny_txt = f'datasets/hate-speech/txt/{filename}/{str(label)}'
        os.makedirs(destiny_img,exist_ok=True)
        os.makedirs(destiny_txt,exist_ok=True)
        
        shutil.move('datasets/hate-speech/img_resized/'+img_source,destiny_img+'/'+img_source)
        with open(f'datasets/hate-speech/txt/{filename}/{str(label)}/{idx}.txt','w') as f:
            f.write(txt)
        gc.collect()

def manage_data():
    metadata = load_metadata()
    with open(r'datasets/hate-speech/splits/train_ids.txt','r') as f:
        train = metadata.loc[[idx[:-1] for idx in f.readlines()]]
    with open(r'datasets/hate-speech/splits/val_ids.txt','r') as f:
        val = metadata.loc[[idx[:-1] for idx in f.readlines()]]    
    with open(r'datasets/hate-speech/splits/test_ids.txt','r') as f:
        test = metadata.loc[[idx[:-1] for idx in f.readlines()]]
        
    processes = [Process(target=data_pipeline,args=(df,tag)) for df,tag in [(train,'train'),(val,'val'),(test,'test')]]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    
if __name__=='__main__':
    manage_data()