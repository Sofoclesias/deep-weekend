import pandas as pd
import numpy as np
import pickle
import re
import gc
import cv2 as cv
from tqdm import tqdm
from multiprocessing import Process, Queue

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

def data_pipeline(df: pd.DataFrame):
    Y = df['labels_str'].apply(lambda x: majority_vote(x)).to_numpy()[None]
    txts = df['tweet_text'].apply(lambda x: clean_tweet(x)).to_numpy()[None]
    imgs = []
    for idx in tqdm(df.index):
        imgs += [cv.resize(cv.cvtColor(cv.imread(f'datasets/hate-speech/img_resized/{idx}.jpg'),cv.COLOR_BGR2RGB),(500,500))]
        gc.collect        
    imgs = np.array(imgs)[None]
    return txts, imgs, Y

def get_data(picklefile = None):
    if picklefile is not None:
        with open(picklefile,'rb') as p:
            return pickle.load(f)
    else:
        metadata = load_metadata()
        with open(r'datasets\hate-speech\splits\train_ids.txt','r') as f:
            train = metadata.loc[[idx[:-1] for idx in f.readlines()]]
            
        with open(r'datasets\hate-speech\splits\val_ids.txt','r') as f:
            val = metadata.loc[[idx[:-1] for idx in f.readlines()]]
            
        with open(r'datasets\hate-speech\splits\test_ids.txt','r') as f:
            test = metadata.loc[[idx[:-1] for idx in f.readlines()]]
          
        queue = Queue()
        processes = [Process(target=data_pipeline,args=(df,)) for df in [train,val,test]]
        
        for p in processes:
            p.start()
        for p in processes:
            p.join()
        results = [queue.get() for _ in [train,val,test]]
            
        with open('hate-speech.pkl','wb') as f:
            pickle.dump(results,f)
            
        # ((TXT_T, IMG_T, Y_T), (TXT_v, IMG_v, Y_v), (TXT_t, IMG_t, Y_t))
        return results
    
if __name__=='__main__':
    get_data()