import tensorflow as tf
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,accuracy_score
import seaborn as sns
import cv2 as cv
import gc
import re
from tensorflow.keras.applications import VGG16, InceptionV3, ResNet50

class HOGFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, block_size=(16, 16), block_stride=(8, 8), cell_size=(8, 8), nbins=9):
        self.block_size = block_size
        self.block_stride = block_stride
        self.cell_size = cell_size
        self.nbins = nbins
        
        self.hog = cv.HOGDescriptor(
            _winSize=(64,128),
            _blockSize=self.block_size,
            _blockStride=self.block_stride,
            _cellSize=self.cell_size,
            _nbins=self.nbins
        )
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        hog_features = []
        for img in X:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            img = cv.resize(img,(64,128))
            features = self.hog.compute(img)
            hog_features.append(features.flatten())
        
        return np.array(hog_features)

class BoVWFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=100):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.centroids = None
        self.sift = cv.SIFT_create()

    def fit(self, X, y=None):
        all_descriptors = []
        for img in X:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            _, descriptors = self.sift.detectAndCompute(img, None)
            if descriptors is not None:
                all_descriptors.append(descriptors)
        all_descriptors = np.vstack(all_descriptors)
        self.kmeans.fit(all_descriptors)
        self.centroids = self.kmeans.cluster_centers_
        return self

    def transform(self, X):
        histograms = []
        for img in X:
            _, descriptors = self.sift.detectAndCompute(img, None)
            if descriptors is not None:
                words = self.kmeans.predict(descriptors)
                histogram, _ = np.histogram(words, bins=np.arange(self.n_clusters + 1))
            else:
                histogram = np.zeros(self.n_clusters)
            histograms.append(histogram)
        return np.array(histograms)

class Word2VecFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, vector_size=100, window=5, min_count=1, sg=0):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.sg = sg
        self.model = None

    def fit(self, X, y=None):
        tokenized_sentences = [sentence.split() for sentence in X]
        self.model = Word2Vec(sentences=tokenized_sentences, vector_size=self.vector_size, window=self.window, min_count=self.min_count, sg=self.sg)
        return self

    def transform(self, X):
        feature_vectors = []
        for text in X:
            vectors = [self.model.wv[word] for word in text if word in self.model.wv]
            
            if len(vectors) > 0 and all(len(vec) == len(vectors[0]) for vec in vectors):
                feature_vectors.append(np.mean(vectors, axis=0))
            else:
                zero_vector = np.zeros(self.model.vector_size)
            feature_vectors.append(zero_vector)
        
        return np.array(feature_vectors)
    
TXT_PIPELINES = [
    Pipeline([('tf',TfidfVectorizer()),('knn',KNeighborsClassifier())]),
    Pipeline([('tf',TfidfVectorizer()),('rf',RandomForestClassifier())]),
    Pipeline([('w2v',Word2VecFeatureExtractor()),('knn',KNeighborsClassifier())]),
    Pipeline([('w2v',Word2VecFeatureExtractor()),('rf',RandomForestClassifier())])
]

IMG_PIPELINES = [
    Pipeline([('hog',HOGFeatureExtractor()),('knn',KNeighborsClassifier())]),
    Pipeline([('hog',HOGFeatureExtractor()),('rf',RandomForestClassifier())]),
    Pipeline([('bowv',BoVWFeatureExtractor()),('knn',KNeighborsClassifier())]),
    Pipeline([('bowv',BoVWFeatureExtractor()),('rf',RandomForestClassifier())])
]

def binarize(images,labels):
    binlabels = tf.where(labels==0,0,1)
    return images, binlabels

def parse_args(yaml_fi):
    import yaml
    with open(yaml_fi,'r') as f:
        args = yaml.safe_load(f)
    parsed_args = {}

    def onelevel_parse(rcs):
        parsed = {}
        for mainkey,subdicts in rcs.items():
            openner = f'{mainkey}__'
            for key, values in subdicts.items():
                parsed.update({openner+key:values})
        return parsed    
            
    parsed_args.update(args['dset'])
    parsed_args.update(onelevel_parse(args['ml']['img']))
    parsed_args.update(onelevel_parse(args['ml']['txt']))
    parsed_args.update(onelevel_parse(args['ml']['models']))
    return parsed_args

def sklearn_dset(dataset,obj):
    batches = []
    for x, y in dataset.as_numpy_iterator():
        if obj == 'txt':
            x = re.sub("b'",'',str(x))
        batches.append((x,y))
        gc.collect()
    
    x, y = zip(*batches)
    x = np.array(x)
    y = np.array(y)
    return x, y

class ML:
    def __init__(self,obj='img',dset_path=None,yaml_path='config.yaml'):
        import os
        self.args = parse_args(yaml_path)
        self.obj = obj
        self.binary_labels = self.args['binary_labels']
        if dset_path is not None:
            path = dset_path
        else:
            path = self.args['path']
        
        if obj=='img':
            self.train = tf.keras.utils.image_dataset_from_directory(path + f'{obj}/train',labels='inferred',label_mode='int',color_mode='rgb',batch_size=None)#,image_size=(128,128))
            self.valid = tf.keras.utils.image_dataset_from_directory(path + f'{obj}/val',labels='inferred',label_mode='int',color_mode='rgb',batch_size=None)#,image_size=(128,128))
            self.test  = tf.keras.utils.image_dataset_from_directory(path + f'{obj}/test',labels='inferred',label_mode='int',color_mode='rgb',batch_size=None) # ,image_size=(128,128))
            self.pipelines = IMG_PIPELINES
            
        elif obj=='txt':
            self.train = tf.keras.utils.text_dataset_from_directory(path + f'{obj}/train',labels='inferred',label_mode='int',batch_size=None)
            self.valid = tf.keras.utils.text_dataset_from_directory(path + f'{obj}/val',labels='inferred',label_mode='int',batch_size=None)
            self.test  = tf.keras.utils.text_dataset_from_directory(path + f'{obj}/test',labels='inferred',label_mode='int',batch_size=None)
            self.pipelines = TXT_PIPELINES
            
        if self.binary_labels:
            self.train = self.train.map(binarize)
            self.valid = self.valid.map(binarize)
            self.test = self.test.map(binarize)
            
    def training(self,log_path = 'log.txt'):
        from sklearn.model_selection import RandomizedSearchCV
        
        x_train, y_train = sklearn_dset(self.train,self.obj)
        
        self.best_models = []
        for pipe in self.pipelines:
            print(f'Training {list(pipe.named_steps.keys())}')
            tmp_dict = {}
            for key, values in self.args.items():
                if key in pipe.get_params().keys():
                    tmp_dict.update({key:values})
            print(tmp_dict)

            rs = RandomizedSearchCV(
                pipe,
                param_distributions=tmp_dict,
                n_iter=50,
                verbose=3,
                cv=3,
                random_state=42
            )
            rs.fit(x_train,y_train)
            self.best_models.append(rs.best_estimator_)
            
            with open(log_path,'a') as f:
                send = f"""
Model {list(pipe.named_steps.keys())} - {rs.best_score_}
{rs.best_params_}
                """
                f.write(send)
            
    def test_accuracies(self):
        fig, axes = plt.subplots(2,2,figsize=(12,10))
        axes = axes.ravel()
        accs = []
        
        for i, model in enumerate(self.best_models):
            X_test, y_test = sklearn_dset(self.test,self.obj)
            y_pred = model.predict(X_test)
            accs.append(accuracy_score(y_test,y_pred))
            cm = confusion_matrix(y_test,y_pred)
            
            sns.heatmap(cm,annot=True,fmt='d',ax=axes[i])
            axes[i].set_title(f'Model {i + 1} Confusion Matrix\nAccuracy: {accs[i]:.2f}')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('True')
            
        plt.tight_layout()
        plt.show()