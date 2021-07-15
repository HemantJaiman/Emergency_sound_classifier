
import pandas as pd 
import librosa
import numpy as np
import glob
from sklearn import svm,metrics


#training data
trainl=pd.read_csv('labels.csv')
data_dir='train/'
    
files=glob.glob(data_dir+'/*.wav')

feat=[]
for i in range(len(files)):
    x,s = librosa.load(files[i],res_type='kaiser_fast')
    mfccs = np.mean(librosa.feature.mfcc(y=x,sr=s,n_mfcc=40).T,axis=0)
    feat.append(mfccs)
    
#testing data
test_dir='F:/ML PROJECTS/Emergency Sound/test'    
test_files=glob.glob(test_dir+'/*.wav')
test_feat=[]
for i in range(len(test_files)):
    x,s = librosa.load(test_files[i],res_type='kaiser_fast')
    print(i)
    mfccs = np.mean(librosa.feature.mfcc(y=x,sr=s,n_mfcc=40).T,axis=0)
    test_feat.append(mfccs)

    
l = np.asarray(trainl['label'])
f=np.asarray(feat)
    
l=l.reshape(-1,1)
test_feat=np.array(test_feat)

model = svm.SVC(kernel = "linear")

model.fit(f,l)
pred=model.predict(test_feat)

print(pred)


feat_mean = np.mean(feat,axis=0)
test_feat_mean = np.mean(test_feat[0],axis=0)

def find_nearest(array, value):
    #array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]
print(find_nearest(feat_mean, test_feat_mean))



