"""
Created on Tue Apr 09 11:58:40 2013

@author: baibhav
"""
import hog
import scipy.misc as sm
from random import randint
from sklearn import svm
from sklearn.externals import joblib as jl
from skimage import io, color
import glob
# import pickle


#Settings
classifiers = 3
neg_samples = 20
folder_pos = glob.glob("img/train/pos/*.*")
folder_neg = glob.glob("img/train/neg/*.*")
pos_file_num = len(folder_pos)
neg_file_num = len(folder_neg)

# Dividing Whole Training pos and neg folder into 3 chunks, coming up with 3 classifiers

for i in range(0,classifiers):
    hog_v = []
    hog_v1 = []
    hog_v2 = []
    target = []
    clf = None
    # '''
    # Positive Training
    pos = 0
    for files in folder_pos[0+pos_file_num/classifiers*i:(pos_file_num+pos_file_num*i)/classifiers]:
        pos +=1
        im = io.imread(files)
        im = color.rgb2gray(im)
        h,w = im.shape

        hog_v1 = hog.hog(im, orientations=9, pixels_per_cell=(6, 6),cells_per_block=(3, 3), visualise=False, normalise=True)
        hog_v.append(hog_v1)
        target.append(1)
        if (pos%100 == 0):
            print ("Positive Training ",i+1,", Progress status - ",pos*100/(pos_file_num/3),"%")
         
     
    # Negative Training
    neg = 0
     
    for files in folder_neg[0+neg_file_num/classifiers*i:(neg_file_num+neg_file_num*i)/classifiers]:
        neg +=1
        im = io.imread(files)
        im = color.rgb2gray(im)
        h,w = im.shape
        # Get Random image windows of size 128*64 in the big images
        for k in range(0,neg_samples):
            # Random left top point of the window
            pointy = randint(0, h-128)
            pointx = randint(0, w-64)
            window = im.copy()[pointy:pointy+128,pointx:pointx+64]
            hog_v2 = hog.hog(window, orientations=9, pixels_per_cell=(6, 6),cells_per_block=(3, 3), visualise=False, normalise=True)
            hog_v.append(hog_v2)
            target.append(0)
        if (neg%100 == 0):
            print ("Negative Training ",i+1,", Progress status - ",neg*100/(neg_file_num/3),"%")
     
    # print  hog_v," -> ",target
    # print  tuple(hog_v).__sizeof__()," -> ",tuple(target).__sizeof__()
    # SVM
    clf = svm.LinearSVC()
    clf.fit(hog_v, target)
      
    # Dump Model
    jl.dump(clf, 'model/model_'+str(i+1)+'.pkl')
    
    # s = pickle.dumps(clf)
    # '''
print "Training complete! Congratulations, you have created ",classifiers," classifiers!!!"
print "Total positive images - ",pos_file_num
print "Total negative images - ",neg_file_num
print neg_samples," random images of size 128x64 was used in training from each negative images!"