"""
Created on Tue Apr 09 11:58:40 2013

@author: baibhav

Calculates Target and prediction score 
Saves the variables into test_score.dat file

"""
import hog,shelve,glob
# from sklearn import svm
from random import randint
from sklearn.externals import joblib as jl
from skimage import io, color

hog_v = []
target = []
neg_samples = 20

# Positive Test
for files in glob.glob("img/test/pos/*.*"):
    im = io.imread(files)
    im = color.rgb2gray(im)
    hog_v1 = hog.hog(im, orientations=9, pixels_per_cell=(6, 6),cells_per_block=(3, 3), visualise=False, normalise=True)
    hog_v.append(hog_v1)
    target.append(1)
#     print files

# Negative Test
for files in glob.glob("img/test/neg/*.*"):
    im = io.imread(files)
    im = color.rgb2gray(im)
    h,w = im.shape
    for k in range(0,neg_samples):
        # Random left top point of the window
        pointy = randint(0, h-128)
        pointx = randint(0, w-64)
        window = im.copy()[pointy:pointy+128,pointx:pointx+64]
        hog_v1 = hog.hog(window, orientations=9, pixels_per_cell=(6, 6),cells_per_block=(3, 3), visualise=False, normalise=True)
        hog_v.append(hog_v1)
        target.append(0)
#     print files

print (tuple(hog_v).__sizeof__()," -> ",tuple(target).__sizeof__())
# SVM
# clf = svm.SVC()
 
# Load Model
clf1 = jl.load('model/model_1.pkl')
clf2 = jl.load('model/model_2.pkl')
clf3 = jl.load('model/model_3.pkl')

d1 = clf1.decision_function(hog_v)
d2 = clf2.decision_function(hog_v)
d3 = clf3.decision_function(hog_v)

# Predict
# p1 = clf1.predict(hog_v)
# p2 = clf2.predict(hog_v)
# p3 = clf3.predict(hog_v)

# print 'Target ',target, ", Sum =",sum(target)
# print 'Prediction 1 ',p1, ", Sum =",sum(p1)
# print 'Prediction 2 ',p2, ", Sum =",sum(p2)
# print 'Prediction 3 ',p3, ", Sum =",sum(p3)

#Decision Function
# print 'Decision Function 1 ',d1
# print 'Decision Function 2 ',d2
# print 'Decision Function 3 ',d3

cascade_score = d1+d2+d3
print ('Decision Function ',cascade_score)

x = shelve.open('test_score.dat')
x['target'] = target
x['test_score'] = cascade_score
x.close()
