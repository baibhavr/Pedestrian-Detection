"""
Created on Tue Apr 09 11:58:40 2013

@author: baibhav
"""
import hog
from sklearn.externals import joblib as jl
from skimage import io, color
import numpy as np
import scipy.misc as sm
import cv
import matplotlib.pyplot as plt

def img_draw_rect (img, x1, y1, x2, y2, v):
    try:
        img[y1,x1:x2+1,:] = v
        img[y2,x1:x2+1,:] = v
        img[y1:y2+1,x1,:] = v
        img[y1:y2+1,x2,:] = v
    except:
        print ("img[",y1,":",y2,"+1,",x2,"]")
    return True

def nms(imc,score,inc_ratio):
    h,w = score.shape
    for i in range(0,h):
        for j in range(0,w):
            try:
                if(score[i,j]<=score[i+1,j-1]
                   or score[i,j]<=score[i+1,j]
                   or score[i,j]<=score[i+1,j+1]
                   or score[i,j]<=score[i,j+1]):
                    score[i,j] = 0
            except:
                print ("Index error!")
                continue
            if (score[i,j]>0):
                imc = img_draw_rect (imc, j/inc_ratio, i/inc_ratio,(j+64)//inc_ratio,(i+128)//inc_ratio, 255)
    return imc
            
imc = io.imread("img/fm/fm3107.jpg")
im = color.rgb2gray(imc)

h,w = im.shape
# im_detect = im.copy()#cv.fromarray(im.copy())
print ("im shape",im.shape)

# for classes in range(0,3):
    # Load Model
# clf = jl.load('model/model_'+str(classes+1)+'.pkl')
clf = jl.load('model/model_1.pkl')

# Create Sliding Window sizes appropriate for the given image
# for window_sw in range(64,65 if (h>=128 and w>=64) else (w if w<64 else h//2)):
for window_sw in range(h//4,h//2,10):
    window_sh = window_sw*2
    
    inc_ratio = 64.0/window_sw # Width Increase factor of image necessary to make windows of width 64
    im_big = sm.imresize(im.copy(),(int(h*inc_ratio),int(w*inc_ratio)),'cubic')
    print ("inc_ratio = ",inc_ratio,", (window_sh,window_sw) = ",window_sh,",",window_sw," -> ",im_big.shape)
    # Slide windows in the image
    print ("im_big shape = ",im_big.shape)
    nh,nw = im_big.shape
#     window = np.zeros((window_sy,window_sx),dtype=np.int)

#     try:
    score = np.zeros((nh-128,nw-64,3),dtype=np.float)
    for i in range(0,nh-128,int(inc_ratio+10)):
        for j in range(0,nw-64,int(inc_ratio+10)):
            window = im_big.copy()[i:i+128,j:j+64]
#             edge
            hog_v = hog.hog(window, orientations=9, pixels_per_cell=(6, 6),cells_per_block=(3, 3), visualise=False, normalise=True)
            # Predict
            if(clf.decision_function([hog_v])>0):
                score[i,j,0] = clf.decision_function([hog_v])
                print ("i,j -> ",i,",",j)
                try:
                    if(score[i,j,0]<=score[i-1,j-1,0] 
                       or score[i,j,0]<=score[i-1,j,0]
                       or score[i,j,0]<=score[i,j-1,0]
                       or score[i,j,0]<=score[i-1,j+1,0]):
                        score[i,j,0] = 0
                except:
                    score[i,j,0] = 0
                    continue
#                 if (clf.decision_function([hog_v])>0):
#                     img_draw_rect (imc, j/inc_ratio, i/inc_ratio,(j+64)//inc_ratio,(i+128)//inc_ratio, 255)
                
    imc = nms(imc,score,inc_ratio)
    
    #                 cv.Rectangle(im_detect, (j/inc_ratio,i/inc_ratio), ((j+128)/inc_ratio,(i+64)/inc_ratio), 255)
    #                 print bool(clf.predict([hog_v]))
    #                 print "Found at [",j/inc_ratio,",",i/inc_ratio,"] "," Size ",window_sw,",",window_sh
#     except:
#         print "Error!"
#         continue
                
plt.subplot(121);plt.title("Big Image");plt.imshow(im_big);plt.set_cmap ('gray')
plt.subplot(122);plt.title("With Rectangle");plt.imshow(imc);
plt.show()