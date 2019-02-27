"""
Created on Tue May 07 11:58:40 2013

@author: baibhav
"""

import shelve,numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

get = shelve.open('test_score.dat')
target = get['target']
test_score = get['test_score']
get.close()

y = np.array(target)
scores = np.array(test_score)

# Error and percentage error

error = 0.0
for i in range(0,len(y)):
    output = 0 if scores[i]<0 else 1
    if y[i]!=output:
        error += 1

err_perc = round(error*100/len(y),2)

print ("Error percentage = ",err_perc,"%")


#Precision Recall curve
precision, recall, threshold = metrics.precision_recall_curve(y, scores)

plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall of Cascade Classifier')
plt.text(0.5,0.8, 'Error percentage = %.2f' %err_perc)
plt.show()

# y = np.asarray(y)
# scores = np.asarray(scores)

#ROC               
# fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)


  
