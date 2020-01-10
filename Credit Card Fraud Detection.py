#!/usr/bin/env python
# coding: utf-8

# In[1]:


#IMPORT LIBRARIES

import pandas as pd 
import numpy as np

from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn import svm

import itertools

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import seaborn

get_ipython().run_line_magic('matplotlib', 'inline')


# # Data recuperation

# In[2]:


data = pd.read_csv('C:/Users/Christine Rasi/proyek_dami/creditcard.csv') # read data dalam bentuk .csv
df = pd.DataFrame(data) # Convert data ke Dataframe
df


# In[3]:


df.dtypes


# # Data Visualization

# In[4]:


df.describe() # Deskripsi data menggunakan Fitur Statistik 


# In[5]:


# Pulihkan fraud data
df_fraud = df[df['Class'] == 1]
plt.figure(figsize=(15,10))
# Tampilan jumlah fraud berdasarkan waktu
plt.scatter(df_fraud['Time'], df_fraud['Amount']) 
plt.title('Scratter Plot Amount Fraud')
plt.xlabel('Time')
plt.ylabel('Amount')
plt.xlim([0,175000])
plt.ylim([0,2500])
plt.show()


# In[6]:


nb_big_fraud = df_fraud[df_fraud['Amount'] > 1000].shape[0] 
print('Terdapat '+str(nb_big_fraud) + ' penipuan dari 1000 data lebih yang diambil, dari ' + str(df_fraud.shape[0]) + ' penipuan asli')


# 
# 
# Unbalanced data
# 

# In[7]:


number_fraud = len(data[data.Class == 1])
number_no_fraud = len(data[data.Class == 0])
print('Terdapat '+ str(number_fraud) + ' penipuan dalam dataset asli, sedangkan non-penipuan berjumlah  ' + str(number_no_fraud) +' dalam dataset.')


# Dataset ini tidak seimbang sehingga mengakibatkan terdapat kelakuan yang tidak diinginkan  dari pengawas classifier. Sehingga untuk memahaminya dengan mudah maka dilakukan train dataset untuk mengetahui keakuratan data untuk mendapatkan label tiap transaksi yang dilakukan apakah merupakan fraud atau non-fraud

# In[8]:


print("Akurasi yang dihasilkan : "+ str((284315-492)/284315))


# # Correlation of features

# In[9]:


df_corr = df.corr()


# In[10]:


plt.figure(figsize=(15,10))

seaborn.heatmap(df_corr, cmap="YlGnBu") 
seaborn.set(font_scale=2,style='white')

plt.title('Heatmap correlation')
plt.show()


# Berdasarkan gambar di atas, dapat diketahui bahwa sebagian besar fitur berkorelasi negatif. Untuk mengatasinya perlu dilakukan dimension reduction atau PCA

# In[11]:


# Retrieving the correlation coefficients per feature in relation to the feature class
rank = df_corr['Class'] 
df_rank = pd.DataFrame(rank) 
 # Ranking the absolute values of the coefficients
df_rank = np.abs(df_rank).sort_values(by='Class',ascending=False)
# Removing Missing Data (not a number)                                                  
df_rank.dropna(inplace=True) 


# # Data Selection

# In[12]:


# Membagi data menjadi train dan test dataset

# train dataset
df_train_all = df[0:150000] # potong ke dalam 2 dataset original
# Bagi data menjadi data yang memiliki fraud dan no-fraud
df_train_1 = df_train_all[df_train_all['Class'] == 1]
df_train_0 = df_train_all[df_train_all['Class'] == 0]
print('Dalam dataset, memiliki ' + str(len(df_train_1)) +" frauds")

df_sample=df_train_0.sample(300)
# Gabungkan data fraud dengan no-fraud 
df_train = df_train_1.append(df_sample) 
# Gabungkan seluruh dataset
df_train = df_train.sample(frac=1) 


# In[13]:


X_train = df_train.drop(['Time', 'Class'],axis=1) # We drop the features Time (useless), and the Class (label)
y_train = df_train['Class'] # We create our label
X_train = np.asarray(X_train)
y_train = np.asarray(y_train)


# In[14]:


#test dataset
df_test_all = df[150000:]

X_test_all = df_test_all.drop(['Time', 'Class'],axis=1)
y_test_all = df_test_all['Class']
X_test_all = np.asarray(X_test_all)
y_test_all = np.asarray(y_test_all)


# In[ ]:





# In[15]:


X_train_rank = df_train[df_rank.index[1:11]] 
X_train_rank = np.asarray(X_train_rank)


# In[16]:


X_test_all_rank = df_test_all[df_rank.index[1:11]]
X_test_all_rank = np.asarray(X_test_all_rank)
y_test_all = np.asarray(y_test_all)


# # Confusion Matrix

# In[17]:


#Labeli Class=1 "fraud", 0="no fraud"
class_names=np.array(['0','1'])


# In[18]:


# Fungsi Confusion Matrix
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd' 
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# # Model Selection

# Kami menggunakan SVM Model Classifier dengan scikit-learn library

# In[19]:


#Set SVM default sebagai classifier
classifier = svm.SVC(kernel='linear')
#train model dengan train data yang seimbang
classifier.fit(X_train, y_train)


# # Testing the model

# In[20]:


prediction_SVM_all = classifier.predict(X_test_all)
cm = confusion_matrix(y_test_all, prediction_SVM_all)
plot_confusion_matrix(cm,class_names)


# Hasil prediksi di atas menjelaskan bahwa kesalahan tentang actual fraud jauh lebih buruk daripada kesalahan pada transaksi non-fraud. Sehingga diperlukan akurasi sebagai standar klasifikasi.

# In[21]:


print('Hasil akurasi standar   ' 
      + str( ( (cm[0][0]+cm[1][1]) / (sum(cm[0]) + sum(cm[1])) + 4 * cm[1][1]/(cm[1][0]+cm[1][1])) / 5))


# In[22]:


print('Terdeteksi sebanyak ' + str(cm[1][1]) + ' fraud  ' + str(cm[1][1]+cm[1][0]) + ' dari total fraud.')
print('\nJadi kemungkinan untuk mendeteksi satu fraud adalah ' + str(cm[1][1]/(cm[1][1]+cm[1][0])))
print("Sehingga akurasi yang dihasilkan adalah : "+str((cm[0][0]+cm[1][1]) / (sum(cm[0]) + sum(cm[1]))))


# # Models Rank

# Karena sebelumnya telah dilakukan dimension reduction dari 29 menjadi 10, maka perlu menghitung fit method lagi

# In[23]:


classifier.fit(X_train_rank, y_train)
prediction_SVM = classifier.predict(X_test_all_rank)


# In[24]:


cm = confusion_matrix(y_test_all, prediction_SVM)
plot_confusion_matrix(cm,class_names)


# In[25]:


print('Hasil standar akurasi:  ' 
      + str( ( (cm[0][0]+cm[1][1]) / (sum(cm[0]) + sum(cm[1])) + 4 * cm[1][1]/(cm[1][0]+cm[1][1])) / 5))


# In[ ]:





# In[26]:


print('Terdeteksi sebanyak ' + str(cm[1][1]) + ' fraud  ' + str(cm[1][1]+cm[1][0]) + ' dari total fraud.')
print('\nProbabiliti untuk mendeteksi satu fraud adalah ' + str(cm[1][1]/(cm[1][1]+cm[1][0])))
print("Akurasinya yang dihasilkan : "+str((cm[0][0]+cm[1][1]) / (sum(cm[0]) + sum(cm[1]))))


# Kesimpulannya:
# Hasil data yang diperoleh lebih relevan dengan menggunakan reduction data sehingga langkah terakhir yang dilakukan dari PCA yang pada tahap awal dapat dilakukan dengan cara yang lebih efisien.

# # Re-balanced class weigh :

# In[ ]:





# Pada model SVM sebelumnya, bobot tiap kelas sama sehingga dapat dikatakan bahwa nilai fraud sama buruknya dengan salah menilai no-fraud. Tujuan yang diinginkan oleh pihak Bank adalah untuk memaksimalkan jumlah fraud yang terdeteksi, bahwa perlu juga mempertimbangkan lebih banyak tuple no-fraud sebagai operasi fraud. Sehingga diperlukan tuple untuk meminimalkan Positif Fraud sebagai jumlah fraud yang tidak terdeteksi.
# 
# Dengan memodifikasi parameter class_weight, perlu memilih class mana 
# yang lebih penting pada tahap train data. Karena pada data dengan Class=0 yang lebih besar dan memungkinkan banyaknya kesalahan pada klasifikasi no-fraud sehingga memberikan nilai yang lebih penting. Dengan tujuan untuk menghilangkan fraud sesedikit mungkin

# In[27]:


classifier_b = svm.SVC(kernel='linear',class_weight={0:0.60, 1:0.40})


# In[28]:


#train data
classifier_b.fit(X_train, y_train)


# In[29]:


#test model
prediction_SVM_b_all = classifier_b.predict(X_test_all)
cm = confusion_matrix(y_test_all, prediction_SVM_b_all)
plot_confusion_matrix(cm,class_names)


# In[30]:


print('Hasil standar akurasi:  ' 
      + str( ( (cm[0][0]+cm[1][1]) / (sum(cm[0]) + sum(cm[1])) + 4 * cm[1][1]/(cm[1][0]+cm[1][1])) / 5))


# In[31]:


print('Terdeteksi sebanyak ' + str(cm[1][1]) + ' fraud ' + str(cm[1][1]+cm[1][0]) + ' dari total fraud.')
print('\nProbabiliti untuk mendeteksi satu fraud adalah ' + str(cm[1][1]/(cm[1][1]+cm[1][0])))
print("Akurasi yang dihasilkan adalah : "+str((cm[0][0]+cm[1][1]) / (sum(cm[0]) + sum(cm[1]))))


# In[32]:


#models Rank
classifier_b.fit(X_train_rank, y_train)
prediction_SVM = classifier_b.predict(X_test_all_rank)


# In[33]:


cm = confusion_matrix(y_test_all, prediction_SVM)
plot_confusion_matrix(cm,class_names)


# In[34]:


print('Hasil standar akurasi: ' 
      + str( ( (cm[0][0]+cm[1][1]) / (sum(cm[0]) + sum(cm[1])) + 4 * cm[1][1]/(cm[1][0]+cm[1][1])) / 5))


# In[35]:


print('Terdeteksi sebanyak' + str(cm[1][1]) + ' fraud ' + str(cm[1][1]+cm[1][0]) + ' dari total frauds.')
print('\nJadi Probabiliti untuk mendeteksi satu fraud adalah ' + str(cm[1][1]/(cm[1][1]+cm[1][0])))
print("Akurasi yang dihasilkan adalah : "+str((cm[0][0]+cm[1][1]) / (sum(cm[0]) + sum(cm[1]))))


# In[ ]:




