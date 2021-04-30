"""
==============================================
Name    : Eashan Adhikarla
Subject : Media Forensics
Project : Mini-project-2 (Task 2)
Data    : April 10, 2021
==============================================

"""
# --- Sklearn ---
from sklearn.datasets import fetch_openml
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn import decomposition, discriminant_analysis, linear_model, svm, tree, neural_network
from sklearn.model_selection import GridSearchCV

# --- Models ---
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn import neural_network
from sklearn.model_selection import StratifiedShuffleSplit

# --- Utility ---
import os, cv2, glob
import pickle
import numpy as np, pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from prepareData import AzimuthalAverage

# rootdir = "/Users/eashan22/Dropbox (LU Student)/Macbook/Desktop/Media Forensics/mini-project-2/Task 2/"
rootdir = "/data/MediaForensics/DeepFake/Frequency/"

path    = ['Faces-HQ/thispersondoesntexists_10K',
           'Faces-HQ/100KFake_10K',
           'Faces-HQ/Flickr-Faces-HQ_10K',
           'Faces-HQ/celebA-HQ_10K',
           ]

datadir = "./data/FacesHQ_Data.pkl"
# datadir = "./data/Data.pkl"

labels = [1, 1, 0, 0]
epsilon = 1e-8
Data = {}
classes = ('Real', 'Fake')

# ======================================================================
# # Number of samples from each dataset
# stop = 625
# z, iter_ = 0, 0

# number_iter = 4 * stop
# Azimuthalavg1D = np.zeros([number_iter, 722])
# label_total = np.zeros([number_iter])


# for data in range(len(path)):
#     dataIdxCount = 0
#     psd1D_average_org = np.zeros(722)
#     print(f"Processing dataset {path[data]}...")


# for z in range(len(path)):
#     cont = 0
#     psd1D_average_org = np.zeros(722)
    
#     for filename in glob.glob(str(rootdir)+path[data]+"/*.jpg"):  
#         # print(filename)
#         img = cv2.imread(filename,0)
        
#         f = np.fft.fft2(img)
#         fshift = np.fft.fftshift(f)
#         fshift += epsilon
        
#         magnitude_spectrum = 20*np.log(np.abs(fshift))

#         # Calculate the azimuthally averaged 1D power spectrum
#         psd1D = AzimuthalAverage(magnitude_spectrum)
#         Azimuthalavg1D[iter_,:] = psd1D
#         label_total[iter_] = labels[z]

#         cont += 1
#         iter_ += 1
#         if cont >= stop:
#             break

# Data["data"], Data["label"] = Azimuthalavg1D, label_total
# print(len(label_total))

# output = open(datadir, 'wb')
# pickle.dump(Data, output)
# output.close()
# print("Data Preprocessed and Saved")

# ======================================================================

# read python dict back from the file
pkl_file = open(datadir, 'rb')
data = pickle.load(pkl_file)

pkl_file.close()
images = data["data"]
labels = data["label"]
print(f"Total images: {len(images)}, Total labels: {len(labels)}")

# Random Split
# X_train, X_test, Y_train, Y_test = train_test_split(images, labels, test_size = 0.2)

# Stratified Split
stratSplit = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
stratSplit.get_n_splits(images, labels)

for train_index, test_index in stratSplit.split(images, labels):
    X_train, X_test = images[train_index], images[test_index]
    Y_train, Y_test = labels[train_index], labels[test_index]
    
print("\nTraining data shape :", X_train.shape, Y_train.shape)
print("Training data shape :", X_test.shape, Y_test.shape)

print("Train label=0 :", np.count_nonzero(Y_train==0))
print("Train label=1 :", np.count_nonzero(Y_train==1))
print("Test label=0 :",  np.count_nonzero(Y_test==0))
print("Test label=1 :",  np.count_nonzero(Y_test==1))

# ======================================================================

def train_and_tune(X, y, model, parameters, scoring='f1_macro', kfold=5, verbose=0):
    """
    X:          array-like of shape (n_samples, n_features)
    y:          array-like of shape (n_samples,)
    model:      (object) a sklearn model class
    parameters: (dict) contains the parameters you want to tune in the model
    metric:     (str) the metric used to evaluate the quality of the model
    return:     a trained model with the best parameters
    """
    cvSearchObj = GridSearchCV(model,
                               parameters,
                               scoring=scoring,
                               n_jobs=-1,
                               cv=kfold,
                               verbose=verbose)
    cvSearchObj.fit(X,y)
    return cvSearchObj.best_estimator_


def save_model(filename, model):
    """
    filename: Filename to save the model
    model:    Model weights to be saved
    """
    pickle.dump(model, open(filename, 'wb'))
    print("Model Saved")

    
def load_model(filename):
    """
    filename: Filename to load the model
    return:   Model weights that are reloaded
    """
    model_reloaded = pickle.load(open(filename, 'rb'))
    return model_reloaded


def SupportVectorMachine(train, save, test):
    filename = "./checkpoint/svcBest.pt"
    svc = svm.SVC(random_state=999)
    if train:
        '''
        Train
        '''
        params = {"kernel":('linear', 'rbf'), 
                "C":[1, 10, 500, 1000]
                }

        svcBest = train_and_tune(X_train,
                                 Y_train,
                                 svc,
                                 params,
                                 scoring='f1_macro',
                                 kfold=5)
        if save:
            save_model(filename, svcBest)
    if test:
        '''
        Test
        '''
        svcBest_reloaded = load_model(filename)
        pred = svcBest_reloaded.predict(X_test)
        acc  = svcBest_reloaded.score(X_test, Y_test)
        
        # cf_matrix = confusion_matrix(Y_test, pred)
        # df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix)*10, index = [i for i in classes],
        #                      columns = [i for i in classes])
        # plt.figure(figsize = (12,10))
        # sn.heatmap(df_cm, annot=True)
        
        # print("Accuracy: ", 100*acc)
    print("Method-SVC completed!")
    return 100*acc


def MultiLayerPerceptron(train, save, test):
    filename = "./checkpoint/mlpBest.pt"
    mlp = neural_network.MLPClassifier(random_state=999)
    if train:
        '''
        Train
        '''
        params = {
                   "alpha" : [0.0001],
                   "learning_rate_init" : [0.005],
                   "batch_size" : [8, 32, 64, 128],
                   "activation" : ["relu"],
                   "early_stopping" : [True],
                   "hidden_layer_sizes" : [3, 10, 50, 100],
                 }

        mlpBest = train_and_tune(X_train,
                                Y_train,
                                mlp,
                                params,
                                scoring='f1_macro',
                                kfold=5)

        if save:
            save_model(filename, mlpBest)

    if test:
        '''
        Test
        '''
        mlpBest_reloaded = load_model(filename)
        pred = mlpBest_reloaded.predict(X_test)
        acc  = mlpBest_reloaded.score(X_test, Y_test)
        
        # cf_matrix = confusion_matrix(Y_test, pred)
        # df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [i for i in classes],
        #                      columns = [i for i in classes])
        # plt.figure(figsize = (12,10))
        # sn.heatmap(df_cm, annot=True)
        
        # print("Accuracy: ", 100*acc)
    print("Method MLP completed!")
    return 100*acc


def KNearestNeighbors(train, save, test):
    filename = "./checkpoint/knnclassifierBest.pkl"
    knnclassifier = KNeighborsClassifier()
    if train:
        '''
        Train
        '''
        params = {"n_neighbors": [2, 6, 10],
                }

        knnclassifierBest = train_and_tune(X_train,
                                Y_train,
                                knnclassifier,
                                params,
                                scoring='f1_macro',
                                kfold=5)

        if save:
            save_model(filename, knnclassifierBest)

    if test:
        '''
        Test
        '''
        knnclassifierBest_reloaded = load_model(filename)
        pred = knnclassifierBest_reloaded.predict(X_test)
        acc  = knnclassifierBest_reloaded.score(X_test, Y_test)
        
        # cf_matrix = confusion_matrix(Y_test, pred)
        # df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [i for i in classes],
        #                      columns = [i for i in classes])
        # plt.figure(figsize = (12,10))
        # sn.heatmap(df_cm, annot=True)
        
        # print("Accuracy: ", 100*acc)
    print("Method-KNN completed!")
    return 100*acc

svm_acc = SupportVectorMachine(train=False, save=False, test=True)
mlp_acc = MultiLayerPerceptron(train=False, save=False, test=True)
knn_acc = KNearestNeighbors(train=False, save=False, test=True)

print("")
print("="*25)
print("MODEL ARCH.\t ACCURACY")
print("-"*25)
print("SVM\t\t  ", svm_acc)
print("-"*25)
print("MLP\t\t   ", mlp_acc)
print("-"*25)
print("KNN\t\t  ", knn_acc)
print("="*25)

# ======================================================================
































