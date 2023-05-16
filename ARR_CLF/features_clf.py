#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 14:58:35 2022

@author: xuxinzi
"""

from asyncore import write
from contextlib import AsyncExitStack
import numpy as np
from utils import fiducial_points_vary,select_train,select_finetune,cal_statistic, convert_bin,select_train_for_4classes, write_image
import random
from features import Features
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import confusion_matrix
#import graphviz 
from sklearn.tree import export_text
import sklearn.metrics as metrics
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from dataset import Dataset, get_infer_loader, get_test_loader, get_train_loader

from PIL import Image
import config
from trainer import Trainer

data = np.load('train_data_256.npy', allow_pickle = True)
label = np.load('train_label_256.npy', allow_pickle = True)
segments = np.load('train_segments_256.npy', allow_pickle = True)
rr_feas = np.load('train_feas_256.npy', allow_pickle = True)

data = data.item()
label  =label.item()
segments = segments.item()
rr_feas = rr_feas.item()

DS_train = [101, 106, 108, 109, 112, 114, 115, 116, 118,119, 122,124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
DS_test = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]


# DS_test = [100]
# DS_train  = [100]

############################## segment_data ##############################
def segment_data():
    DS = DS_train + DS_test
    points_dirs = {}
    for id in DS:
        
        id = str(id)
        print(id)
        segments_id = segments[id]
        data_id = data[id]
        rr_id = np.int32(rr_feas[id])
        label_id = label[id]
        label_id = label_id.T
        
        # points = fiducial_points(segments_id, data_id,id)

        points, dirs = fiducial_points_vary(segments_id, data_id, id)
        points_dirs_id =  np.concatenate((points,dirs), axis =1)
        points_dirs[id] = points_dirs_id
    np.save('./points_dirs.npy',points_dirs)


#################################### prepare_train_data ####################################
def prepare_train_data():
    train_data = []
    train_label = []
    for id in DS_train:
        id = str(id)
        # print(id)
        segments_id = segments[id]
        data_id = data[id]
        rr_id = np.int32(rr_feas[id])
        label_id = label[id]
        label_id = label_id.T

        # label_id_bin = np.zeros_like(label_id)
        # for j in range(label_id.shape[0]):
        #     if label_id[j] != 0:
        #         label_id_bin[j] = 1
        
        points_dirs = np.load('./points_dirs.npy', allow_pickle = True)
        points_dirs = points_dirs.item()
        points_dirs_train = points_dirs[id]
        # for index in range(data_id.shape[0]):
            # plot_with_points(data_id[index,:,0], segments_id[index], points[index], index,id)
        FeatureExtraction = Features(data_id,points_dirs_train, rr_id, label_id)
        features = FeatureExtraction.feature_map()
        label_duiqi = FeatureExtraction.duiqi_label()
        # print(features.shape)
        # print(label_duiqi.shape)
        train_data.append(features)
        train_label.append(label_duiqi)

    train_data = np.concatenate(train_data, axis = 0)
    train_label = np.concatenate(train_label, axis = 0)

############### add icen data #############################
    # print('segment_icen',segment_icen.shape)
    # print('data_icen',data_icen.shape)
    # points_icen, dirs_icen = fiducial_points_vary(segment_icen, data_icen, 0)
    # points_dirs_icen =  np.concatenate((points_icen,dirs_icen), axis =1)
    # np.save('points_dirs_icen.npy', points_dirs_icen)
    
    points_dirs_icen = np.load('points_dirs_icen.npy',allow_pickle=True)
    
    FeatureExtraction = Features(data_id, points_dirs_icen, rrs_icen, label_icen)
    features_icen = FeatureExtraction.feature_map()
    label_duiqi_icen = FeatureExtraction.duiqi_label()
    train_data = np.concatenate((train_data,features_icen), axis = 0)
    train_label = np.concatenate((train_label, label_duiqi_icen), axis = 0)

    train_data, train_label = select_train(train_data, train_label, mode='up')
    return train_data, train_label


################################## prepare_test #####################
def prepare_test_data():
    test_data = []
    test_label = []

    for id in DS_test:
        
        id = str(id)
        # print(id)
        segments_id = segments[id]
        data_id = data[id]
        label_id = label[id]
        label_id = label_id.T

        label_id_bin = np.zeros_like(label_id)
        # for j in range(label_id.shape[0]):
        #     if label_id[j] != 0:
        #         label_id_bin[j] = 1

        rr_id = np.int32(rr_feas[id])
        
        points_dirs = np.load('./points_dirs.npy', allow_pickle = True)
        points_dirs = points_dirs.item()
        points_dirs_test = points_dirs[id]
            
        # for index in range(data_id.shape[0]):
            # plot_with_points(data_id[index,:,0], segments_id[index], points[index], index,id)    
        FeatureExtraction = Features(data_id, points_dirs_test, rr_id, label_id)
        features = FeatureExtraction.feature_map()
        label_duiqi = FeatureExtraction.duiqi_label()
        test_data.append(features)
        test_label.append(label_duiqi)
    test_data = np.concatenate(test_data, axis = 0)
    test_label = np.concatenate(test_label, axis = 0)
    return test_data, test_label


###################################### pytorch_train ############################################################
def Trainer_2stage():
    train_data = np.load('train_data_img.npy')
    train_label = np.load('train_label_img.npy')
    test_data = np.load('test_data_img.npy')
    test_label = np.load('test_label_img.npy') 

    train_data_loader = get_train_loader(batch_size = config.batch_size, train_data = train_data,train_label = train_label)
    test_data_loader = get_test_loader(batch_size = config.batch_size, test_data = test_data, test_label = test_label)
    data_loader = (train_data_loader, test_data_loader)

    trainer = Trainer(data_loader)
    trainer.train()

###################################### keras_train ############################################################
# from tensorflow.keras.datasets import mnist
# from tensorflow.keras import models
# from tensorflow.keras import layers
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.models import load_model
# import matplotlib.pyplot as plt
# import numpy as np
# def data_preprocess():
#     train_images = np.load('train_data_img.npy')
#     train_labels = np.load('train_label_img.npy')
#     test_images = np.load('test_data_img.npy')
#     test_labels = np.load('test_label_img.npy') 

#     train_images = np.expand_dims(train_images,axis = 3 )
#     train_images = train_images.astype('float32')
#     print(train_images[0])
#     test_images = np.expand_dims(test_images,axis = 3 )
#     test_images = test_images.astype('float32')

#     train_labels = to_categorical(train_labels)
#     test_labels = to_categorical(test_labels)
#     return train_images,train_labels,test_images,test_labels

# #搭建网络
# def build_module():
#     model = models.Sequential()
#     model.add(layers.Conv2D(8, (9,9), activation='relu', input_shape=(256,256,1)))
#     model.add(layers.MaxPooling2D((2,2)))
#     model.add(layers.Conv2D(16, (7,7), activation='relu'))
#     model.add(layers.MaxPooling2D((2,2)))
#     model.add(layers.Conv2D(32, (5,5), activation='relu'))
#     model.add(layers.MaxPooling2D((2,2)))
#     model.add(layers.Conv2D(64, (3,3), activation='relu'))
#     model.add(layers.Flatten())
#     model.add(layers.Dropout(0.4))
#     model.add(layers.Dense(64, activation='relu'))
#     model.add(layers.Dropout(0.2))
#     model.add(layers.Dense(4, activation='softmax'))
#     return model

# def draw_loss(history):
#     loss=history.history['loss']
#     epochs=range(1,len(loss)+1)
#     plt.subplot(1,2,1)#第一张图
#     plt.plot(epochs,loss,'bo',label='Training loss')
#     plt.title("Training loss")
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()

#     plt.subplot(1,2,2)#第二张图
#     accuracy=history.history['accuracy']
#     plt.plot(epochs,accuracy,'bo',label='Training accuracy')
#     plt.title("Training accuracy")
#     plt.xlabel('Epochs')
#     plt.ylabel('Accuracy')
#     plt.suptitle("Train data")
#     plt.legend()
#     plt.show()

# def Trainer_2stage():
#     train_images,train_labels,test_images,test_labels=data_preprocess()
#     model=build_module()
#     print(model.summary())
#     model.compile(optimizer='rmsprop', loss = 'categorical_crossentropy', metrics=['accuracy'])
#     history=model.fit(train_images, train_labels, epochs = 30, batch_size=64)
#     draw_loss(history)
#     model.save('keras_model.h5') 

#     model = load_model('keras_model.h5')
#     test_loss, test_acc = model.evaluate(test_images, test_labels)
#     print('test_loss=',test_loss,'  test_acc = ', test_acc)


def Test_four_classes_imgs():
    with open('train_mit_model.pkl', 'rb') as f:
        clf = pickle.load(f)
    predictes_all = []
    test_label_all = []
    for id in DS_test:
        id = str(id)
        print(id)
        segments_id = segments[id]
        data_id = data[id]
        label_id = label[id]
        label_id = np.int32(label_id.T)
        rr_id = np.int32(rr_feas[id])
        
        points_dirs = np.load('./points_dirs.npy', allow_pickle = True)
        points_dirs = points_dirs.item()
        points_dirs_test = points_dirs[id]
            
        # for index in range(data_id.shape[0]):
            # plot_with_points(data_id[index,:,0], segments_id[index], points[index], index,id)    
        FeatureExtraction = Features(data_id, points_dirs_test, rr_id, label_id)
        features = FeatureExtraction.feature_map()
        label_duiqi = FeatureExtraction.duiqi_label()
        data_id_duiqi = FeatureExtraction.duiqi_data()
        y_predicted = clf.predict(features)

        

        for i in range(y_predicted.shape[0]):
            if y_predicted[i] == 1:
                ecg_img = convert_bin(np.expand_dims(data_id_duiqi[i],axis = 0))
                data_loader = get_infer_loader(batch_size=config.batch_size, infer_data = ecg_img)
                trainer = Trainer(data_loader)
                predict = trainer.infer()
                y_predicted[i] = predict + 1
        
        confusion_matrix_id = confusion_matrix(label_duiqi, y_predicted)
        print(confusion_matrix_id)
        
        predictes_all.append(y_predicted)
        test_label_all.append(label_duiqi)

    predictes_all = np.concatenate(predictes_all, axis= 0)
    test_label_all = np.concatenate(test_label_all, axis = 0)
    confusion_matrix_test = confusion_matrix(test_label_all, predictes_all)

    print(confusion_matrix_test)
    pre_i, rec_i, F1_i = cal_statistic(confusion_matrix_test)
    print('rec_i is : {rec_i}'.format(rec_i = rec_i))



def test_with_plot():
    with open('train_mit_model.pkl', 'rb') as f:
        clf = pickle.load(f)
    for id in DS_test:
        
        id = str(id)
        print(id)
        segments_id = segments[id]
        data_id = data[id]
        label_id = label[id]
        label_id = label_id.T
        rr_id = np.int32(rr_feas[id])
        
        points_dirs = np.load('./points_dirs.npy', allow_pickle = True)
        points_dirs = points_dirs.item()
        points_dirs_test = points_dirs[id]
            
        # for index in range(data_id.shape[0]):
            # plot_with_points(data_id[index,:,0], segments_id[index], points[index], index,id)    
        FeatureExtraction = Features(data_id, points_dirs_test, rr_id, label_id)
        features = FeatureExtraction.feature_map()
        label_duiqi = FeatureExtraction.duiqi_label()
        y_predicted=clf.predict(features)
        confusion_matrix_id = confusion_matrix(label_duiqi, y_predicted)
        print(confusion_matrix_id)

        # for i in range(y_predicted.shape[0]):
        #     if label_duiqi[i] != 0 and y_predicted[i] != label_duiqi[i]:
        #         plot_with_points(data_id[i,:,0], segments_id[i], points_dirs_test[i,0:-2],id,i,label_duiqi[i], y_predicted[i])                    

######################################### Trainer 1 stage (sklearn) ##################################################
# segment_data()
def Trainer_1_stage():
    train_data, train_label = prepare_train_data()
    test_data, test_label = prepare_test_data()

    # clf = DecisionTreeClassifier(random_state=0, max_depth=20)

    # clf = RandomForestClassifier(n_estimators=10, criterion='entropy',
                                # max_depth=20, min_samples_split=10, min_samples_leaf=5, min_weight_fraction_leaf=0.0, 
                                # max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0,
                                # min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, 
                                # class_weight = {0: 1, 1:20, 2:10, 3:10, 4:10}, ccp_alpha=0.0, max_samples=None)

    # clf = make_pipeline(StandardScaler(),LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight= {0: 1, 1:20, 2:10, 3:10, 4:10}, random_state=None, solver='liblinear', max_iter=1000, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None))

    # clf = make_pipeline(StandardScaler(), GaussianNB())
    # clf = make_pipeline(StandardScaler(), SVC(C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight={0: 1, 1:20, 2:10, 3:10, 4:10}, verbose=False, max_iter=- 1, decision_function_shape='ovr', break_ties=False, random_state=None))


    clf = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000))


    clf = clf.fit(train_data, train_label)
    # with open('train_mit_model.pkl', 'wb') as f:
    #     pickle.dump(clf, f) 

    #####################################Result######################################################
    acc_train=clf.score(train_data, train_label)
    # print('file_index:',file_index)
    print('训练集准确率：',acc_train)
    # print(confusion_matrix(train_label, clf.predict(train_data)).ravel())
    confusion_matrix_train = confusion_matrix(train_label, clf.predict(train_data))
    print(confusion_matrix_train)
    
    


    y_predicted=clf.predict(test_data)


    acc_test=metrics.accuracy_score(test_label, y_predicted)
    print ('测试集准确率:', metrics.accuracy_score(test_label, y_predicted))
    confusion_matrix_test = confusion_matrix(test_label, y_predicted)

    print(confusion_matrix_test)
    pre_i, rec_i, F1_i = cal_statistic(confusion_matrix_test)
    # print('pre_i is : {pre_i}'.format(pre_i=pre_i))
    print('rec_i is : {rec_i}'.format(rec_i = rec_i))
    # print('F1_i is : {F1_i}'.format(F1_i=F1_i))

    # test_with_plot()

def Trainer_global():
    train_data, train_label = prepare_train_data()
    test_data, test_label = prepare_test_data()

    # transfer = StandardScaler()
    # train_data = transfer.fit_transform(train_data)
    # test_data = transfer.transform(test_data)

    train_data_loader = get_train_loader(batch_size = config.batch_size, train_data = train_data,train_label = np.squeeze(train_label))
    test_data_loader = get_test_loader(batch_size = config.batch_size, test_data = test_data, test_label = np.squeeze(test_label))
    data_loader = (train_data_loader, test_data_loader)

    trainer = Trainer(data_loader)
    # trainer.train() 
    trainer.qat() 

def Test():
    preds = []
    test_labels = []  
    for id in DS_test:
        
        id = str(id)
        print(id)
        segments_id = segments[id]
        data_id = data[id]
        label_id = label[id]
        label_id = label_id.T
        rr_id = np.int32(rr_feas[id])
        
        points_dirs = np.load('./points_dirs.npy', allow_pickle = True)
        points_dirs = points_dirs.item()
        points_dirs_test = points_dirs[id]
            
        # for index in range(data_id.shape[0]):
            # plot_with_points(data_id[index,:,0], segments_id[index], points[index], index,id)    
        FeatureExtraction = Features(data_id, points_dirs_test, rr_id, label_id)
        features = FeatureExtraction.feature_map()
        label_duiqi = FeatureExtraction.duiqi_label()

        data_loader = get_infer_loader(batch_size=config.batch_size, infer_data = features)
        trainer = Trainer(data_loader)
        y_predicted = trainer.infer(is_finetune = False)
        confusion_matrix_id = confusion_matrix(label_duiqi, y_predicted)
        print(confusion_matrix_id)


        preds.append(y_predicted)

        label_duiqi = np.expand_dims(label_duiqi, axis = 1)
        test_labels.append(label_duiqi)

    preds = np.concatenate(preds, axis = 0)
    test_labels = np.concatenate(test_labels, axis = 0)
    test_labels = np.squeeze(test_labels)      

    acc_test=metrics.accuracy_score(test_labels, preds)
    print ('测试集准确率:', metrics.accuracy_score(test_labels, preds))
    confusion_matrix_test = confusion_matrix(test_labels, preds)

    print(confusion_matrix_test)
    pre_i, rec_i, F1_i = cal_statistic(confusion_matrix_test)
    # print('pre_i is : {pre_i}'.format(pre_i=pre_i))
    print('rec_i is : {rec_i}'.format(rec_i = rec_i))

def Finetune():
    known_num = 400

    preds = []
    test_labels = []  

    for id in DS_test:
        id = str(id)
        print(id)
        segments_id = segments[id]
        data_id = data[id]
        label_id = label[id]
        label_id = label_id.T
        rr_id = np.int32(rr_feas[id])
        
        points_dirs = np.load('./points_dirs.npy', allow_pickle = True)
        points_dirs = points_dirs.item()
        points_dirs_test = points_dirs[id]
            
        # for index in range(data_id.shape[0]):
            # plot_with_points(data_id[index,:,0], segments_id[index], points[index], index,id)    
        FeatureExtraction = Features(data_id, points_dirs_test, rr_id, label_id)
        features = FeatureExtraction.feature_map()
        label_duiqi = FeatureExtraction.duiqi_label()


        finetuned_data, finetuned_label, only_mormal = select_finetune(features[0:known_num,:], label_duiqi[0:known_num,:])
        if not only_mormal:
            train_data_loader = get_train_loader(batch_size = config.batch_size, train_data = finetuned_data,train_label = np.squeeze(finetuned_label))
            test_data_loader = get_test_loader(batch_size = config.batch_size, test_data = features, test_label = np.squeeze(label_duiqi))
            trainer = Trainer((train_data_loader, test_data_loader))
            trainer.finetune()

        data_loader = get_infer_loader(batch_size=config.batch_size, infer_data = features)
        trainer = Trainer(data_loader)
        y_predicted = trainer.infer(is_finetune = True)
        confusion_matrix_id = confusion_matrix(label_duiqi, y_predicted)
        print(confusion_matrix_id)


        preds.append(y_predicted)

        label_duiqi = np.expand_dims(label_duiqi, axis = 1)
        test_labels.append(label_duiqi)

    preds = np.concatenate(preds, axis = 0)
    test_labels = np.concatenate(test_labels, axis = 0)
    test_labels = np.squeeze(test_labels)      

    acc_test=metrics.accuracy_score(test_labels, preds)
    print ('测试集准确率:', metrics.accuracy_score(test_labels, preds))
    confusion_matrix_test = confusion_matrix(test_labels, preds)

    print(confusion_matrix_test)
    pre_i, rec_i, F1_i = cal_statistic(confusion_matrix_test)
    print('pre_i is : {pre_i}'.format(pre_i=pre_i))

    print('rec_i is : {rec_i}'.format(rec_i = rec_i))    

def main():
    # Trainer_global()
    Test()

    # Finetune()
    # prepare_train_imgs()
    # prepare_test_imgs()
    # Trainer_2stage()
    # Test_four_classes_imgs()



if __name__ == "__main__":
  main()