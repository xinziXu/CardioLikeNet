#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 14:58:35 2022

@author: xuxinzi
"""

from asyncore import write
import numpy as np
from utils import fiducial_points_vary,select_train,select_finetune,cal_statistic
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
######################################### Trainer with finetune and quantization (pytorch) ##################################################
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
    Trainer_global()
    Test()

    # Finetune()





if __name__ == "__main__":
  main()