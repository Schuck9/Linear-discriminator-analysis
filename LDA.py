"""
A simple implementation of Fisher linear classifier/Linear discriminator analysis
@data: 2019.12.18
@author: Tingyu Mo
"""
import pandas as pd
import numpy as np
import os
import math
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score,accuracy_score

class LDA():
    '''
    A simple implementation of Fisher linear classifier/Linear discriminator analysis
    '''
    def __init__(self):
        self.w0 = 0
        self.w_opt = 0

    def cal_means(self,x,axis=0):
        return x.mean(axis=axis)
    
    def cal_inter_variance_matrix(self,x,mean,axis=0):
        '''
        Si = sum((xi-mi)*(xi-mi)T)
        ''' 
        return (x-mean).T.dot((x-mean))

    def cal_inv_matrix(self,x):
        return np.linalg.inv(x)

    def cal_optim_weight(self,Sw_inv,m1,m2):
        '''
        W^*= S_W^(-1) (m_1-m_2)
        '''
        w_opt = Sw_inv.dot((m1-m2).T)
        self.w_opt = w_opt
        return w_opt
    
    def cal_mapping_means(self,w_opt,mean):
        '''
        mi* = w_opt^T * mean
        '''
        return mean.dot(w_opt)
    
    def cal_decision_threshold(self,m1,m2):
        '''
        w0 = - m1+m2/2
        '''
        w0 = - (m1 + m2)*1.0/2
        self.w0 = w0
        return w0
    
    def predict(self,x_test):
        w_opt = self.w_opt
        w0 = self.w0
        y_pred= list()
        for x in x_test:
            y = x.dot(w_opt)+w0
            if(y>=0):
                y_pred.append(1)
            if(y<0):
                y_pred.append(-1)
        # print(y_pred)
        return np.array(y_pred)

    def plot_decision_boundary(self,X,y):
        # 设定最大最小值，附加一点点边缘填充
        x1_min, x1_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        x2_min, x2_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        h = 0.01
        
        x1_grid, x2_grid = np.meshgrid(np.arange(x1_min, x1_max, h), np.arange(x2_min, x2_max, h))
        
        # 用预测函数预测一下
        Z = self.predict(np.c_[x1_grid.ravel(), x2_grid.ravel()])
        Z = Z.reshape(x1_grid.shape)
        
        # 然后画出图
        plt.title("Fisher linear classifier")
        plt.contourf(x1_grid, x2_grid, Z, cmap=plt.cm.Spectral)
        plt.scatter(X[:, 0], X[:, 1], c=y,marker='x',cmap=plt.cm.Spectral)

    def viz(self, data_one,data_two):
        plt.rcParams['font.sans-serif']=['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.title("样本特征空间分布图")
        plt.scatter(data_one[:,0], data_one[:,1],marker='+',color='b',label='w1',s=47)
        plt.scatter(data_two[:,0], data_two[:,1],marker='o',color='r',label='w2',s=47)
        plt.legend(loc = 'upper right')
        plt.show()
        # self.plot_decision_boundary(X,y)
        # plt.show()
        

        
if __name__=="__main__":
    w1_features = np.array([[0.2331, 2.3385], [1.5207, 2.1946], [0.6499, 1.6730], [0.7757, 1.6365],
                [1.0524, 1.7844], [1.1974, 2.0155], [0.2908, 2.0681], [0.2518, 2.1213],
                [0.6682, 2.4797], [0.5622, 1.5118], [0.9023, 1.9692], [0.1333, 1.8340],
                [-0.5431, 1.8704], [0.9407, 2.2948], [-0.2126, 1.7714], [0.0507, 2.3939],
                [-0.0810, 1.5648], [0.7315, 1.9329], [0.3345, 2.2027], [1.0650, 2.4568],
                [-0.0247, 1.7523], [0.1043, 1.6991], [0.3122, 2.4883], [0.6655, 1.7259],
                [0.5838, 2.0466], [1.1653, 2.0226], [1.2653, 2.3757], [0.8137, 1.7987],
                [-0.3399, 2.0828], [0.5152, 2.0798], [0.7226, 1.9449], [-0.2015, 2.3801],
                [0.4070, 2.2373], [-0.1717, 2.1614], [-1.0573, 1.9235], [-0.2099, 2.2604]])
    w2_features = np.array([[1.4010, 1.0298], [1.2301, 0.9611], [2.0814, 0.9154], [1.1655, 1.4901],
                [1.3740, 0.8200], [1.1829, 0.9399], [1.7632, 1.1405], [1.9739, 1.0678],
                [2.4152, 0.8050], [2.5890, 1.2889], [2.8472, 1.4601], [1.9539, 1.4334],
                [1.2500, 0.7091], [1.2864, 1.2942], [1.2614, 1.3744], [2.0071, 0.9387],
                [2.1831, 1.2266], [1.7909, 1.1833], [1.3322, 0.8798], [1.1466, 0.5592],
                [1.7087, 0.5150], [1.5920, 0.9983], [2.9353, 0.9120], [1.4664, 0.7126],
                [2.9313, 1.2833], [1.8349, 1.1029], [1.8340, 1.2680], [2.5096, 0.7140],
                [2.7198, 1.2446], [2.3148, 1.3392], [2.0353, 1.1808], [2.6030, 0.5503],
                [1.2327, 1.4708], [2.1465, 1.1435], [1.5673, 0.7679], [2.9414, 1.1288]])
    w1_labels = np.ones(w1_features.shape[0]) # 1 为w1类的label
    w2_labels = -1*np.ones(w2_features.shape[0])# -1 为w2类的label
    Fisher = LDA()

    #计算样本均值mi
    w1_mean = Fisher.cal_means(w1_features) #shape = (1,n)
    w2_mean = Fisher.cal_means(w2_features)
    #计算类内离散程度矩阵Si
    S1 = Fisher.cal_inter_variance_matrix(w1_features,w1_mean)
    S2 = Fisher.cal_inter_variance_matrix(w2_features,w2_mean)
    #求总类内离散程度矩阵Sw
    Sw = S1+S2
    #求总类内离散程度矩阵Sw的逆矩阵Sw^(-1)
    Sw_inv = Fisher.cal_inv_matrix(Sw)
    #求最优投影方向权向量w*
    w_opt = Fisher.cal_optim_weight(Sw_inv,w1_mean,w2_mean)
    #将样本均值mi投影到投影方向上得到mi^
    w1_mean_map = Fisher.cal_mapping_means(w_opt,w1_mean)
    w2_mean_map = Fisher.cal_mapping_means(w_opt,w2_mean)
    #求分解阈值点w0
    w0 = Fisher.cal_decision_threshold(w1_mean_map,w2_mean_map)
    #待判别样本分类
    x_test = np.array([[1, 1.5], [1.2, 1.0], [2.0, 0.9], [1.2, 1.5], [0.23, 2.33]])
    # x_test = np.array([[1.0524, 1.7844], [1.1974, 2.0155], [0.2908, 2.0681], [0.2518, 2.1213]]) #正类
    # x_test = np.array([[1.3740, 0.8200], [1.1829, 0.9399], [1.7632, 1.1405], [1.9739, 1.0678]]) #负类
    y_pred = list()
    for x in x_test:
        y = x.dot(w_opt)+w0
        if(y>=0):
            y_pred.append(1)
        if(y<0):
            y_pred.append(-1)
    print(y_pred)
    
    Fisher.viz(np.c_[w1_features,w1_labels],np.c_[w2_features,w2_labels] )

    


