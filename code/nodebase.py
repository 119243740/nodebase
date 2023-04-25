#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score,recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import argparse



def embed(name,file,nodenum,MAXDEGREE):

    ###################### 构建出邻接矩阵########################
    adjMatrix=np.zeros((nodenum,nodenum)).astype("int32")
    for i in range(nodenum):
        # 选出和节点 i 相连的节点
        one_nodel=file[file[:,1]==i][:,:1]
        one_noder=file[file[:,0]==i][:,1:2]
        # one_node[:,0] 中是i的邻居节点
        adjMatrix_ind=list(np.vstack([one_nodel,one_noder])[:,0])
        # 将i节点的邻居节点置为1。完成邻接矩阵的构建。
        adjMatrix[i,adjMatrix_ind]=1
    ##############################################################
    ##                 寻找正交基
    ##############################################################
    ### 允许的，节点基的最大度是最大度节点的百分比。
    MINDEGREE=0
    findtime=30000
    embedsize=0
    epich=0
    # 统计每一个节点的度
    d_everynode=np.sum(adjMatrix,axis=1)
    # 去掉度太大的节点
    ch_std=int(np.max(d_everynode)*MAXDEGREE)
    ### 待采样的节点基集合
    ind=np.where((d_everynode <= ch_std)&(d_everynode>MINDEGREE))[0]
    #设置一个标示基节点的列表
    base_number=[]
    # 初始化一个只包含正交基的矩阵
    ch_node=np.random.choice(list(ind),size=1,replace=False,p=None)
    ind=np.delete(ind,np.where(ind==ch_node))

    BASE=adjMatrix[ch_node,:][0,:]
    base_number.append(int(ch_node))
    embedsize=embedsize+1

    while epich < findtime:
        ch_node=np.random.choice(list(ind),size=1,replace=False,p=None)
        #计算加入节点的度
        vect=adjMatrix[ch_node,:][0,:]
        norelation=np.dot(vect,BASE.T)
    #     print("norelation.sum:{}".format(norelation.sum()))
        if(np.sum(norelation)==0):
            ind=np.delete(ind,np.where(ind==ch_node))
            BASE=np.vstack([BASE,vect])
            base_number.append(int(ch_node))
    #         totledegree=totledegree+d_everynode[ch_node]
            embedsize=embedsize+1
        #print("第{0}次嵌入后维度：{1}".format(epich,embedsize))
        epich=epich+1
        if ind.shape[0]==0:
            break    
    ##############################################################
    ##   单位化 && 嵌入
    ##############################################################
    ### 对基向量进行单位化
    fb=BASE.sum(axis=1)
    NORMBASE=[]
    for i in range(BASE.shape[0]):
        NORMBASE.append(BASE[i]/fb[i])

    NORMBASE=np.array(NORMBASE)  
    NORMBASE.shape

    ############ 这就是我们需要的嵌入表征 ##############
    embeded1=np.dot(NORMBASE,adjMatrix.T).T
    embeded1.shape
    ##############################################################
    ##   结合边集后的嵌入
    ##############################################################
    ########### 添加边的信息 ##################
    selfweight=0.5
    embededge=[]
    for i in range(nodenum):
        embededge.append(selfweight*embeded1[i]+(1-selfweight)*embeded1[np.where(adjMatrix[i])[0]].sum(axis=0))

    ##############################################################
    ##     性能测量 + 分类
    ##############################################################
    if name in ["cora","cite","pub"]:    
        data=pd.read_csv("../dataset/{}_node_label.csv".format(name),usecols=['node','label'],dtype="int32")
        data["label"].value_counts(normalize=True)
        Label=data["label"].values

    ################### 分层抽样 方法2 ################
    train_set,test_set,Label_train,Label_test=train_test_split(embededge,Label,test_size=0.1,random_state=42,stratify=data["label"])
    ################### 分类任务 #####################  
    result_micro_f1=[]
    result_macro_f1=[]
    result_accuracy=[]
    result_recall=[]
    for _ in range(3):    
        rnd_clf = RandomForestClassifier(n_estimators= 2000, max_leaf_nodes=430 , n_jobs=-1) 
        rnd_clf.fit(train_set,Label_train)  
    #     rnd_clf = SVC() 
    #     rnd_clf.fit(train_set,Label_train) 

        # cross_val_score(rnd_clf,train_set,Label_train,cv=5,scoring="accuracy")
        y_pred_rf = rnd_clf.predict(test_set)
        result_micro_f1.append(f1_score(Label_test,y_pred_rf,average='micro'))
        result_macro_f1.append(f1_score(Label_test,y_pred_rf,average='macro'))
        result_accuracy.append(accuracy_score(Label_test,y_pred_rf))
        result_recall.append(recall_score(Label_test,y_pred_rf,average='macro'))
    #     print("set:{},macro f1:{},micro f1:{},accuracy:{}\n".format(name,f1_score(Label_test,y_pred_rf,average='macro'),
    #                                                                       f1_score(Label_test,y_pred_rf,average='micro'),
    #                                                                       accuracy_score(Label_test,y_pred_rf)))
    return np.mean(result_macro_f1),np.mean(result_micro_f1),np.mean(result_accuracy),np.mean(result_recall)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n","--name",type=str,default="cora",help="Name of dataset")
    parser.add_argument("-c","--coeffmaxdegree",type=float,default=0.059,help="The coefficient of maxdegree")
    parser.add_argument("-e","--epoch",type=int,default=1,help="the compute times of evalue process")
    args = parser.parse_args()
    

    micro_f1=[]
    macro_f1=[]
    accuracy=[]
    recall=[]
    #############################################################
    # name=sys.argv[1]
    name=args.name
    num={"cora":2708,"cite":3312,"pub":19717}
    read_file=pd.read_csv("../dataset/{}_node_diffdegree.csv".format(name),usecols=['nodeA','nodeB'],dtype="int32")
    file=read_file.values
    nodenum=num[name]
    MAXDEGREE=args.coeffmaxdegree

    for _ in range(args.epoch):
        a1,a2,a3,a4=embed(name,file,nodenum,MAXDEGREE)
        micro_f1.append(a1)
        macro_f1.append(a2)
        accuracy.append(a3)
        recall.append(a4)
        
    mydata={"micro_f1":micro_f1,"macro_f1":macro_f1,"accuracy":accuracy,"recall":recall}
    data=pd.DataFrame(mydata)
    data.to_csv("./result_{}_set_MD_{}.csv".format(name,MAXDEGREE),columns=["micro_f1","macro_f1","accuracy","recall"])
