# -*- coding: utf-8 -*-
"""
@author: Guansong Pang
The algorithm was implemented using Python 3.6.6, Keras 2.2.2 and TensorFlow 1.10.1.
More details can be found in our KDD19 paper.
Guansong Pang, Chunhua Shen, and Anton van den Hengel. 2019. 
Deep Anomaly Detection with Deviation Networks. 
In The 25th ACM SIGKDDConference on Knowledge Discovery and Data Mining (KDD ’19),
August4–8, 2019, Anchorage, AK, USA.ACM, New York, NY, USA, 10 pages. https://doi.org/10.1145/3292500.3330871
"""

import numpy as np
np.random.seed(42)
import tensorflow as tf
tf.set_random_seed(42)
sess = tf.Session()

from keras import regularizers
from keras import backend as K
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, TensorBoard

import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.sparse import vstack, csc_matrix
from utils import dataLoading, aucPerformance, writeResults, get_data_from_svmlight_file
from sklearn.model_selection import train_test_split

import time

MAX_INT = np.iinfo(np.int32).max
data_format = 0

def dev_network_d(input_shape):
    '''
    deeper network architecture with three hidden layers
    '''
    x_input = Input(shape=input_shape)
    intermediate = Dense(1000, activation='relu',
                kernel_regularizer=regularizers.l2(0.01), name = 'hl1')(x_input)
    intermediate = Dense(250, activation='relu',
                kernel_regularizer=regularizers.l2(0.01), name = 'hl2')(intermediate)
    intermediate = Dense(20, activation='relu',
                kernel_regularizer=regularizers.l2(0.01), name = 'hl3')(intermediate)
    intermediate = Dense(1, activation='linear', name = 'score')(intermediate)
    return Model(x_input, intermediate)

def dev_network_s(input_shape):
    '''
    network architecture with one hidden layer
    '''
    x_input = Input(shape=input_shape)
    intermediate = Dense(20, activation='relu', 
                kernel_regularizer=regularizers.l2(0.01), name = 'hl1')(x_input)
    intermediate = Dense(1, activation='linear',  name = 'score')(intermediate)    
    return Model(x_input, intermediate)

def dev_network_linear(input_shape):
    '''
    network architecture with no hidden layer, equivalent to linear mapping from
    raw inputs to anomaly scores
    '''    
    x_input = Input(shape=input_shape)
    intermediate = Dense(1, activation='linear',  name = 'score')(x_input)
    return Model(x_input, intermediate)

def deviation_loss(y_true, y_pred):
    '''
    z-score-based deviation loss
    '''    
    confidence_margin = 5.     
    ## size=5000 is the setting of l in algorithm 1 in the paper
    ref = K.variable(np.random.normal(loc = 0., scale= 1.0, size = 5000) , dtype='float32')
    dev = (y_pred - K.mean(ref)) / K.std(ref)
    inlier_loss = K.abs(dev) 
    outlier_loss = K.abs(K.maximum(confidence_margin - dev, 0.))
    return K.mean((1 - y_true) * inlier_loss + y_true * outlier_loss)


def deviation_network(input_shape, network_depth):
    '''
    构建基于偏差网络的检测模型
    construct the deviation network-based detection model
    '''
    if network_depth == 4:
        model = dev_network_d(input_shape)
    elif network_depth == 2:
        model = dev_network_s(input_shape)
    elif network_depth == 1:
        model = dev_network_linear(input_shape)
    else:
        sys.exit("The network depth is not set properly")
    rms = RMSprop(clipnorm=1.)
    model.compile(loss=deviation_loss, optimizer=rms)
    return model


def batch_generator_sup(x, outlier_indices, inlier_indices, batch_size, nb_batch, rng):
    """batch generator
    """
    rng = np.random.RandomState(rng.randint(MAX_INT, size = 1))
    counter = 0
    while 1:                
        if data_format == 0:
            ref, training_labels = input_batch_generation_sup(x, outlier_indices, inlier_indices, batch_size, rng)
        else:
            ref, training_labels = input_batch_generation_sup_sparse(x, outlier_indices, inlier_indices, batch_size, rng)
        counter += 1
        yield(ref, training_labels)
        if (counter > nb_batch):
            counter = 0
 
def input_batch_generation_sup(x_train, outlier_indices, inlier_indices, batch_size, rng):
    '''
    batchs of samples. This is for csv data.
    Alternates between positive and negative pairs.
    '''      
    dim = x_train.shape[1]
    ref = np.empty((batch_size, dim))    
    training_labels = []
    n_inliers = len(inlier_indices)
    n_outliers = len(outlier_indices)
    for i in range(batch_size):    
        if(i % 2 == 0):
            sid = rng.choice(n_inliers, 1)
            ref[i] = x_train[inlier_indices[sid]]
            training_labels += [0]
        else:
            sid = rng.choice(n_outliers, 1)
            ref[i] = x_train[outlier_indices[sid]]
            training_labels += [1]
    return np.array(ref), np.array(training_labels)

 
def input_batch_generation_sup_sparse(x_train, outlier_indices, inlier_indices, batch_size, rng):
    '''
    batchs of samples. This is for libsvm stored sparse data.
    Alternates between positive and negative pairs.
    '''      
    ref = np.empty((batch_size))    
    training_labels = []
    n_inliers = len(inlier_indices)
    n_outliers = len(outlier_indices)
    for i in range(batch_size):    
        if(i % 2 == 0):
            sid = rng.choice(n_inliers, 1)
            ref[i] = inlier_indices[sid]
            training_labels += [0]
        else:
            sid = rng.choice(n_outliers, 1)
            ref[i] = outlier_indices[sid]
            training_labels += [1]
    ref = x_train[ref, :].toarray()
    return ref, np.array(training_labels)


def load_model_weight_predict(model_name, input_shape, network_depth, x_test):
    '''
    load the saved weights to make predictions
    '''
    model = deviation_network(input_shape, network_depth)
    model.load_weights(model_name)
    scoring_network = Model(inputs=model.input, outputs=model.output)    
    
    if data_format == 0:
        scores = scoring_network.predict(x_test)
    else:
        data_size = x_test.shape[0]
        scores = np.zeros([data_size, 1])
        count = 512
        i = 0
        while i < data_size:
            subset = x_test[i:count].toarray()
            scores[i:count] = scoring_network.predict(subset)
            if i % 1024 == 0:
                print(i)
            i = count
            count += 512
            if count > data_size:
                count = data_size
        assert count == data_size
    return scores


def inject_noise_sparse(seed, n_out, random_seed):  
    '''
    add anomalies to training data to replicate anomaly contaminated data sets.
    we randomly swape 5% features of anomalies to avoid duplicate contaminated anomalies.
    This is for sparse data.
    '''
    rng = np.random.RandomState(random_seed) 
    n_sample, dim = seed.shape
    swap_ratio = 0.05
    n_swap_feat = int(swap_ratio * dim)
    seed = seed.tocsc()
    noise = csc_matrix((n_out, dim))
    print(noise.shape)
    for i in np.arange(n_out):
        outlier_idx = rng.choice(n_sample, 2, replace = False)
        o1 = seed[outlier_idx[0]]
        o2 = seed[outlier_idx[1]]
        swap_feats = rng.choice(dim, n_swap_feat, replace = False)
        noise[i] = o1.copy()
        noise[i, swap_feats] = o2[0, swap_feats]
    return noise.tocsr()

def inject_noise(seed, n_out, random_seed):   
    '''
    从异常数据中取出n_noise数量的数据,并对其中的5%的特征的数据进行随机交换
    add anomalies to training data to replicate anomaly contaminated data sets.
    we randomly swape 5% features of anomalies to avoid duplicate contaminated anomalies.
    this is for dense data
    '''  
    rng = np.random.RandomState(random_seed) 
    # 427,21 n_sample 数量，dim x的参数的数量
    n_sample, dim = seed.shape
    # 随机交换 5% 的异常特征以避免重复污染异常
    swap_ratio = 0.05
    # 交换特征的数量
    n_swap_feat = int(swap_ratio * dim)
    # 噪音的数量列表（280，21）
    noise = np.empty((n_out, dim))

    for i in np.arange(n_out):
        # 从n_sample中随机选出2条数据的index
        outlier_idx = rng.choice(n_sample, 2, replace = False)
        # 取出对应的数据
        o1 = seed[outlier_idx[0]]
        o2 = seed[outlier_idx[1]]
        # 从dim中随机取出5%的异常特征
        swap_feats = rng.choice(dim, n_swap_feat, replace = False)

        # 将从总的异常数据中，取出的一条异常数据，放进返回的噪音数据列表中
        noise[i] = o1.copy()
        # 将这条数据中的5%的特征数值和o2的交换
        noise[i, swap_feats] = o2[swap_feats]

    return noise

def run_devnet(args):
    # 支持同时选择多种数据集，以 逗号 进行分隔
    names = args.data_set.split(',')
    # 指定使用下面的这个数据集
    names = ['annthyroid_21feat_normalised']
    # 从 args 命令行输入的参数中提取参数 network_depth
    network_depth = int(args.network_depth)
    # 从 args 命令行输入的参数中提取参数 ramdn_seed
    random_seed = args.ramdn_seed
    # 选中多种数据集的情况下，对每一个数据集进行模型训练
    for nm in names:
        # 重复实验 runs 次，取平均值
        runs = args.runs
        # 创建一个 runs 次实验的列表，用于存储每次训练的 rauc 值
        rauc = np.zeros(runs)
        # 创建一个 runs 次实验的列表，用于存储每次训练的 ap 值
        # average_precision_score
        ap = np.zeros(runs)  
        # 获取文件名，并去除可能存在的空格字符
        filename = nm.strip()
        # data_format 0为csv格式，1为libsvm格式
        # 设定全局的文件提取格式 
        global data_format
        data_format = int(args.data_format)
        # 读取文件里的数据，获得参数x和标签label
        if data_format == 0:
            x, labels = dataLoading(args.input_path + filename + ".csv")
        else:
            x, labels = get_data_from_svmlight_file(args.input_path + filename + ".svm")
            # 将 x 转化为 压缩稀疏行格式
            x = x.tocsr()    
        # 选取所有labels==1的index
        # labels.shape(7200,) 取[0] 是将tuple的格式转换为ndarray
        outlier_indices = np.where(labels == 1)[0]
        # labels == 1 的 x 参数
        outliers = x[outlier_indices]  
        # labels == 1 的数量 534
        n_outliers_org = outliers.shape[0]   
        
        train_time = 0
        test_time = 0
        # 重复 runs 次训练，取结果的平均值
        for i in np.arange(runs):  
            # 分割训练集和测试集，测试集比例为0.2
            # stratify保持测试集与整个数据集里labels的数据分类比例一致。
            x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=0.2, random_state=42, stratify = labels)
            # 转化数据类型为nparray
            y_train = np.array(y_train)
            y_test = np.array(y_test)
            print(filename + ': round ' + str(i))

            # 获得训练集中的异常样本的index列表，数据类型为numpy.ndarray
            outlier_indices = np.where(y_train == 1)[0]
            # 获得训练集中的正常样本的index列表，数据类型为numpy.ndarray
            inlier_indices = np.where(y_train == 0)[0]
            
            # 获得训练集中的异常样本的数量
            n_outliers = len(outlier_indices)
            print("Original training size: %d, No. outliers: %d" % (x_train.shape[0], n_outliers))
            
            # 噪音的数量 如：污染率为5%，7200*0.8 * 0.05 / 0.95  = 280
            n_noise  = len(np.where(y_train == 0)[0]) * args.cont_rate / (1. - args.cont_rate)
            # 对 n_noise 进行取整操作
            n_noise = int(n_noise)                
            
            # 产生一个随机数队列
            rng = np.random.RandomState(random_seed)  
            if data_format == 0:                

                # 如果训练集中的异常样本的数量 > 开始时指定的异常样本数量
                if n_outliers > args.known_outliers:
                    # 异常样本总数 减去 指定的异常样本数量
                    mn = n_outliers - args.known_outliers
                    # 从异常样本队列中，生成一个随机的样本队列，大小为mn
                    remove_idx = rng.choice(outlier_indices, mn, replace=False)            
                    # 从训练集中删去mn大小的异常数据
                    x_train = np.delete(x_train, remove_idx, axis=0)
                    y_train = np.delete(y_train, remove_idx, axis=0)
                # 上面if里面的几步，使得训练集中只随机留下指定数量的异常样本，来模拟小数量的已知恶意样本

                # 输入的参数为 总的异常数据列表，噪音数量和随机数种子
                # 取出n_noise个数量的异常噪音数据，其中5%的特征值是经过和其他异常数据随机交换的
                noises = inject_noise(outliers, n_noise, random_seed)
                # 将噪音数据加入的训练集中
                x_train = np.append(x_train, noises, axis = 0)
                y_train = np.append(y_train, np.zeros((noises.shape[0], 1)))
            
            else:
                # 如果训练集中的异常样本的数量 > 开始时指定的异常样本数量
                if n_outliers > args.known_outliers:
                    # 异常样本总数 减去 指定的异常样本数量
                    mn = n_outliers - args.known_outliers
                    # remove_idx 从异常样本队列中，生成一个随机的样本队列，大小为mn
                    remove_idx = rng.choice(outlier_indices, mn, replace=False) 
                    # 从训练集中减去 remove_idx
                    retain_idx = set(np.arange(x_train.shape[0])) - set(remove_idx)
                    # 转换数据类型为list
                    retain_idx = list(retain_idx)
                    # 
                    x_train = x_train[retain_idx]
                    y_train = y_train[retain_idx]                               
                
                # 输入的参数为 总的异常数据列表，噪音数量和随机数种子
                # 取出n_noise个数量的异常噪音数据，其中5%的特征值是经过和其他异常数据随机交换的
                noises = inject_noise_sparse(outliers, n_noise, random_seed)

                # 垂直堆叠稀疏矩阵（按行）
                x_train = vstack([x_train, noises])
                # y_train 添加噪音数据
                y_train = np.append(y_train, np.zeros((noises.shape[0], 1)))
            
            # 获取训练集中异常的index列表
            outlier_indices = np.where(y_train == 1)[0]
            # 获取训练集中正常的index列表
            inlier_indices = np.where(y_train == 0)[0]
            # 打印 训练集的数量，训练集中异常的数量，训练集中正常的数量，噪音的数量
            print(y_train.shape[0], outlier_indices.shape[0], inlier_indices.shape[0], n_noise)
            # (21,)，取列
            # shape[:1] 取行 (5760,)
            input_shape = x_train.shape[1:]
            # 训练集中的数量
            n_samples_trn = x_train.shape[0]
            # 异常的数量
            n_outliers = len(outlier_indices)         
            print("Training data size: %d, No. outliers: %d" % (x_train.shape[0], n_outliers))
            

            # 启动时间
            start_time = time.time() 
            # 重复赋值
            # input_shape = x_train.shape[1:]
            epochs = args.epochs
            batch_size = args.batch_size    
            nb_batch = args.nb_batch  
            # 取出模型
            model = deviation_network(input_shape, network_depth)
            print(model.summary())  
            model_name = "./model/devnet_"  + filename + "_" + str(args.cont_rate) + "cr_"  + str(args.batch_size) +"bs_" + str(args.known_outliers) + "ko_" + str(network_depth) +"d.h5"
            checkpointer = ModelCheckpoint(model_name, monitor='loss', verbose=0,
                                           save_best_only = True, save_weights_only = True)            
            
            model.fit_generator(batch_generator_sup(x_train, outlier_indices, inlier_indices, batch_size, nb_batch, rng),
                                          steps_per_epoch = nb_batch,
                                          epochs = epochs,
                                          callbacks=[checkpointer])   
            train_time += time.time() - start_time
            
            start_time = time.time() 
            scores = load_model_weight_predict(model_name, input_shape, network_depth, x_test)
            test_time += time.time() - start_time
            rauc[i], ap[i] = aucPerformance(scores, y_test)     
        
        mean_auc = np.mean(rauc)
        std_auc = np.std(rauc)
        mean_aucpr = np.mean(ap)
        std_aucpr = np.std(ap)
        # 总的训练时间 / 实验重复次数
        train_time = train_time/runs
        # 总的训练时间 / 实验重复次数
        test_time = test_time/runs
        print("average AUC-ROC: %.4f, average AUC-PR: %.4f" % (mean_auc, mean_aucpr))    
        print("average runtime: %.4f seconds" % (train_time + test_time))
        writeResults(filename+'_'+str(network_depth), x.shape[0], x.shape[1], n_samples_trn, n_outliers_org, n_outliers,
                     network_depth, mean_auc, mean_aucpr, std_auc, std_aucpr, train_time, test_time, path=args.output)


      
parser = argparse.ArgumentParser()
parser.add_argument("--network_depth", choices=['1','2', '4'], default='2', help="the depth of the network architecture")
parser.add_argument("--batch_size", type=int, default=512, help="batch size used in SGD")
parser.add_argument("--nb_batch", type=int, default=20, help="the number of batches per epoch")
parser.add_argument("--epochs", type=int, default=50, help="the number of epochs")
parser.add_argument("--runs", type=int, default=10, help="how many times we repeat the experiments to obtain the average performance")
parser.add_argument("--known_outliers", type=int, default=30, help="the number of labeled outliers available at hand")
parser.add_argument("--cont_rate", type=float, default=0.02, help="the outlier contamination rate in the training data")
parser.add_argument("--input_path", type=str, default='./dataset/', help="the path of the data sets")
parser.add_argument("--data_set", type=str, default='annthyroid_21feat_normalised', help="a list of data set names")
parser.add_argument("--data_format", choices=['0','1'], default='0',  help="specify whether the input data is a csv (0) or libsvm (1) data format")
parser.add_argument("--output", type=str, default='./results/devnet_auc_performance_30outliers_0.02contrate_2depth_10runs.csv', help="the output file path")
parser.add_argument("--ramdn_seed", type=int, default=42, help="the random seed number")
args = parser.parse_args()
run_devnet(args)
