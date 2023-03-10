# -*- coding: utf-8 -*-
"""


@author: Ammar.Abasi
"""

import pickle
import tensorflow
import numpy as np
import os
from scipy.interpolate import splev, splrep
import pandas as pd 
import time
from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers
from Hyperactive.hyperactive import RandomSearchOptimizer, MHoneyBadgerAlgorithm
import threading
import concurrent.futures
from threading import Thread
from keras.models import Input, Model




base_dir = "dataset"

# normalize
scaler = lambda arr: (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

ir = 3 # interpolate interval
before = 2
after = 2


def load_data():
    tm = np.arange(0, (before + 1 + after) * 60, step=1 / float(ir)) 

    with open(os.path.join(base_dir, "apnea-ecg.pkl"), 'rb') as f: # read preprocessing result
        apnea_ecg = pickle.load(f)

    x_train = []
    o_train, y_train = apnea_ecg["o_train"], apnea_ecg["y_train"]
    groups_train = apnea_ecg["groups_train"]
    for i in range(len(o_train)):  #for i in range(len(o_train)): for i in range(10)
        (rri_tm, rri_signal), (ampl_tm, ampl_siganl) = o_train[i]
		# Curve interpolation
        rri_interp_signal = splev(tm, splrep(rri_tm, scaler(rri_signal), k=3), ext=1) 
        ampl_interp_signal = splev(tm, splrep(ampl_tm, scaler(ampl_siganl), k=3), ext=1)
        x_train.append([rri_interp_signal, ampl_interp_signal])
    x_train = np.array(x_train, dtype="float32").transpose((0, 2, 1)) # convert to numpy format
    y_train = np.array(y_train, dtype="float32")

    x_test = []
    o_test, y_test = apnea_ecg["o_test"], apnea_ecg["y_test"]
    groups_test = apnea_ecg["groups_test"]
    for i in range(len(o_test)): #for i in range(len(o_test)): for i in range(10)
        (rri_tm, rri_signal), (ampl_tm, ampl_siganl) = o_test[i]
		# Curve interpolation
        rri_interp_signal = splev(tm, splrep(rri_tm, scaler(rri_signal), k=3), ext=1)
        ampl_interp_signal = splev(tm, splrep(ampl_tm, scaler(ampl_siganl), k=3), ext=1)
        x_test.append([rri_interp_signal, ampl_interp_signal])
    x_test = np.array(x_test, dtype="float32").transpose((0, 2, 1))
    y_test = np.array(y_test, dtype="float32")

    return x_train, y_train, groups_train, x_test, y_test, groups_test

x_train, y_train, groups_train, x_test, y_test, groups_test = load_data()

y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes=2) # Convert to two categories
y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes=2)
    
sgd = optimizers.SGD(lr=0.01)
adam = optimizers.Adam(lr=0.01)
inputs = Input(shape=x_train.shape[1:])

search_config = {
  "keras.compile.0": {"loss": ["categorical_crossentropy"], "optimizer": [adam, sgd]},
  "keras.fit.0": {"epochs":[100], "batch_size": range(10, 101), "verbose": [2]},
  "keras.layers.Conv1D.1": {
      "filters": [32],
      "kernel_size": [3, 5, 7],
      "strides":[2],
      "padding":["valid"],
      "kernel_initializer":["he_normal"],
      "activation": ["sigmoid", "relu", "tanh"],      
      "input_shape": [x_train.shape[1:]],
  },
  "keras.layers.MaxPooling1D.2": {"pool_size": [3]},
  "keras.layers.Conv1D.3": {
      "filters": [64],
      "kernel_size": [3, 5, 7],
      "strides":[2],
      "padding":["valid"],
      "kernel_initializer":["he_normal"], 
      "activation": ["sigmoid", "relu", "tanh"],
  },
  "keras.layers.MaxPooling1D.4": {"pool_size": [3]},  
  "keras.layers.Dropout.5":{"rate":[.8]},   
  #"keras.layers.Dropout.5":{"rate":np.arange (0.5, 0.8, 0.1)},  
  "keras.layers.Flatten.6": {},
  "keras.layers.Dense.7": {"units":[32], "activation": ["sigmoid", "relu", "tanh"]},
  #"keras.layers.Dense.7": {"units": range(4, 201), "activation": ["sigmoid", "relu", "tanh"]},
  "keras.layers.Dense.8": {"units": [2], "activation": ["softmax"]},
}


def thread_function(k):

  run=k+1
  print("Run Number",run)
  Optimizer = MHoneyBadgerAlgorithm(search_config, n_iter=100, n_part=10, metric='accuracy', cv=10, h_beta=6.0, h_c=2.0,run=run)
   t1 = time.time()
  Optimizer.fit(x_train, y_train)
  t2 = time.time()
  
  print("time: {}".format(t2-t1)) 
  
  # predict from test data
  Optimizer.predict(x_test)
  score = Optimizer.score(x_test, y_test)
  
  output = pd.DataFrame({"run":[run],"Accuracy":[score],"time":[t2-t1]})
  output.to_csv(os.path.join("output", "mhba_apnea_best_all_runs.csv"), mode='a', index=False,header=False)
  
  print("test score: {}".format(score))
NumOfRuns=1   
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor: 
                    executor.map(thread_function, range(NumOfRuns))
  
    
    
    
    