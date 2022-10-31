#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 12:42:42 2019

@author: Mojtaba Jafaritadi
"""

from itertools import groupby
import pandas as pd
import csv, json, time, base64, os, sys, time
import numpy as np
from array import array
from preprocessor import create_algorithm_input

dirpath = os.getcwd()
fullpath = os.path.join(dirpath,"json_file_check")

def load_json_from_dict(fullpath):
    fname=os.listdir(fullpath) 
    dict_of_data = {}
    for item in fname:
        with open(os.path.join(fullpath,item), 'r') as data_file:
            dict_of_data[item] = json.load(data_file) 
            del item
                    
        dict_tmp=dict_of_data[fname[0]]
            
        
    user_id_all=[]
    for i in range(len(dict_tmp)):
        user_id_all.append(dict_tmp[i]['user_id'])
        
    my_dict = {i:user_id_all.count(i) for i in user_id_all}
#            
    rep_user=[len(list(group)) for key, group in groupby(user_id_all)]
        
        
    user_id_ext=dict()
    key_ids=dict()
    for key in my_dict.keys():
        for i in range(len(rep_user)):
            string = "_"
            user_id_ext[i]=[string+str(i) for i in range(rep_user[i])]
            key_ids[key]=key
            
    new_key_ids = {i: str(key_ids[k]) for i, k in enumerate(sorted(key_ids.keys()))}
  
    new_id_users=dict()  
    for key,value in user_id_ext.items():
      new_id_users[key]= ["{}{}".format(new_key_ids[key],i) for i in value]
          
    list_of_key_names = [y for x in new_id_users.values() for y in x]
    
    dict_of_MEMS_data = {}
    for i in range(len(dict_tmp)):
        MEMS_dict = dict()
        accTS = array('Q') #unsigned long long, 8 bytes
        accTS.frombytes(base64.b64decode(dict_tmp[i]['accel_time']))
        gyrTS = array('Q') #unsigned long long, 8 bytes
        gyrTS.frombytes(base64.b64decode(dict_tmp[i]['gyro_time']))
        accX = array('f') #unsigned long long, 8 bytes
        accX.frombytes(base64.b64decode(dict_tmp[i]['accel_x']))
        accY = array('f') #unsigned long long, 8 bytes
        accY.frombytes(base64.b64decode(dict_tmp[i]['accel_y']))
        accZ = array('f') #unsigned long long, 8 bytes
        accZ.frombytes(base64.b64decode(dict_tmp[i]['accel_z']))
        accelerometer_data_raw = np.array((accX, accY, accZ, accTS)).T
        gyrX = array('f') #unsigned long long, 8 bytes
        gyrX.frombytes(base64.b64decode(dict_tmp[i]['gyro_x']))
        gyrY = array('f') #unsigned long long, 8 bytes
        gyrY.frombytes(base64.b64decode(dict_tmp[i]['gyro_y']))
        gyrZ = array('f') #unsigned long long, 8 bytes
        gyrZ.frombytes(base64.b64decode(dict_tmp[i]['gyro_z']))
        gyroscope_data_raw = np.array((gyrX, gyrY, gyrZ, gyrTS)).T
        user_id=dict_tmp[i]['user_id']
    
        
        MEMS_dict['acc'] = np.transpose(accelerometer_data_raw)
        MEMS_dict['gyro'] = np.transpose(gyroscope_data_raw)
        dict_of_MEMS_data[list_of_key_names[i]] = MEMS_dict
#                list_of_key_names = list(dict_of_data.keys()) 
    for key in dict_of_MEMS_data.keys(): 
        data = dict_of_MEMS_data[key]
        dataset[key] = create_algorithm_input(data[0], data[1])
    print('Loaded json' + fullpath)           
    return keys, dataset
                    

                

#dict_of_json_files,list_of_key_names=load_json_from_dict(fullpath)