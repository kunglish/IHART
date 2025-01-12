import os
import random
import time
import numpy as np
import csv
import torchvision.models as models

from feature_compare import compare_feature, get_vggnet_conv3_2_features
from hash_compare import aHash, dHash, pHash, compare_hash
from predict import predict_image


import torch
from model.resnet32 import ResNet32
from model.resnet50 import ResNet50



def getLabel(filename):

    last_underscore_index = filename.rfind('_')

    last_dot_index = filename.rfind('.')
    if last_underscore_index != -1 and last_dot_index != -1:
        return filename[last_underscore_index + 1:last_dot_index]
    else:
        print("file error!")
    return None


def mr_test(filename, predict):
    label = getLabel(filename)
    # print("label: {}, {}; predict: {}, {}".format(label, type(label), predict, type(predict)))
    if str(label) == str(predict):
        return True
    else:
        return False


########################################################################################################################

def gen_executed_set_distance(dataset, vggnet, k1):
    executed_set = {}
    executedDataset= random.sample(dataset, k1)
    for dataPath in executedDataset:
        executed_data = get_vggnet_conv3_2_features(dataPath, vggnet)
        executed_set[dataPath] = executed_data

    # print(executed_set)
    return executed_set


def gen_candidate_set_distance(dataset, vggnet, k2):
    candidate_set = {}
    executedDataset= random.sample(dataset, k2)
    for dataPath in executedDataset:
        executed_data = get_vggnet_conv3_2_features(dataPath, vggnet)
        candidate_set[dataPath] = executed_data

    return candidate_set



########################################################################################################################

def gen_executed_set_hash(dataset, hashType, k1):
    executed_set = {}
    executedDataset = random.sample(dataset, k1)
    if hashType == 'A':
        for dataPath in executedDataset:
            executed_data = aHash(dataPath)
            executed_set[dataPath] = executed_data
    if hashType == 'D':
        for dataPath in executedDataset:
            executed_data = dHash(dataPath)
            executed_set[dataPath] = executed_data
    if hashType == 'P':
        for dataPath in executedDataset:
            executed_data = pHash(dataPath)
            executed_set[dataPath] = executed_data

    # print(executed_set)
    return executed_set


def gen_candidate_set_hash(dataset, hashType, k2):
    candidate_set = {}
    executedDataset= random.sample(dataset, k2)
    if hashType == 'A':
        for dataPath in executedDataset:
            executed_data = aHash(dataPath)
            candidate_set[dataPath] = executed_data
    if hashType == 'D':
        for dataPath in executedDataset:
            executed_data = dHash(dataPath)
            candidate_set[dataPath] = executed_data
    if hashType == 'P':
        for dataPath in executedDataset:
            executed_data = pHash(dataPath)
            candidate_set[dataPath] = executed_data

    return candidate_set


########################################################################################################################
#feature_distance
def feature_distance_select_best_case(executed_set, candidate_set, compare_type='ED'):
    best_distance = -1.0

    for key_candidate_set, value_candidate_set in candidate_set.items():
        distance = 0
        for key_executed_set, value_executed_set in executed_set.items():
            distance = (distance + compare_feature(value_executed_set, value_candidate_set, compare_type))
        distance = distance / len(executed_set)
        if distance > best_distance:
            best_distance = distance
            best_case = (key_candidate_set, value_candidate_set)  

    # print(best_case)
    return best_case

def art_feature_distance(originalDataset, noiseDataset, vggnet, model, compare_type='ED', k1=10, k2=5):

    F_measures= 1
    reveal_failure = False

    executed_set = gen_executed_set_distance(originalDataset, vggnet, k1)

    while reveal_failure == False:

        candidate_set = gen_candidate_set_distance(noiseDataset, vggnet, k2)
        best_case = feature_distance_select_best_case(executed_set, candidate_set, compare_type)

        # print("best:",best_case[0])

        predict = predict_image(model, best_case[0])
        if predict == 0:
            predict = 10

        # print(predict, type(predict))

        if mr_test(best_case[0], predict) == False:
            reveal_failure = True
            # print("Failure is", str(best_case), "F-measures is", str(F_measures))
        else:
            # ("This test data did't find any failure, F-measure is", str(F_measures))
            F_measures += 1
            # if F_measures % 50 == 0:
            #     print(F_measures)
            noiseDataset.remove(best_case[0])
            executed_set[best_case[0]] = best_case[1]

    return F_measures

########################################################################################################################
#Hash
def hash_select_best_case(executed_set, candidate_set, compare_type='None'):
    best_distance = -1.0

    for key_candidate_set, value_candidate_set in candidate_set.items():
        distance = 0
        for key_executed_set, value_executed_set in executed_set.items():
            distance = (distance + compare_hash(value_executed_set, value_candidate_set, compare_type))
        distance = distance / len(executed_set)
        if distance > best_distance:
            best_distance = distance
            best_case = (key_candidate_set, value_candidate_set)

    # print(best_case)
    return best_case

def art_hash_distance(originalDataset, noiseDataset, model, hash_type='P', compare_type='None', k1=10, k2=5):

    F_measures= 1
    reveal_failure = False


    executed_set = gen_executed_set_hash(originalDataset, hash_type, k1)

    while reveal_failure == False:

        candidate_set = gen_candidate_set_hash(noiseDataset, hash_type, k2)
        best_case = hash_select_best_case(executed_set, candidate_set, compare_type)

        # print("best:",best_case[0])

        predict = predict_image(model, best_case[0])
        if predict == 0:
            predict = 10

        # print(predict, type(predict))


        if mr_test(best_case[0], predict) == False:
            reveal_failure = True
            # print("Failure is", str(best_case), "F-measures is", str(F_measures))
        else:
            # ("This test data did't find any failure, F-measure is", str(F_measures))
            F_measures += 1
            # if F_measures % 50 == 0:
            #     print(F_measures)
            noiseDataset.remove(best_case[0])
            executed_set[best_case[0]] = best_case[1]

    return F_measures







