import os
import random
import time
import numpy as np
import csv
import torchvision.models as models

from feature_compare import compare_feature, get_vggnet_conv3_2_features
from hash_compare import aHash, dHash, pHash, compare_hash
from predict import predict_image




def mr_test(label, predict_before, predict_after, value=5):
    # print(abs(np.square(label - predict_after) - (np.square(label - predict_before))))
    # if abs(np.square(label - predict_after) - (np.square(label - predict_before))) <= value:
    if np.square(label - predict_after) <= value * (np.square(label - predict_before)):
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

def art_feature_distance(model_type, originalDataset, noiseDataset, labels, vggnet, model, compare_type='ED', k1=10, k2=5):

    F_measures= 1
    reveal_failure = False

    executed_set = gen_executed_set_distance(originalDataset, vggnet, k1)

    while reveal_failure == False:

        candidate_set = gen_candidate_set_distance(noiseDataset, vggnet, k2)
        best_case = feature_distance_select_best_case(executed_set, candidate_set, compare_type)

        # print("best:",best_case[0])

        predict_after = predict_image(best_case[0], model, model_type)
        predict_before = predict_image(best_case[0].split("_")[-1], model, model_type)
        label = labels[best_case[0].split("_")[-1]]

        # print(predict, type(predict))

        if mr_test(float(label), float(predict_before), float(predict_after)) == False:
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

def art_hash_distance(model_type, originalDataset, noiseDataset, labels, model, hash_type='P', compare_type='None', k1=10, k2=5):

    F_measures= 1
    reveal_failure = False


    executed_set = gen_executed_set_hash(originalDataset, hash_type, k1)

    while reveal_failure == False:

        candidate_set = gen_candidate_set_hash(noiseDataset, hash_type, k2)
        best_case = hash_select_best_case(executed_set, candidate_set, compare_type)

        # print("best:",best_case[0])

        predict_after = predict_image(best_case[0], model, model_type)
        predict_before = predict_image(best_case[0].split("_")[-1], model, model_type)
        label = labels[best_case[0].split("_")[-1]]

        # print(predict, type(predict))

        if mr_test(float(label), float(predict_before), float(predict_after)) == False:
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








