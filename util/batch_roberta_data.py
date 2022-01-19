import random
import click
from transformers import AutoModelForSequenceClassification, AutoTokenizer, MBartForConditionalGeneration, MBartTokenizer
import torch
import csv
import numpy as np
import torch.nn as nn

def noise_sanity_check(cand_arr):
    # decide noise type upon function called
    noise_type = random.choices([1, 2, 3], weights=(1, 1, 1), k=1)[0]
    start_index = random.choices(range(cand_arr.shape[0]), k=1)[0]
    # this is the addition noise
    if noise_type == 1:
        # check if noise position and span length fits current noise context
        if cand_arr[start_index] > 0:
            return noise_type, start_index, 1
    # this is the delete noise
    elif noise_type == 2:
        num_deletes = random.choices([1, 2, 3, 4], weights=(0.50, 0.30, 0.10, 0.10), k=1)[0]
        # check if noise position and span length fits current noise context
        if cand_arr[start_index] >= num_deletes and cand_arr[start_index] != 0:
            return noise_type, start_index, num_deletes
    else:
        num_replace = random.choices([1, 2, 3, 4, 5, 6], weights=(0.50, 0.25, 0.10, 0.05, 0.05, 0.05), k=1)[0]
        # check if noise position and span length fits current noise context
        if cand_arr[start_index] >= num_replace and cand_arr[start_index] != 0:
            return noise_type, start_index, num_replace
    return -1, -1, -1
