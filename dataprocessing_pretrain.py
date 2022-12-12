import argparse
import logging
import os
import numpy as np
import pandas as pd
import tensorflow as tf

from model.utils import Params
from model.utils import set_logger
from model.model_fn import model_fn
from model.data_processing import load_data
from model.conversion import convert_to_unstructure_for_pretraining

parser = argparse.ArgumentParser()
parser.add_argument('--data_version', default='V2',
                    help="Data version of patient visit CVS file")
parser.add_argument('--data_dir', default='data/visits',
                    help="Directory containing the dataset")

# Load the parameters from json file
args = parser.parse_args()
    
#Dataload
datafile = 'visitdataclassification_'+args.data_version+'.csv'
dataframe = load_data(args.data_dir, datafile)
data_dir_pretrain = args.data_dir+'/pretrain'
if __name__ == '__main__':
	#Create text file from Visit data
	convert_to_unstructure_for_pretraining(dataframe, data_dir_pretrain)