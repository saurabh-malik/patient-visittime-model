import random
import numpy as np
import pandas as pd


def load_data(data_dir, data_file_name):
	#DataLoad & Split
    #Dataload
    file = data_dir + "/"+data_file_name
    dataframe = pd.read_csv(file)
    dataframe.head()

    return dataframe