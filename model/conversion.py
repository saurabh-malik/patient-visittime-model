import os
import numpy as np
import pandas as pd

from string import Template

def convert_to_unstructure(dataframe, text_data_dir, isTraining=True):
	## template
	#unstructured_template_v1 = Template('A 2 months old, $Sex $AnimalBreed-$AnimalClass is checked in today ($Day) in hospital for $AppointmentTypeName. Patient weight is $Weight and reason for this visit are $AppointmentReasons.')
	unstructured_template_v2 = Template('A $Age months old, $Weight lb, $Sex $AnimalBreed-$AnimalClass is checked-in on $Day in hospital for $AppointmentTypeName due to $AppointmentReasons.')

	if isTraining == True:
		text_data_dir+='train/'
	else:
		text_data_dir+='test/'

	for ind in dataframe.index:
		#Ignoring outliers and missing data
		if(dataframe['TotalTimeInWindow-15'][ind] <=30 and dataframe['AgeInMonths'][ind] >0 and dataframe['Weight'][ind] > 0.0):
			a = unstructured_template_v2.substitute(Sex=dataframe['Sex'][ind], AnimalBreed=dataframe['AnimalBreed'][ind], Age=dataframe['AgeInMonths'][ind], AnimalClass=dataframe['AnimalClass'][ind], Day=dataframe['Day'][ind], AppointmentTypeName=dataframe['AppointmentTypeName'][ind], Weight=dataframe['Weight'][ind], AppointmentReasons=dataframe['AppointmentReasons'][ind])
			#file_path = folderName+"\"+str(dataframe['TotalTimeInWindow-15'][ind])+"\"+str(ind)+".txt"
			#check for directory
			class_dir = text_data_dir+str(dataframe['TotalTimeInWindow-15'][ind])
			if(os.path.exists(class_dir) == False):
				os.makedirs(class_dir)
			file_path = class_dir+"/"+str(ind)+".txt"
			with open(file_path, 'w') as fp:
	        	# uncomment if you want empty file
				fp.write(a)

	 

def convert_to_unstructure_for_pretraining(dataframe, data_dir_pretrain):
	## template
	unstructured_template_v2 = Template('A $Age months old, $Weight lb, $Sex $AnimalBreed-$AnimalClass is checked-in on $Day in hospital for $AppointmentTypeName due to $AppointmentReasons.')

	folderName = "data/visits/pretrain"
	
	a = ''
	for ind in dataframe.index:
		#Ignoring outliers and missing data
		if (dataframe['AgeInMonths'][ind] >0 and dataframe['Weight'][ind] > 0.0):
			a += unstructured_template_v2.substitute(Sex=dataframe['Sex'][ind], AnimalBreed=dataframe['AnimalBreed'][ind], Age=dataframe['AgeInMonths'][ind], AnimalClass=dataframe['AnimalClass'][ind], Day=dataframe['Day'][ind], AppointmentTypeName=dataframe['AppointmentTypeName'][ind], Weight=dataframe['Weight'][ind], AppointmentReasons=dataframe['AppointmentReasons'][ind])
			a += "\n\n"
			#file_path = data_dir_pretrain+"\"+str(dataframe['TotalTimeInWindow-15'][ind])+"\"+str(ind)+".txt"
			#check for directory
	if(os.path.exists(data_dir_pretrain) == False):
		os.makedirs(data_dir_pretrain)
	file_path = data_dir_pretrain+"/pre-train-data.txt"
	with open(file_path, 'w') as fp:
		fp.write(a)
