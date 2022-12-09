import os
import numpy as np
import pandas as pd

from string import Template

def convert_to_unstructure(dataframe, isTraining=True):
	## template
	data = []
	y = []
	#unstructured_template_v1 = Template('A 2 months old, $Sex $AnimalBreed-$AnimalClass is checked in today ($Day) in hospital for $AppointmentTypeName. Patient weight is $Weight and reason for this visit are $AppointmentReasons.')
	unstructured_template_v2 = Template('A $Age months old, $Weight lb, $Sex $AnimalBreed-$AnimalClass is checked-in on $Day in hospital for $AppointmentTypeName due to $AppointmentReasons.')

	folderName = "data/visits/"
	if isTraining == True:
		folderName+='train/'
	else:
		folderName+='test/'

	for ind in dataframe.index:
		#Ignoring outliers and missing data
		if(dataframe['TotalTimeInWindow-15'][ind] <=30 and dataframe['AgeInMonths'][ind] >0 and dataframe['Weight'][ind] > 0.0):
			a = unstructured_template_v2.substitute(Sex=dataframe['Sex'][ind], AnimalBreed=dataframe['AnimalBreed'][ind], Age=dataframe['AgeInMonths'][ind], AnimalClass=dataframe['AnimalClass'][ind], Day=dataframe['Day'][ind], AppointmentTypeName=dataframe['AppointmentTypeName'][ind], Weight=dataframe['Weight'][ind], AppointmentReasons=dataframe['AppointmentReasons'][ind])
			#file_path = folderName+"\"+str(dataframe['TotalTimeInWindow-15'][ind])+"\"+str(ind)+".txt"
			#check for directory
			classfolder = folderName+str(dataframe['TotalTimeInWindow-15'][ind])
			if(os.path.exists(classfolder) == False):
				os.makedirs(classfolder)
			file_path = classfolder+"/"+str(ind)+".txt"
			with open(file_path, 'w') as fp:
	        	# uncomment if you want empty file
				fp.write(a)
			#data.append(a)	
			#y.append(dataframe['TotalTimeInWindow-15'][ind])

	return data, y

def convert_to_unstructure_for_pretraining(dataframe):
	## template
	data = []
	y = []
	#unstructured_template_v1 = Template('A 2 months old, $Sex $AnimalBreed-$AnimalClass is checked in today ($Day) in hospital for $AppointmentTypeName. Patient weight is $Weight and reason for this visit are $AppointmentReasons.')
	unstructured_template_v2 = Template('A $Age months old, $Weight lb, $Sex $AnimalBreed-$AnimalClass is checked-in on $Day in hospital for $AppointmentTypeName due to $AppointmentReasons.')

	folderName = "data/visits/pretrain"
	
	a = ''
	for ind in dataframe.index:
		#Ignoring outliers and missing data
		if (dataframe['AgeInMonths'][ind] >0 and dataframe['Weight'][ind] > 0.0):
			a += unstructured_template_v2.substitute(Sex=dataframe['Sex'][ind], AnimalBreed=dataframe['AnimalBreed'][ind], Age=dataframe['AgeInMonths'][ind], AnimalClass=dataframe['AnimalClass'][ind], Day=dataframe['Day'][ind], AppointmentTypeName=dataframe['AppointmentTypeName'][ind], Weight=dataframe['Weight'][ind], AppointmentReasons=dataframe['AppointmentReasons'][ind])
			a += "\n\n"
			#file_path = folderName+"\"+str(dataframe['TotalTimeInWindow-15'][ind])+"\"+str(ind)+".txt"
			#check for directory
	if(os.path.exists(folderName) == False):
		os.makedirs(folderName)
	file_path = folderName+"/pre-train-data.txt"
	with open(file_path, 'w') as fp:
		fp.write(a)

	return data, y