import numpy as np
import pandas as pd

from string import Template

def convert_to_unstructure(dataframe):
	## template
	data = []
	y = []
	unstructured_template = Template('A 2 months old, $Sex $AnimalBreed-$AnimalClass is checked in today ($Day) in hospital for $AppointmentTypeName. Patient weight is $Weight and reason for this visit are $AppointmentReasons.')

	for ind in dataframe.index:
		a = unstructured_template.substitute(Sex=dataframe['Sex'][ind], AnimalBreed=dataframe['AnimalBreed'][ind], AnimalClass=dataframe['AnimalClass'][ind], Day=dataframe['Day'][ind], AppointmentTypeName=dataframe['AppointmentTypeName'][ind], Weight=dataframe['Weight'][ind], AppointmentReasons=dataframe['AppointmentReasons'][ind])
		data.append(a)	
		y.append(dataframe['TotalTimeInWindow-15'][ind])

	return data, y