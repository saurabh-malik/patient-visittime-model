import tensorflow as tf
from model.encoder import get_category_encoding_layer
from model.encoder import get_normalization_layer


def encode_feature(numerical_features, categorical_features, dataset):
	"""Function generating the encoded features
	"""
	encoded_features = []
	all_inputs = []
	#Numerical Feature
	for header in numerical_features:
		numeric_col = tf.keras.Input(shape=(1,), name=header)
		normalization_layer = get_normalization_layer(header, dataset)
		encoded_numeric_col = normalization_layer(numeric_col)
		all_inputs.append(numeric_col)
		encoded_features.append(encoded_numeric_col)
	
	#Categorical Feature
	for header in categorical_features:
		isFreeText = False
		if header=='AppointmentReasons':
			isFreeText = True
		categorical_col = tf.keras.Input(shape=(1,), name=header, dtype='string')
		encoding_layer = get_category_encoding_layer(name=header,
                                               dataset=dataset,
                                               dtype='string',
                                               isFreeText=isFreeText,
                                               max_tokens=250)
		encoded_categorical_col = encoding_layer(categorical_col)
		all_inputs.append(categorical_col)
		encoded_features.append(encoded_categorical_col)
	

	return all_inputs, encoded_features
	
