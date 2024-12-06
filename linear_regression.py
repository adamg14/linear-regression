import pandas as pd
import tensorflow as tf

# Load the dataset
df_train = pd.read_csv('../train.csv')
df_eval = pd.read_csv('../eval.csv')

# remove the label (output) from the features (input)
feature_train = df_train.pop('survived')
feature_eval = df_eval.pop('survived')

# separate the catgorical columns and the numeric columns
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 
                       'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

# data preprocessing - converting the categorical data into encoded format
def preprocess_categorical(data, columns):
    return pd.get_dummies(data, columns=columns)


# encoding the categorical columns into binary
df_train = preprocess_categorical(df_train, CATEGORICAL_COLUMNS)
df_eval = preprocess_categorical(df_eval, CATEGORICAL_COLUMNS)

# fill in the missing columns due to different categories (say there is not one boarding from southampton in the testing set, so there will be no boardSouthampton (true/false) columns)
# align the columns of both datasets and fill missing columns with zeros
df_train, df_eval = df_train.align(df_eval, join='left', axis=1, fill_value=0)

# convert the dataframes into tensors - generalised multidimension matrices
# each dimension is a column
train_features = tf.convert_to_tensor(df_train.values, dtype=tf.float32)
train_labels = tf.convert_to_tensor(feature_train, dtype=tf.float32)
eval_features = tf.convert_to_tensor(df_eval.values, dtype=tf.float32)
eval_labels = tf.convert_to_tensor(feature_eval.values, dtype=tf.float32)

print(train_features)

# create dataset pipeline
# defining we are training the model to go from the features (input) to the label (output)
# both training and evaluation datasets batched
train_dataset = tf.data.Dataset.from_tensor_slices((train_features, train_labels)).batch(32)
eval_dataset = tf.data.Dataset.from_tensor_slices((eval_features, eval_labels)).batch(32)

# define a sequential model. Dense layer for immediate processing, A Dense layer with sigmoid activation for binary classification of survival (true or false)
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(df_train.shape[1],)),
    # sigmoid for binary classification
    tf.keras.layers.Dense(1, activation='sigmoid')  
])


# the model is compiled with an optimiser, using accuracy as the metric
optimiser = tf.keras.optimizers.Adagrad(learning_rate=0.05)
model.compile(optimizer=optimiser, loss="binary_crossentropy", metrics=['accuracy'])

# training the model using the training dataset, epoch = 10 so that the model is trained with the whole dataset 10 times
model.fit(train_dataset, epochs=10)

# evaluate the results
evaluation_results = model.evaluate(eval_dataset)
print("evaluation results: ")
print(evaluation_results)