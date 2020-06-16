'''*Upload the Training and Testing Dataset xlsx and run all*'''


import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.metrics import confusion_matrix

column_names = ['ID', 'Age', 'Gender','TB', 'DB', 'ALK', 'SGPT', 'SGOT', 'TP', 'ALB', 'AG_Ratio', 'Class']
 
train_data = pd.read_excel("IS4003_SCS4104_CS4104_dataset.xlsx",
                         sheet_name = "Training Dataset",
                         header = 0,
                         names = column_names,
                         na_filters = True,
                         na_values = "?")

 
test_data = pd.read_excel("IS4003_SCS4104_CS4104_dataset.xlsx",
                         sheet_name = "Testing Dataset",
                         header = 0,
                         names = column_names,
                         na_filters = True,
                         na_values = "?")

#Preprocessing

train_data.pop('ID')
#assigning numerical values to nominals
train_data['Gender'] = pd.Categorical(train_data.Gender).codes.astype(np.int32)
train_data['Class'] = pd.Categorical(train_data.Class).codes.astype(np.int32)
train_y = train_data.pop('Class')
 
test_data.pop('ID')
#assigning numerical values to nominals
test_data['Gender'] = pd.Categorical(test_data.Gender).codes.astype(np.int32)
test_data['Class'] = pd.Categorical(test_data.Class).codes.astype(np.int32)
test_y = test_data.pop('Class')

#missing value handling
 
train_data = train_data.fillna(train_data.mean())
test_data = test_data.fillna(test_data.mean())

#Dataframe to dataset conversion
def df_conv(attributes, classes, training=True, batch_size=len(train_data)):
        dataset = tf.data.Dataset.from_tensor_slices((dict(attributes), classes))
 
        if training:
          dataset = dataset.shuffle(1000).repeat()
 
        return dataset.batch(batch_size)

feat_columns = []
for k in train_data.keys():
  feat_columns.append(tf.feature_column.numeric_column(key=k))

# Classifier, training and evaluation code

# Boosted Trees Classifier
# Accuracy 1 / 0.993 / 0.996

classifier = tf.estimator.BoostedTreesClassifier(
    feat_columns,
    n_batches_per_layer=1
    )

classifier.train(
    input_fn=lambda:df_conv(
        attributes = train_data, 
        classes = train_y), 
      max_steps=5000
  )
result = classifier.evaluate(
    input_fn=lambda: df_conv(
      test_data,
      test_y,
      training=False
    )
)
print('\n', pd.Series(result))

"""**Confusion Matrix for the Test Dataset**"""

#retrieval of predictions for each test data & confusion matrix

predictions = classifier.predict (
    input_fn = lambda: df_conv(
        test_data,
        test_y,
        training = False
    )
)

predictions_y = []

for p in list(predictions):
  predictions_y.append(p['class_ids'][0])

confusion_matrix = confusion_matrix(test_y, predictions_y)
cm_df = pd.DataFrame(
    confusion_matrix,
    columns = np.unique(test_y),
    index = np.unique(test_y)
)

cm_df.index.name = 'True Class'
cm_df.columns.name = 'Predicted'


plt.figure(figsize = (10,8))
sb.set(font_scale=1.5)
figure = sb.heatmap(
    cm_df,
    annot = True,
    annot_kws = {'size': 16}
)
figure.set_title("Confusion Matrix")

plt.show()

"""**Classifier Performance Measures based on the Test Dataset**"""

#Classifier performance indicators

TP = confusion_matrix[1][1]
TN = confusion_matrix[0][0]
FP = confusion_matrix[0][1]
FN = confusion_matrix[1][0]

accuracy = (TP + TN)/ (TP + FP + TN + FN)
precision = TP/ (TP + FP)
sensitivity = TP/ (TP + FN)
specificity = TN/ (TN + FP)
error_rate =  (FP + FN)/ (TP + FP + TN + FN)

print ('\n')
print("Accuracy \t= {:.2f}\n".format(accuracy))
print("Precision \t= {:.2f}\n".format(precision))
print("Sensitivity \t= {:.2f}\n".format(sensitivity))
print("Specificity \t= {:.2f}\n".format(specificity))
print("Error Rate \t= {:.2f}\n".format(error_rate))

"""Additional: Classifier comparisons"""

# #LinearClassifier
# #Accuracy 0.7138

# classifier = tf.estimator.LinearClassifier(
#     feature_columns= feat_columns)

# classifier.train(
#     input_fn = lambda: df_conv(
#         attributes = train_data, 
#         classes = train_y),
    
#     max_steps = 5000
# )

# result = classifier.evaluate(
#     input_fn=lambda: df_conv(test_data, test_y, training=False))

# print('\n',pd.Series(result))



# #DNNRegressor

# regressor = tf.estimator.DNNRegressor(
#     feature_columns= feat_columns,
#     hidden_units=[11, 10, 9])

# regressor.train(
#     input_fn=lambda:df_conv(
#         attributes = train_data, 
#         classes = train_y), 
#       max_steps=5000
#   )

# result = regressor.evaluate(
#     input_fn=lambda: df_conv(test_data, test_y, training=False))

# print('\n',pd.Series(result))



# # Boosted Trees Classifier
# # Accuracy 1 / 0.993
# classifier = tf.estimator.BoostedTreesClassifier(
#     feat_columns,
#     n_batches_per_layer=1
#     )

# classifier.train(
#     input_fn=lambda:df_conv(
#         attributes = train_data, 
#         classes = train_y), 
#       max_steps=5000
#   )
# result = classifier.evaluate(input_fn=lambda: df_conv(test_data, test_y, training=False))
# print('\n', pd.Series(result))



# # DNN Classifier
# # Accuracy 0.62

# classifier = tf.estimator.DNNClassifier(
#     feature_columns=feat_columns,
#     hidden_units=[11, 10, 9],
#     n_classes=2)
 
# classifier.train(
#     input_fn = lambda: df_conv(
#         attributes = train_data, 
#         classes = train_y),
    
#     max_steps = 5000
# )
 
# result = classifier.evaluate(
#     input_fn=lambda: df_conv(test_data, test_y, training=False))

# print('\n',pd.Series(result))
