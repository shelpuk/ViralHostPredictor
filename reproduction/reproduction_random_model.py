# This code illustrates data leakage mechanism introduced by the inference mechanism suggested in
# chapter "Study-wide accuracies" of the Supplementary Materials for "Predicting reservoir hosts
# and arthropod vectors from evolutionary signatures in RNA virus genomes" by Simon A. Babayan,
# Richard J. Orton, Daniel G. Streicker, published 2 November 2018, Science 362, 577 (2018)
# DOI: 10.1126/science.aap9072

import lightgbm as lgb
import numpy as np
import scipy
from scipy import stats
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

class RandomClassifier(object):
    def __init__(self, num_classes):
        """
        This class implements random classifier. For every data point, it assigns a random class label.
        The classifier assigns this label consistently. I.e. if you request prediction for the same data point
        several times, the predicted class is going to be the same.
        :param num_classes: number of classes to use for the classifier.
        """
        self.prediction_dictionary = {}
        self.num_classes = num_classes

    def add_data(self, data):
        """
        This simulates training of a regular classifier such as logistic regression or XGBooost
        although this classifier does not learn any model. Instead, it memorizes the data and assigns
        to each of the data points random class taken uniformly at random from the range [0, num_classes - 1]
        :param data: numpy array of m rows and n columns where each row is a training example.
        """
        for i in range(data.shape[0]):
            training_example = data[i]
            self.prediction_dictionary[hash(training_example.tostring())] = np.random.randint(0, self.num_classes - 1)

    def predict(self, test_data):
        """
        This simulates prediction. The model takes every testing data point, examines memorized predictions
        for each of them and returns memorized random class.
        :param test_data: numpy array of m rows and n columns where each row is a testing example.
        :return: numpy array of size m each element of which is a predicted class for the corresponding
        testing data point.
        """
        predictions = np.zeros(test_data.shape[0])
        for i in range(test_data.shape[0]):
            testing_example = test_data[i]
            if hash(testing_example.tostring()) not in self.prediction_dictionary:
                raise ValueError('The testing example is not in self.prediction_dictionary. This random classifier requires all data points to be introdused using add_data() method.')
            predictions[i] = self.prediction_dictionary[hash(testing_example.tostring())]
        return predictions

# This part shows that you can get any accuracy up to 100% from models predicting labels randomly by applying inference
# the algorithm suggested by Babayan et al. at Predicting reservoir hosts and arthropod vectors from evolutionary
# signatures in RNA virus genomes (https://science.sciencemag.org/content/362/6414/577).
#
# Here we load data from the original paper similarly to other reproductions:

data = pd.read_csv('../BabayanEtAl_VirusData.csv', low_memory=False)

experiment_data = data[data['Reservoir'].isin(['Artiodactyl',
                                               'Carnivore',
                                               'Fish',
                                               'Galloanserae',
                                               'Insect',
                                               'Neoaves',
                                               'Plant',
                                               'Primate',
                                               'Pterobat',
                                               'Rodent',
                                               'Vespbat'])]

experiment_data = experiment_data.replace({'Reservoir': ['Artiodactyl', 'Carnivore', 'Fish', 'Galloanserae', 'Insect', 'Neoaves', 'Plant', 'Primate', 'Pterobat', 'Rodent', 'Vespbat']},
                                          {'Reservoir': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})

data_array = experiment_data.iloc[:, 6:].to_numpy()
label_array = experiment_data['Reservoir'].to_numpy()

# Here we create `num_models` random classifiers using this data. All of the classifiers just arrange a random class
# to every data point and remembers it to use for consistent prediction.

num_models = 100
classifiers = []
for _ in range(num_models):
    new_classifier = RandomClassifier(num_classes=11)
    new_classifier.add_data(data_array)
    classifiers.append(new_classifier)

accuracy = np.zeros(num_models)

for model_id in range(num_models):
    prediction = classifiers[model_id].predict(data_array)
    accuracy[model_id] = (np.sum(prediction==label_array))/prediction.shape

print('Mean accuracy for {} models is: {}'.format(num_models, np.mean(accuracy)))

subset_size = 50

subsets_data = []
subsets_labels = []
subsets_indices = []
for _ in range(num_models):
  random_indices = np.random.randint(0, data_array.shape[0], subset_size)
  subsets_data.append(data_array[random_indices])
  subsets_labels.append(label_array[random_indices])
  subsets_indices.append(random_indices)

def predict_focal_point(focal_point_id):

  # Let us calculate accuracy of each of our random classifiers on every of the subsets.

  model_performance = []

  for model_id in range(len(classifiers)):
    # Checking if focal point is in the CV set for the model number `model_id`.
    # If it is, we will use corresponding CV set as is. If it is not, we will
    # add focal point to the CV set.
    if focal_point_id in subsets_indices[model_id]:
      test_data = subsets_data[model_id]
      test_labels = subsets_labels[model_id]
    else:
      test_data = np.append(subsets_data[model_id], [data_array[focal_point_id]], axis=0)
      test_labels = np.append(subsets_labels[model_id], [label_array[focal_point_id]], axis=0)

    model_accuracy = np.sum(classifiers[model_id].predict(test_data) == test_labels) / test_labels.shape[0]

    model_performance.append(model_accuracy)

  # Here we get the indices of the models with the highest accuracy on CV sets, get 25% of them and
  # make prediction for the focal point.

  indices_accent = np.argsort(model_performance)
  top_25_classifier_indices = indices_accent[int(len(indices_accent)*0.75):]
  top_25_classifiers = [classifiers[i] for i in top_25_classifier_indices]

  top_25_predictions = [classifier.predict(np.array([data_array[focal_point_id]]))[0] for classifier in top_25_classifiers]

  # Here we implement bagging mechanism for the best 25% of the models.
  prediction = stats.mode(top_25_predictions)[0]

  return int(prediction)

predictions = []

# for focal_point_id in range(data_array.shape[0]):
for focal_point_id in range(data_array.shape[0]):
    focal_point_prediction = predict_focal_point(focal_point_id)
    predictions.append(focal_point_prediction)

# print(np.array(predictions)==np.array(label_array))

final_accuracy = np.sum(np.array(predictions)==label_array)/label_array.shape

print('Overall accuracy is: {}'.format(final_accuracy))

num_models = 5000
classifiers = []
for _ in range(num_models):
    new_classifier = RandomClassifier(num_classes=11)
    new_classifier.add_data(data_array)
    classifiers.append(new_classifier)

subset_size = 50

subsets_data = []
subsets_labels = []
subsets_indices = []
for _ in range(num_models):
  random_indices = np.random.randint(0, data_array.shape[0], subset_size)
  subsets_data.append(data_array[random_indices])
  subsets_labels.append(label_array[random_indices])
  subsets_indices.append(random_indices)

predictions = []

for focal_point_id in range(data_array.shape[0]):
    focal_point_prediction = predict_focal_point(focal_point_id)
    predictions.append(focal_point_prediction)

final_accuracy = np.sum(np.array(predictions)==label_array)/label_array.shape

print('Overall accuracy is: {}'.format(final_accuracy))