import random
from sklearn.metrics import accuracy_score

for n_folds in range(100, 1505, 100):
    c = 0
    for k in range(100):
        random.seed(k)
        data_points = []
        n_test_samples = 100

        for i in range(n_folds):
            data_points.append([1])
            for j in range(n_test_samples -1):
                data_points[i].append(random.choice([0,1]))

        predictions = []
        for i in range(n_folds):
            predictions.append([])
            for j in range(n_test_samples):
                predictions[i].append(random.choice([0,1]))

        accuracies = []
        for i in range(n_folds):
            accuracies.append(accuracy_score(data_points[i], predictions[i]))

        top_accuracies_threshold = sorted(accuracies)[int(0.75*n_folds)]

        focus_predictions = []
        for i in range(n_folds):
            if accuracy_score(data_points[i], predictions[i]) > top_accuracies_threshold:
                focus_predictions.append(predictions[i][0])

        # print(sum(focus_predictions)/len(focus_predictions))
        if sum(focus_predictions)/len(focus_predictions) > 0.5:
            c += 1

    print(n_folds, c)


