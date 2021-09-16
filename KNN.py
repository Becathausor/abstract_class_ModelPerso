import numpy as np
from ModelPerso import *
from Dataset import *


class KNN(ModelPerso):
    def __init__(self, training_set=Dataset(), testing_set=Dataset(), name="KNN", k=3):
        self.training_set = training_set
        self.testing_set = testing_set
        self.name = name

        self.trainable_parameters = None
        self.trained = False
        self.unknown_data = None
        super().__init__(training_set, testing_set, name)
        self.K = k

    def __fit__(self):
        """
        Train the model according to the training_set. According to the KNN algorithm
        :return:
        """
        self.__classes__()
        self.trainable_parameters = []
        for label in self.classes:
            self.trainable_parameters.append(
                np.array([self.training_set.features[i] for i in range(self.training_set.n) if
                          self.training_set.labels[i] == label])
            )
        return None

    def __predict_one__(self,elem):
        norms = list(map(np.linalg.norm, self.training_set.features - elem))
        first_winners = self.training_set.labels[argmin_k(norms, self.K)]

        position = [self.classes[int(first_winner)] for first_winner in first_winners]
        scores_labels = [count(position, self.classes[i]) for i in range(len(self.classes))]
        winner = self.classes[int(np.argmax(scores_labels))]
        return winner

    def __predict__(self):
        """
        Gives the predictions of the different features
        :return: labels predicted
        """

        result = [self.__predict_one__(elem) for elem in self.unknown_data]
        return np.array(result)


def argmin_k(list_iter: list, k: int):
    """
    Compute the indices of the k highest elements of the list l.
    :param list_iter: list
    :param k: int
    :return: bests_ind: list
    """
    l_ = list_iter.copy()
    bests_ind = []
    maxi = max(l_) + 1
    for i in range(k):
        if i >= len(l_):
            return bests_ind
        ind = np.argmin(l_)
        bests_ind.append(ind)
        l_[ind] = maxi
    return bests_ind


def count(array_iter, elem):
    return np.sum(array_iter == elem)


def cover_plan(data: Dataset, precision=100):
    x = data.features
    min_x = min(x[:, 0])
    max_x = max(x[:, 0])
    min_y = min(x[:, 1])
    max_y = max(x[:, 1])

    abscisses = np.linspace(min_x, max_x, precision)
    ordonnees = np.linspace(min_y, max_y, precision)

    points = np.zeros((precision**2, 2))
    for i in range(precision):
        for j in range(precision):
            points[precision * i + j] = np.array([abscisses[i], ordonnees[j]])

    return points
    