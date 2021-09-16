from Dataset import *
from abc import ABC, abstractmethod


class ModelPerso(ABC):
    """Classe abstraite pour la classification"""
    def __init__(self, training_set=Dataset(), testing_set=Dataset(), name="model perso"):

        self.training_set = training_set
        self.testing_set = testing_set
        self.name = name

        self.trainable_parameters = None
        self.trained = False
        self.unknown_data = None

        if not(self.training_set.is_null()):
            self.__fit__()
            self.trained = True

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name

    def fit(self, training_set: Dataset, testing_set=Dataset(), skip_test=True):
        """
        Train the model according the data on the training_set and test if the testing_set allows it to.
        :param training_set:
        :param testing_set:
        :param skip_test:
        :return:
        """
        self.training_set = training_set

        self.__trainable__()
        self.__fit__()
        self.trained = True
        if skip_test:
            return None
        return self.test(testing_set)

    def __trainable__(self):
        """
        Check if the model has data to be trained on and raises an exception in the negative case.
        :return:
        """
        if self.training_set.is_null():
            raise Exception("No data to be trained on.")
        pass

    def test(self, data_test=Dataset()):
        """
        Compute the precision of the model on the testing set
        :param data_test:
        :return:
        """
        if data_test.is_null():
            return self.__test__()
        else:
            self.testing_set = data_test
            return self.__test__()

    def __test__(self):
        if self.testing_set.is_null():
            raise Exception("No data to test on.")

        if not self.trained:
            self.__fit__()

        y_predict = self.predict(self.testing_set.features)
        self.precision = np.mean(y_predict == self.testing_set.labels)
        print(f"Precision: {self.precision}")
        return None

    def __classes__(self):
        self.classes = list({y for y in self.training_set.labels})
        self.classes.sort()


    @abstractmethod
    def __fit__(self):
        """
        Train the model according to the training_set.
        :return:
        """
        # TODO : To be implemented
        pass

    def predict(self, features):
        """
        Gives the predictions of the different features
        :param features: iterable
        :return: labels predicted
        """
        self.unknown_data = features
        if not self.trained:
            raise Exception("Not trained model.")

        return self.__predict__()

    @abstractmethod
    def __predict__(self):
        """
        Gives the predictions of the different features
        :return: labels predicted
        """
        # TODO: To be implemented
        pass