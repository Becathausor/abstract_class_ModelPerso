import numpy as np


class Dataset:
    def __init__(self, features=None, labels=None, weights=None):
        """
        Create a Dataset ready to be used in a Data Science Model.
        :param features:
        :param labels:
        :param weights:
        """
        self.features = features
        self.labels = labels
        self.weights = weights

        if self.features is None:
            assert self.labels is None, f"Number of features and labels doesn't correspond, labels not null while " \
                                        f"features is null. "
            return

        assert not self.labels is None, f"Number of features and labels doesn't correspond, labels null while features isn't null."

        self.n = len(self.features)

        if self.n != len(self.labels):
            raise Exception(
                f"Number of features and labels doesn't correspond, {self.n} different from {len(self.labels)}.")

        if self.n != 0:
            self.d = len(self.features[0])
        else:
            self.d = 0

        if self.weights is None:
            self.weights = np.ones(self.n)/self.n

        else:
            self.weights = self.weights / np.sum(weights)

    def to_string(self):
        string_ = f"Features: {self.features}\nLabels: {self.labels} \nWeights: {self.weights}"
        return string_

    def __repr__(self):
        return self.to_string()

    def __str__(self):
        return self.to_string()

    def is_null(self):
        return self.features is None

    def shuffle(self):
        if self.is_null():
            return None
        mat_total = np.concatenate([self.features, self.labels.reshape(self.n, 1)], axis=1)
        np.random.shuffle(mat_total)
        self.features = mat_total[:,:-1]
        self.labels = mat_total[:,-1]
        return None

