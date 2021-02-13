import warnings
from mxnet import nd


class Client:

    def __init__(self, client_id, group, train_data, test_data, model):
        self._model = model
        self.id = client_id
        self.group = group
        self.train_data = {
            "x": self.process_data(train_data["x"]),
            "y": self.process_data(train_data["y"])
        }
        self.test_data = {
            "x": self.process_data(test_data["x"]),
            "y": self.process_data(test_data["y"])
        }

    def train(self, num_epochs=1, batch_size=10):
        """Trains on self.model using the client's train_data.
        Args:
            num_epochs: Number of epochs to train.
            batch_size: Size of training batches.
        Return:
            comp: number of FLOPs executed in training process
            num_samples: number of samples used in training
            update: set of weights
            update_size: number of bytes in update
        """

        comp, update = self.model.train(
            self.train_data, num_epochs, batch_size)
        return comp, self.num_train_samples, update

    def test(self, set_to_use="test"):
        """Tests self.model on self.test_data.

        Args:
            set_to_use. Set to test on. Should be in ["train", "test"].
        Return:
            dict of metrics returned by the model.
        """
        assert set_to_use in ["train", "test", "val"]
        if set_to_use == "train":
            data = self.train_data
        elif set_to_use == "test" or set_to_use == "val":
            data = self.test_data
        return self.model.test(data)

    @property
    def num_train_samples(self):
        """Number of train samples for this client.
        Return:
            int: Number of train samples for this client
        """
        return len(self.train_data["y"])

    @property
    def num_test_samples(self):
        """Number of test samples for this client.
        Return:
            int: Number of test samples for this client
        """
        return len(self.test_data["y"])

    @property
    def num_samples(self):
        """Number of samples for this client (train + test).
        Return:
            int: Number of samples for this client
        """
        return self.num_train_samples + self.num_test_samples

    @property
    def model(self):
        """Returns this client reference to model being trained"""
        return self._model

    @model.setter
    def model(self, model):
        warnings.warn('The current implementation shares the model among all clients.'
                      'Setting it on one client will effectively modify all clients.')
        self._model = model

    def process_data(self, data):
        return nd.array(data, ctx=self.model.ctx)
