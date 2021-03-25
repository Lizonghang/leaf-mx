import os
import numpy as np
from mxnet import init

from utils.model_utils import build_net
from baseline_constants import BYTES_WRITTEN_KEY, \
    BYTES_READ_KEY, LOCAL_COMPUTATIONS_KEY


class Server:

    def __init__(self, client_model, dataset, model_name, num_classes, ctx):
        self.dataset = dataset
        self.model_name = model_name
        self.ctx = ctx
        self.selected_clients = []

        # build and synchronize the global model
        self.model = build_net(dataset, model_name, num_classes, self.ctx)
        self.model.set_params(client_model.get_params())

        # build a model for merging updates
        self.merged_update = build_net(
            dataset, model_name, num_classes, self.ctx, init.Zero())
        self.total_weight = 0

    def select_clients(self, my_round, possible_clients, num_clients=20):
        """Selects num_clients clients randomly from possible_clients.
        Note that within function, num_clients is set to
            min(num_clients, len(possible_clients)).
        Args:
            my_round: The current training round, used for
                random sampling.
            possible_clients: Clients from which the server can select.
            num_clients: Number of clients to select, default 20.
            base_dist: Real data distribution, usually global_dist.
            display: Visualize data distribution when set to True.
            metrics_dir: Directory to save metrics files.
        Returns:
            clients_info: List of (num_train_samples, num_test_samples)
                of selected clients.
        """
        # Randomly select num_clients clients
        num_clients = min(num_clients, len(possible_clients))
        np.random.seed(my_round)
        self.selected_clients = np.random.choice(
            possible_clients, num_clients, replace=False)

        clients_info = [(c.num_train_samples, c.num_test_samples)
                        for c in self.selected_clients]
        return clients_info

    def train_model(self, my_round, num_epochs, batch_size, clients=None):
        """Trains model on self.selected_clients if clients=None;
        Each client's data is trained with the given number of epochs
        and batches.
        Args:
            my_round: The current training round, used for learning rate
                decay.
            num_epochs: Number of epochs to train.
            batch_size: Size of training batches.
            clients: list of Client objects (optional).
        Returns:
            sys_metrics, including:
            bytes_written: number of bytes written by each client to server
                dictionary with client ids as keys and integer values.
            client computations: number of FLOPs computed by each client
                dictionary with client ids as keys and integer values.
            bytes_read: number of bytes read by each client from server
                dictionary with client ids as keys and integer values.
        """
        if clients is None:
            clients = self.selected_clients

        sys_metrics = {
            c.id: {BYTES_WRITTEN_KEY: 0,
                   BYTES_READ_KEY: 0,
                   LOCAL_COMPUTATIONS_KEY: 0}
            for c in clients}

        for c in clients:
            c.set_model(self.model)
            comp, num_samples, update = c.train(
                my_round, num_epochs, batch_size)
            self.merge_updates(num_samples, update)

            sys_metrics[c.id][BYTES_READ_KEY] += c.model.size
            sys_metrics[c.id][BYTES_WRITTEN_KEY] += c.model.size
            sys_metrics[c.id][LOCAL_COMPUTATIONS_KEY] = comp

        return sys_metrics

    def merge_updates(self, client_samples, update):
        """Aggregate updates from clients based on train data size.
        Args:
            client_samples: Size of train data used by this client.
            update: The model trained by this client.
        """
        merged_update_ = list(self.merged_update.get_params())
        current_update_ = list(update)
        num_params = len(merged_update_)

        self.total_weight += client_samples

        for p in range(num_params):
            merged_update_[p].set_data(
                merged_update_[p].data() +
                (client_samples * current_update_[p].data()))

    def update_model(self):
        """Update self.model with averaged merged update."""
        merged_update_ = list(self.merged_update.get_params())
        num_params = len(merged_update_)

        for p in range(num_params):
            merged_update_[p].set_data(
                merged_update_[p].data() / self.total_weight)

        self.model.set_params(self.merged_update.get_params())

        self.total_weight = 0
        self.merged_update.set_params([])

    def test_model(self, clients_to_test, set_to_use='test'):
        """Tests self.model on given clients.
        Tests model on self.selected_clients if clients_to_test=None.
        Args:
            clients_to_test: list of Client objects.
            set_to_use: Dataset to test on, either "train" or "test".
        Returns:
            metrics: Dict of metrics returned by the model.
        """
        metrics = {}

        if clients_to_test is None:
            clients_to_test = self.selected_clients

        for client in clients_to_test:
            client.set_model(self.model)
            c_metrics = client.test(set_to_use)
            metrics[client.id] = c_metrics

        return metrics

    def set_model(self, model):
        """Set the model data to specified model.
        Args:
            model: The specified model.
        """
        self.model.set_params(model.get_params())

    def online(self, clients):
        """Return clients that are online.
        Args:
            clients: List of all clients registered at this
                middle server.
        Returns:
            online_clients: List of all online clients.
        """
        online_clients = clients
        assert len(online_clients) != 0, "No client available."
        return online_clients

    def get_clients_info(self, clients):
        """Returns the ids, groups and num_samples for the given clients.
        Args:
            clients: List of Client objects.
        Returns:
            ids: List of client_ids for the given clients.
            groups: Map of {client_id: group_id} for the given clients.
            num_samples: Map of {client_id: num_samples} for the given
                clients.
        """
        if clients is None:
            clients = self.selected_clients

        ids = [c.id for c in clients]
        groups = {c.id: c.group for c in clients}
        num_samples = {c.id: c.num_samples for c in clients}
        return ids, groups, num_samples

    def save_model(self, log_dir):
        """Saves the server model to:
            {log_dir}/{self.model_name}.params
        """
        self.model.save_parameters(
            os.path.join(log_dir, self.model_name + ".params"))
