import importlib
import numpy as np
import os
import random
import mxnet as mx

import metrics.writer as metrics_writer

from client import Client
from server import Server
from baseline_constants import MODEL_PARAMS
from utils.args import parse_args
from utils.model_utils import read_data


def main():
    args = parse_args()
    log_dir = os.path.join(
        args.log_dir, args.dataset, str(args.log_rank))
    os.makedirs(log_dir, exist_ok=True)
    log_fn = "output.%i" % args.log_rank
    log_file = os.path.join(log_dir, log_fn)
    log_fp = open(log_file, "w+")

    # Set the random seed, affects client sampling and batching
    random.seed(1 + args.seed)
    np.random.seed(12 + args.seed)
    mx.random.seed(123 + args.seed)

    client_path = "%s/client_model.py" % args.dataset
    if not os.path.exists(client_path):
        print("Please specify a valid dataset.",
              file=log_fp, flush=True)
    client_path = "%s.client_model" % args.dataset

    mod = importlib.import_module(client_path)
    ClientModel = getattr(mod, "ClientModel")

    num_rounds = args.num_rounds
    eval_every = args.eval_every
    clients_per_round = args.clients_per_round
    ctx = mx.gpu(args.ctx) if args.ctx >= 0 else mx.cpu()

    # model params (hyper)
    param_key = "%s.%s" % (args.dataset, args.model)
    model_params = MODEL_PARAMS[param_key]
    if args.lr != -1:
        model_params_list = list(model_params)
        model_params_list[0] = args.lr
        model_params = tuple(model_params_list)
    num_classes = model_params[1]

    # Create client model, and share params with server model
    client_model = ClientModel(
        args.seed, args.dataset, args.model, ctx, args.count_ops, *model_params)

    # Create server
    server = Server(
        client_model, args.dataset, args.model, num_classes, ctx)

    # Create clients
    clients = setup_clients(
        args.dataset, client_model, args.use_val_set)
    _ = server.get_clients_info(clients)
    client_ids, client_groups, client_num_samples = _
    print("Total number of clients: %d" % len(clients),
          file=log_fp, flush=True)

    # Display initial status
    print("--- Random Initialization ---",
          file=log_fp, flush=True)
    stat_writer_fn = get_stat_writer_function(
        client_ids, client_groups, client_num_samples, args)
    sys_writer_fn = get_sys_writer_function(args)
    print_stats(
        0, server, clients, client_num_samples,
        stat_writer_fn, args.use_val_set, log_fp)

    # Training simulation
    for r in range(num_rounds):
        print("--- Round %d of %d: Training %d clients ---"
              % (r, num_rounds-1, clients_per_round),
              file=log_fp, flush=True)

        # Select clients
        server.select_clients(r, online(clients), clients_per_round)
        _ = server.get_clients_info(server.selected_clients)
        c_ids, c_groups, c_num_samples = _

        # Simulate server model training on selected clients' data
        sys_metrics = server.train_model(
            num_epochs=args.num_epochs, batch_size=args.batch_size)
        sys_writer_fn(r, c_ids, sys_metrics, c_groups, c_num_samples)

        # Update server model
        server.update_model()

        # Test model
        if (r + 1) % eval_every == 0 or (r + 1) == num_rounds:
            print_stats(
                r, server, clients, client_num_samples,
                stat_writer_fn, args.use_val_set, log_fp)

    server.save_model(log_dir)
    log_fp.close()


def online(clients):
    """Users that are always online."""
    return clients


def create_clients(users, groups, train_data, test_data, model):
    if len(groups) == 0:
        groups = [[] for _ in users]
    clients = [Client(u, g, train_data[u], test_data[u], model)
               for u, g in zip(users, groups)]
    return clients


def setup_clients(dataset, model=None, use_val_set=False):
    """Instantiates clients based on given train and test data directories.
    Return:
        all_clients: list of Client objects.
    """
    eval_set = "test" if not use_val_set else "val"
    train_data_dir = os.path.join("data", dataset, "data", "train")
    test_data_dir = os.path.join("data", dataset, "data", eval_set)

    data = read_data(train_data_dir, test_data_dir)
    users, groups, train_data, test_data = data

    clients = create_clients(users, groups, train_data, test_data, model)
    return clients


def get_stat_writer_function(ids, groups, num_samples, args):

    def writer_fn(num_round, metrics, partition):
        metrics_writer.print_metrics(
            num_round, ids, metrics, groups, num_samples, partition,
            args.metrics_dir, "{}_{}".format(args.metrics_name, "stat"))

    return writer_fn


def get_sys_writer_function(args):

    def writer_fn(num_round, ids, metrics, groups, num_samples):
        metrics_writer.print_metrics(
            num_round, ids, metrics, groups, num_samples, "train",
            args.metrics_dir, "{}_{}".format(args.metrics_name, "sys"))

    return writer_fn


def print_stats(num_round, server, clients, num_samples, writer, use_val_set, log_fp=None):
    train_stat_metrics = server.test_model(clients, set_to_use="train")
    print_metrics(
        train_stat_metrics, num_samples, prefix="train_", log_fp=log_fp)
    writer(num_round, train_stat_metrics, "train")

    eval_set = "test" if not use_val_set else "val"
    test_stat_metrics = server.test_model(clients, set_to_use=eval_set)
    print_metrics(
        test_stat_metrics, num_samples, prefix="{}_".format(eval_set), log_fp=log_fp)
    writer(num_round, test_stat_metrics, eval_set)


def print_metrics(metrics, weights, prefix="", log_fp=None):
    """Prints weighted averages of the given metrics.
    Args:
        metrics: dict with client ids as keys. Each entry is a dict
            with the metrics of that client.
        weights: dict with client ids as keys. Each entry is the weight
            for that client.
    """
    ordered_weights = [weights[c] for c in sorted(weights)]
    metric_names = metrics_writer.get_metrics_names(metrics)
    for metric in metric_names:
        ordered_metric = [metrics[c][metric] for c in sorted(metrics)]
        print('%s: %g, 10th percentile: %g, 50th percentile: %g, 90th percentile %g' \
              % (prefix + metric,
                 np.average(ordered_metric, weights=ordered_weights),
                 np.percentile(ordered_metric, 10),
                 np.percentile(ordered_metric, 50),
                 np.percentile(ordered_metric, 90)),
              file=log_fp, flush=True)


if __name__ == "__main__":
    main()
