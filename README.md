# LEAF-MX: A MXNET Benchmark for Federated Learning

LEAF-MX is a MXNET implementation of LEAF, which is originally implemented by TensorFlow. 

More details about LEAF can be found at:

* Github: https://github.com/TalwalkarLab/leaf

* Documentation: https://leaf.cmu.edu/

* Paper: "[LEAF: A Benchmark for Federated Settings](https://arxiv.org/abs/1812.01097)"

## Note

* Go to directory of respective dataset for instructions on generating data.

* Docker is required to run this demo.

## Installation

Enter the ``scripts`` directory and use ``scripts/build-env.dockerfile`` to build the image
 ``leaf-mx:mxnet1.4.1mkl-cu101-py3.7``:

> sudo docker build -f build-env.dockerfile -t leaf-mx:mxnet1.4.1mkl-cu101-py3.7 .

Run ``scripts/run.sh`` to create the container ``leaf-mx.{CONTAINER_RANK}`` and start the
simulation:

> chmod a+x run.sh && ./run.sh

``CONTAINER_RANK`` is an integer defined in ``scripts/run.sh``.

## Instruction

Default values for hyper-parameters are set in ``utils/args.py``, including:

| Variable Name | Default Value | Optional Values | Description |
|---|---|---|---|
| -dataset | "femnist" | "femnist" | Dataset used for federated training. |
| -model | "cnn" | "cnn" | Neural network used for federated training. |
| --num-rounds | 500 | integer | Number of rounds to simulate. |
| --eval-every | 20 | integer | Evaluate the federated model every few rounds. |
| --clients-per-round | 100 | integer | Number of clients participating in each round. |
| --batch-size | 5 | integer | Number of training samples in each batch. |
| --num-epochs | 3 | integer | Number of local epochs in each round. |
| -lr| 0.01 | float | Learning rate for local optimizers. |
| --seed | 0 | integer | Seed for random client sampling and batch splitting. |
| --metrics-name | "metrics" | string | Name for metrics file. |
| --metrics-dir | "metrics" | string | Directory for metrics files. |
| --log-dir | "logs" | string | Directory for log files. |
| --log-rank | 0 | integer | Identity for current training process (i.e., ``CONTAINER_RANK``). Log files will be written to ``logs/{DATASET}/{CONTAINER_RANK}/`` (e.g., ``logs/femnist/0/``) |
| --use-val-set | None | None | Set this option to use the validation set, otherwise the test set is used. |
| --count-ops | None | None | Set this option to enable operation counter, otherwise ``flops=0`` is returned. Enable this will increase the CPU usage and reduce efficiency. |
| -ctx | -1 | integer | Device used for simulation. -1 for CPU and 0~7 for GPUs.

Some commonly used hyper-parameters can also be set through ``scripts/run.sh``, including:

| Environment Name | Variable Name | Description |
|---|---|---|
| CONTAINER_RANK | --log-rank | This will run a container named ``leaf-mx.{CONTAINER_RANK}`` and write log files to ``logs/{DATASET}/{CONTAINER_RANK}/``. |
| BATCH_SIZE | --batch-size | Same as the above table. |
| LEARNING_RATE | -lr | Same as the above table. |
| NUM_EPOCHS | --num-epochs | Same as the above table. |
| NUM_ROUNDS | --num-rounds | Same as the above table. |
| CLIENTS_PER_ROUND | --clients-per-round | Same as the above table. |
| DATASET | -dataset | Same as the above table. |
| MODEL | -model | Same as the above table. |
| EVAL_EVERY | --eval-every | Same as the above table. |

Other required environment variables are:

| Environment Name | Default Value | Description |
|---|---|---|
| NUM_GPU_AVAILABLE | 2 | The number of GPUs available. |
| NUM_GPU_BEGIN | 0 | The index of the first GPU available. |
| IMAGE_NAME | leaf-mx:mxnet1.4.1mkl-cu101-py3.7 | Name of the basic image to be used. |
| CONTAINER_NAME | leaf-mx.{CONTAINER_RANK} | Name of the container created. |
| HOST_NAME | leaf-mx.{CONTAINER_RANK} | Hostname of the container. |
| USE_GPU | (auto) | Index of the GPU device used for simulation. This is set automatically thourgh {NUM_GPU_BEGIN} + {CONTAINER_RANK} % {NUM_GPU_AVAILABLE}. |

We recommend to mount files from the host computer into the container, modify ``scripts/run.sh
`` and change the file path ``/home/lizh/leaf-mx`` to your own path.

## Other Notes

* To avoid the efficiency bottleneck caused by high-frequency data IO (disk → CPU memory → GPU
 memory), the full dataset is loaded into GPU memory, so small datasets are preferred in
simulation.

* Clients share the same model to reduce the memory usage, while server owns two models, one for
the server model and another for the merged update.

* Both the output log and the trained model are stored in ``logs/{DATASET}/{CONTAINER_RANK}/``.

* More features will be added in the future.

## Results and Visualization

Using the default setting (500 rounds), the average test accuracy reaches 0.840537 and the
average test loss reaches 0.535626:

> tail -n 5 logs/femnist/8

```
--- Round 499 of 499: Training 100 clients ---
train_accuracy: 0.863466, 10th percentile: 0.689265, 50th percentile: 0.877485, 90th percentile 0.951814
train_loss: 0.40957, 10th percentile: 0.140034, 50th percentile: 0.356467, 90th percentile 0.975457
test_accuracy: 0.840537, 10th percentile: 0.611111, 50th percentile: 0.851852, 90th percentile 0.973246
test_loss: 0.535626, 10th percentile: 0.116171, 50th percentile: 0.444283, 90th percentile 1.25239
```

Tools in ``metrics/visualization_utils.py`` can be used for visualization, and we give an example
in ``metrics/visualize.py``. This script will read metrics logs from ``metrics/metrics_stat.csv
`` and ``metrics/metrics_sys.csv``, then plot curves of accuracy, bytes written and read, and
flops of clients versus rounds. Note that the function ``calc_longest_flops(*metrics)`` returns 0 if
``--count-ops`` was not set.