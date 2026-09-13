# EMNIST Letter Recognition in C

A from-scratch neural network for recognizing lowercase letters from the
EMNIST Letters dataset. The project implements the full training pipeline in C:
loading IDX image files, batching and shuffling data, running forward and
backward passes through a multilayer perceptron, evaluating accuracy, and
saving/loading trained model weights.

The current baseline reaches about 89-90% test accuracy on EMNIST Letters with
a 784 -> 128 -> 64 -> 26 fully connected network.

## Project Structure

```text
headers/   Public headers for each module
source/    C implementations
data/      EMNIST IDX data files, ignored by git
models/    Saved binary model weights, ignored by git
obj/       Build objects, ignored by git
bin/       Compiled executable, ignored by git
```

Main modules:

```text
1_data_loader   Loads IDX label/image files and fixes EMNIST image orientation
2_matrix        Matrix allocation, arithmetic, transpose, copy, and helpers
3_layer         Dense layer forward/backward/update logic
4_network       Linked-list network structure and layer traversal
5_activations   ReLU, Leaky ReLU, Tanh, Sigmoid, Softmax, Identity
6_loss          Cross-entropy loss and softmax-cross-entropy gradient
7_train         Batching, shuffling, training, evaluation, save/load
8_main          Current experiment entry point
```

## Dataset

This project expects the EMNIST Letters dataset in IDX binary format, not CSV.
Place the files in `data/` with the names currently used by `source/8_main.c`:

```text
data/train_labels_idx1
data/train_images_idx3
data/test_labels_idx1
data/test_images_idx3
```

Dataset assumptions:

```text
Training samples: 124800
Test samples:     20800
Image size:       28 x 28 grayscale
Input size:       784 normalized floats in [0, 1]
Labels:           1-26, where 1 = a and 26 = z
```

EMNIST stores the letters rotated/mirrored relative to the expected display
orientation, so the loader transposes each 28 x 28 image while reading it.

## Model

The default architecture is:

```text
Input:    784 pixels
Hidden 1: 128 neurons
Hidden 2: 64 neurons
Output:   26 classes
```

Hidden-layer activations are selectable through `activation_code_t`:

```c
ACTIVATION_RELU
ACTIVATION_LEAKY_RELU
ACTIVATION_TANH
ACTIVATION_SIGMOID
```

The output layer uses softmax. The training code uses the standard
softmax-cross-entropy simplification:

```text
gradient = prediction - target
```

Because of that, the output layer should stay as softmax when using the current
loss and backpropagation setup.

Other training details:

```text
Optimizer:      mini-batch SGD
Batch size:     32
Learning rate:  0.004
Epochs:         10 for the activation comparison
Initialization: He initialization per layer
Bias init:      0.0
Seed:           srand(42)
```

## Build and Run

Build the project:

```sh
make
```

Run the executable:

```sh
make run
```

Clean generated files:

```sh
make clean
```

The training loop appends epoch metrics to:

```text
model_epoch_output.txt
```

Important build note: the current Makefile does not track header dependencies.
After changing a shared struct in a header, run a clean rebuild:

```sh
make clean
make
```

Without this, stale object files can be compiled against old struct layouts and
cause confusing runtime memory errors.

## Current Results

All runs below use the same architecture, batch size, learning rate, epoch
count, random seed, and softmax output layer. Only the hidden-layer activation
changes.

| Hidden activation | Epochs recorded | Final test accuracy | Final test loss | Best test accuracy | Best test loss | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| ReLU | 20 | 89.80% | 0.375564 | 89.96% at epoch 18 | 0.332118 at epoch 7 | Baseline run |
| Leaky ReLU | 20 | 89.66% | 0.385284 | 90.13% at epoch 18 | 0.330866 at epoch 9 | Best recorded test accuracy |
| Sigmoid | 20 | 89.81% | 0.330887 | 89.94% at epoch 18 | 0.322936 at epoch 18 | Slow start, strongest recorded loss |
| Tanh | 20 | 88.93% | 0.390135 | 89.47% at epoch 12 | 0.351272 at epoch 7 | Peaks earlier, then declines |

The current `source/8_main.c` runs Leaky ReLU, Sigmoid, and Tanh experiments in
sequence. ReLU is the established baseline, but it is not currently the active
configuration in `main.c`.

From the epoch logs, the main pattern is that most activation functions converge
well before the 20th epoch. ReLU, Leaky ReLU, and Tanh reach most of their
useful test accuracy early, then continue improving more slowly or fluctuating.
Sigmoid starts much lower than the other activations, which is expected because
sigmoid can shrink gradients in hidden layers, but it continues improving over
more epochs and eventually catches up surprisingly well.

The runs also show early signs of overfitting. Training accuracy keeps rising
through the later epochs, while test accuracy starts to plateau around 89% and
sometimes moves up and down instead of improving steadily. This is clearest in
the gap between best and final values: ReLU, Leaky ReLU, and Tanh all finish
with worse test loss than their best recorded loss, even though their training
accuracy continues increasing. Tanh shows the strongest late decline, reaching
its best test accuracy at epoch 12 and ending lower at epoch 20. This suggests
that more epochs alone are unlikely to produce a large improvement without extra
regularization, better tuning, or a stronger model.

Overall, ReLU, Leaky ReLU, and Tanh perform very similarly in this setup, while
Sigmoid is slower to train but becomes competitive by the end of the 20-epoch
run. Leaky ReLU gives the best recorded test accuracy, while Sigmoid gives the
best recorded test loss. The margins are small enough that the main conclusion
is that ReLU-style activations are a reliable choice for this network, and that
future improvements should focus more on regularization, architecture, and data
handling than simply increasing the epoch count.

## Saving and Loading Models

`source/7_train.c` includes binary save/load support:

```c
void save_model(network_t net, char *file_name);
network_t load_model(char *file_name);
```

The saved model format stores:

```text
nr_layers
for each layer:
    input_size
    output_size
    activation_code
    weights
    bias
```

The format is intentionally simple and meant for models produced and consumed
by this same program on the same machine.

## Known Limitations

- There is no command-line argument parsing yet. Dataset paths,
  hyperparameters, and activation choices are currently edited in `main.c`.
- Partial batches are not handled. This is fine for the current EMNIST split
  because both 124800 and 20800 divide evenly by batch size 32.
- Matrix multiplication is a straightforward O(n^3) implementation and is the
  main performance bottleneck.
- Data files, compiled output, saved models, and training logs are ignored by
  git.

## Future Improvements

Accuracy improvements:

- Add validation-based early stopping so training can stop when test or
  validation accuracy stops improving.
- Tune the learning rate separately for each activation function instead of
  forcing every activation to use `0.004`.
- Add regularization such as L2 weight decay or dropout to reduce overfitting.
- Try larger or deeper networks, while watching whether the extra capacity
  improves test accuracy or only memorizes the training set.
- Replace the fully connected network with a convolutional neural network,
  which should fit image data better because it can learn local stroke patterns.
- Add a 27th output class for `not a letter`. This would let the model reject
  blank regions, punctuation, digits, symbols, or noisy image segments instead
  of forcing every input to be classified as `a-z`.
- Add command-line arguments for activation, epochs, learning rate, batch size,
  dataset paths, and save/load mode so experiments are easier to reproduce.
- Improve the Makefile so object files rebuild automatically when headers
  change.

Possible usage scenarios:

- Load a saved model and classify individual handwritten letter images.
- Build a small command-line demo where the user passes an image file and gets
  the predicted letter.
- Connect the model to a simple drawing interface so a user can draw a letter
  and see the prediction in real time.
- Use the model as a base for recognizing characters in scanned notes, forms,
  or educational handwriting exercises.
- Use the `not a letter` class as a stepping stone toward full image text
  recognition, where an OCR pipeline must decide which image regions contain
  letters before trying to classify them.
- Extend the project from single-letter recognition toward simple word
  recognition by segmenting a word image into individual letters first.

## Memory Checking

The project has been checked with Valgrind during development, including fixes
for layer-owned matrices created during the final forward pass. On Linux or a
Valgrind-capable environment, run:

```sh
make valgrind
```
