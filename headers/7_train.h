#ifndef TRAIN_H
#define TRAIN_H

#include "../headers/1_data_loader.h"
#include "../headers/4_network.h"
#include "../headers/5_activations.h"
#include "../headers/6_loss.h"

typedef struct batches_struct {
    // matrices where each row is an image
    // with nr_batches rows
    matrix_t *input;

    // a column for each letter with nr_batches rows
    // filled with 0.0f and one 1.0f for the target letter
    matrix_t *target;

    int nr_batches;
} batches_struct, *batches_t;

// the training loop that will create the model
// the network must be created prior to call
void train_letter_model(dataset_t train, dataset_t test, network_t net, float learning_rate);

// will return the pointer to the allocated structure
// will also allocate the matrix vectors from the dataset
// both the matrices vectors will have nr_batches number of rows
batches_t create_batches_struct(dataset_t dataset, int nr_batches, int input_cols, int target_cols);

// will free the matrix_t vectors and the structure itself
// checks if the target and input are NULL before freeing
// also setting pointer to NULL
void destroy_batches_struct(batches_t *batches);

// will shuffle the allocated batches_t structure
// used beofre each epoch
void shuffle_batches(dataset_t df, batches_t batches, int *index);

#endif
