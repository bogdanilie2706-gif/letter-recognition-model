#ifndef TRAIN_H
#define TRAIN_H

#include "../headers/1_data_loader.h"
#include "../headers/4_network.h"
#include "../headers/5_activations.h"
#include "../headers/6_loss.h"

#endif

// the training loop that will create the model
// the network must be created prior to call
void train_letter_model(dataset_t train, dataset_t test, network_t net, float learning_rate);
