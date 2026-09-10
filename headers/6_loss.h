#ifndef LOSS_H
#define LOSS_H

#include "../headers/2_matrix.h"

// for monitoring progress, not for actual training
float cross_entropy_loss(matrix_t predicted, matrix_t target);

// computes predicted - target for backpropagation
matrix_t cross_entropy_gradient(matrix_t predicted, matrix_t target);

#endif
