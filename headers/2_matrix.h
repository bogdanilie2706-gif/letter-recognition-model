#ifndef MATRIX_H
#define MATRIX_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

typedef struct mat {
	int rows;
	int cols;
	float *data;
	// flat row major data[i*cols + j] = element (i,j)
} mat, *matrix_t;

// allocates a flat row major matrix
matrix_t create_matrix(int rows, int cols);

// frees a matrix structure
void free_matrix(matrix_t *m);

// sets the whole matrix to the same value given
void fill_matrix(matrix_t m, float value);

// sets the values of the matrix randomly between min and max for weights
void matrix_random_init(matrix_t m, float min, float max);

// multiplies two matrices a(m x n) * b(n x p) = results (m x p)
// check a->cols == b->rows
matrix_t multiply_matrix(matrix_t a, matrix_t b);

// adds two matrices of same dimensions toghether
matrix_t add_matrix(matrix_t a, matrix_t b);

// subtracts two matrices of same dimensions (for backpropagation / gradient)
matrix_t subtract_matrix(matrix_t a, matrix_t b);

// adds the  bias (1 x n) to each row of a matrix (batch x n)
matrix_t matrix_add_bias(matrix_t m, matrix_t bias);

// multiplying each element of a matrix (for backpropagation)
matrix_t matrix_elementwise_multiply(matrix_t a, matrix_t b);

// multiplies a matrix with a scalar 
matrix_t matrix_scalar_multiply(matrix_t m, float scalar);

// transposes a matrix matrix[i][j] = rez[j][i]
matrix_t transpose_matrix(matrix_t m);

// applies a given function to all of the elemnts of a matrix
// useful for activations: sigmoid, ReLU etc.
matrix_t apply_matrix(matrix_t m, float (*func)(float));

// returns a complete copy of a matrix
matrix_t copy_matrix(matrix_t m);

// Prints the whole matrix (for debugging)
void print_matrix(matrix_t m);

// sum of all elements on each column result (1 x cols)
// for backpropagation or the gradient of the bias (sum on the batch batch)
matrix_t matrix_sum_rows(matrix_t m);

#endif