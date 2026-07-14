#include "../headers/6_loss.h"

float cross_entropy_loss(matrix_t predicted, matrix_t target)
{
	if (predicted->cols != target->cols || predicted->rows != target->rows) {
		perror("predicted and target have different dimensions in cross_entropy_loss");
		return NAN;
	}

	float loss = 0.0f;
	for (int i = 0; i < predicted->rows; i++) {
		for (int j = 0; j < predicted->cols; j++) {
			loss -= target->data[i * predicted->cols + j] * 
				logf(predicted->data[i * predicted->cols + j] + 1e-9f);
			// added 1e-9 to avoid log(0)
		}
	}
	loss /= predicted->rows;
	return loss;
}

matrix_t cross_entropy_gradient(matrix_t predicted, matrix_t target)
{
	if (predicted->cols != target->cols || predicted->rows != target->rows) {
		perror("predicted and target have different dimensions in cross_entropy_gradient");
		return NULL;
	}
	return subtract_matrix(predicted, target);
}
