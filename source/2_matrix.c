#include "../headers/2_matrix.h"

matrix_t create_matrix(int rows, int cols)
{
	matrix_t matrix = malloc(sizeof(mat));
	if (!matrix) {
		perror("couldn't allocate the matrix structure in create_matrix");
		return NULL;
	}

	matrix->data = malloc(sizeof(float) * rows * cols);
	// flat matrix row by row
	if (!matrix->data) {
		free(matrix);
		perror("couldn't allocate the float* in the matrix structure in create_matrix");
		return NULL;
	}
	matrix->cols = cols;
	matrix->rows = rows;

	return matrix;
}

void free_matrix(matrix_t *m)
{
	if (!*m) return;
	free((*m)->data);
	free(*m);
	*m = NULL;
}

void fill_matrix(matrix_t m, float value)
{
	int len = m->cols * m->rows;
	for (int i = 0; i < len; i++)
		m->data[i] = value;
}

void matrix_random_init(matrix_t m, float min, float max)
{
	int len = m->cols * m->rows;
	for (int i = 0; i < len; i++) {
		float x = (float)rand() / (float)RAND_MAX;
		m->data[i] = min + x * (max - min);
	}
}

matrix_t multiply_matrix(matrix_t a, matrix_t b)
{
	if (a->cols != b->rows) {
		perror("tried to multiply two matrices that couldn't be multiplied");
		return NULL;
	}
	matrix_t rez = create_matrix(a->rows, b->cols);
	if (!rez) {
		perror ("couldn't create matrix in multiply_matrix");
		return NULL;
	}
	fill_matrix(rez, 0.0f);

	for (int i = 0; i < a->rows ; i++)
		for (int j = 0; j < b->cols ; j++)
			for (int k = 0; k < a->cols ; k++)
				rez->data[i * rez->cols + j] += a->data[i * a->cols + k] * b->data[k * b->cols + j];
	return rez;
}

matrix_t add_matrix(matrix_t a, matrix_t b)
{
	if ((a->cols != b->cols) || (a->rows != b->rows)) {
		perror("tried to add two matrices that dont have the same dimensions");
		return NULL;
	}

	matrix_t rez = copy_matrix(a);
	if (!rez) {
		perror ("couldn't copy matrix in add_matrix");
		return NULL;
	}
	int len = a->cols * a->rows;

	for (int i = 0; i < len; i++)
		rez->data[i] += b->data[i];

	return rez;
}

matrix_t subtract_matrix(matrix_t a, matrix_t b)
{
	if ((a->cols != b->cols) || (a->rows != b->rows)) {
		perror("tried to subtract two matrices that dont have the same dimensions");
		return NULL;
	}

	matrix_t rez = copy_matrix(a);
	if (!rez) {
		perror ("couldn't copy matrix in subtract_matrix");
		return NULL;
	}
	int len = a->cols * a->rows;

	for (int i = 0; i < len; i++)
		rez->data[i] -= b->data[i];

	return rez;
}

matrix_t matrix_add_bias(matrix_t m, matrix_t bias)
{
	if (m->cols != bias->cols) {
		perror("the bias and the matrix dont have the same number of columns");
	}
	matrix_t rez = copy_matrix(m);
	if (!rez) {
		perror ("couldn't copy matrix in matrix_add_bias");
		return NULL;
	}
	for (int i = 0; i < rez->rows; i++)
		for (int j = 0; j < rez->cols; j++)
			rez->data[i * rez->cols + j] += bias->data[j];
	return rez;
}

matrix_t matrix_elementwise_multiply(matrix_t a, matrix_t b)
{
	if ((a->cols != b->cols) || (a->rows != b->rows)) {
		perror("tried to add two matrices that dont have the same dimensions");
		return NULL;
	}

	matrix_t rez = copy_matrix(a);
	if (!rez) {
		perror ("couldn't copy matrix in matrix_elementwise_multply");
		return NULL;
	}
	int len = a->cols * a->rows;

	for (int i = 0; i < len; i++)
		rez->data[i] *= b->data[i];

	return rez;    
}

matrix_t matrix_scalar_multiply(matrix_t m, float scalar)
{
	
	matrix_t rez = copy_matrix(m);
	if (!rez) {
		perror ("couldn't copy matrix in matrix_scalar_multiply");
		return NULL;
	}
	int len = rez->cols * rez->rows;

	for (int i = 0; i < len; i++)
		rez->data[i] *= scalar;

	return rez; 
}

matrix_t transpose_matrix(matrix_t m)
{
	matrix_t rez = create_matrix(m->cols, m->rows);
	if (!rez) {
		perror ("couldn't create matrix in transpose_matrix");
		return NULL;
	}
	for (int i = 0; i < rez->rows; i++) 
		for (int j = 0; j < rez->cols; j++)
			rez->data[i * rez->cols + j] = m->data[j * m->cols + i];
			
	return rez;
}

matrix_t apply_matrix(matrix_t m, float (*func)(float))
{
	matrix_t rez = copy_matrix(m);
	if (!rez) {
		perror ("couldn't copy matrix in apply_matrix");
		return NULL;
	}
	int len = rez->cols * rez->rows;

	for (int i = 0; i < len; i++)
		rez->data[i] = func(rez->data[i]);

	return rez; 
}

matrix_t copy_matrix(matrix_t m)
{
	matrix_t cpy = create_matrix(m->rows, m->cols);
	if (!cpy) {
		perror ("couldn't create matrix in copy_matrix");
		return NULL;
	}
	memcpy(cpy->data, m->data, m->rows * m->cols * sizeof(float));
	return cpy;
}

void print_matrix(matrix_t m)
{
	for (int i = 0; i < m->rows; i++) {
		for (int j = 0; j < m->cols; j++)
			printf("%6.2f ",m->data[i * m->cols + j]);
		printf("\n");
	}
}

matrix_t matrix_sum_rows(matrix_t m)
{
	matrix_t rez = create_matrix(1, m->cols);
	if (!rez) {
		perror ("couldn't create matrix in matrix_sum_rows");
		return NULL;
	}
	fill_matrix(rez, 0.0f);
	for (int i = 0; i < m->rows; i++) {
		for (int j = 0; j < m->cols; j++)
			rez->data[j] += m->data[i * m->cols + j];
	}
	return rez;
}
