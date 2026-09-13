#include "../headers/5_activations.h"

static float relu_elem(float value)
{
    if (value > 0)
        return value; // max(0,x)
    return 0;
}

static float relu_derivative_elem(float value)
{
    if (value > 0)
        return 1;
    return 0;
}

matrix_t relu(matrix_t z)
{
    return apply_matrix(z, relu_elem);
}

matrix_t relu_derivative(matrix_t z)
{
    return apply_matrix(z, relu_derivative_elem);
}

static float leaky_relu_elem(float value)
{
    if (value > 0)
        return value;
    return value * 0.01;
}

static float leaky_relu_derivative_elem(float value)
{
    if (value > 0)
        return 1;
    return 0.01;
}

matrix_t leaky_relu(matrix_t z)
{
    return apply_matrix(z, leaky_relu_elem);
}

matrix_t leaky_relu_derivative(matrix_t z)
{
    return apply_matrix(z, leaky_relu_derivative_elem);
}

static float tanh_elem(float value)
{
    float x1 = expf(value);
    float x2 = expf(- value);
    return (x1 - x2) / (x1 + x2); // tanh(x)
}

static float tanh_der_elem(float value)
{
    float x = tanh_elem(value);
    return 1 - x * x; // 1 - tanh^2(x)
}

matrix_t tanh_activation(matrix_t z)
{
    return apply_matrix(z, tanh_elem);
}

matrix_t tanh_derivative(matrix_t z)
{
    return apply_matrix(z, tanh_der_elem);
}

static float sigmoid_elem(float value)
{
    return 1.0f / (1.0f + expf(- value));
}

static float sigmoid_der_elem(float value)
{
    float x = sigmoid_elem(value);
    return x * (1 - x);
}

matrix_t sigmoid(matrix_t z)
{
    return apply_matrix(z, sigmoid_elem);
}

matrix_t sigmoid_derivative(matrix_t z)
{
    return apply_matrix(z, sigmoid_der_elem);
}

matrix_t softmax(matrix_t z)
{
    matrix_t rez = create_matrix(z->rows, z->cols);
    for (int i = 0; i < z->rows; i++) {
        float max = z->data[i * z->cols]; // first element of each column
        
        for (int j = 1; j < z->cols; j++) {
            if (max < z->data[i * z->cols + j])
                max = z->data[i * z->cols + j]; // searches for each row
        }

        float sum = 0.0f;
        for (int j = 0; j < z->cols; j++) {
            rez->data[i * rez->cols + j] = 
                expf(z->data[i * z->cols + j] - max);
            // writes in the created matrix, rez, the exponentiated values

            // creates the sum
            sum += rez->data[i * rez->cols + j];
        }
        for (int j = 0; j < rez->cols; j++)
            rez->data[i * rez->cols + j] /= sum; 
        // makes the final division on each elemment on the row
    }
    return rez;
}

matrix_t identity(matrix_t z)
{
    matrix_t id = create_matrix(z->rows, z->cols);
    fill_matrix(id, 1.0f); // returns a matrix full of 1.0f
    return id;
}