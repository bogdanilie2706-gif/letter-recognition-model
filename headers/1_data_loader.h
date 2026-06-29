#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <stdio.h>
#include <stdlib.h>

typedef struct dataset {
	int nr_samples;
	char *labels;
	char **images;
	// images[i] = image
} dataset, *dataset_t;

// loads a idx3 file in dataset_t structure
int load_idx3_file(FILE *idx, dataset_t data);

// loads a idx1 file in a dataset_t structure
int load_idx1_file(FILE *idx, dataset_t data);

//prints the image at image_index from data
void print_image(dataset_t data, int image_index);

// frees the dataset
void destroy_dataset(dataset_t *data);

// loads the entire dataset with the labels and the images
int load_dataset(dataset_t data, char *idx1_filename, char *idx3_filename);

#endif 

