#include <stdio.h>
#include <stdlib.h>

typedef struct dataset {
	int nr_samples;
	char *labels;
	char **images;
	// images[i] = image
} dataset, *dataset_t;


int load_idx3_file(FILE *idx, dataset_t data);
int load_idx1_file(FILE *idx, dataset_t data);
void print_image(dataset_t data, int image_index);
void destroy_dataset(dataset_t *data);
int load_dataset(dataset_t data, char *idx1_filename, char *idx3_filename);


