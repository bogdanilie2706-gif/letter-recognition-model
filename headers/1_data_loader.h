#include <stdio.h>
#include <stdlib.h>

typedef struct idx1 {
	int nr_labels;
	char *labels[2];
	// labels[0] = index | labels[1] = value
} idx1, *idx1_t;

typedef struct idx3 {
	int nr_images;
	char **images;
} idx3, *idx3_t;


idx3_t load_idx3_file(FILE *idx);
