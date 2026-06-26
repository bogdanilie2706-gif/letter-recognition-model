#include "../headers/8_main.h"

int main(void)
{
    FILE *test_file = fopen("data/emnist-letters-test-images-idx3-ubyte", "rb");
    if(!test_file) {
        perror("couldn't open file");
        return 1;
    }
    idx3_t test_data = load_idx3_file(test_file);
    if (!test_data) {
        printf("nu a mers\n");
        return 0;
    }
    printf("nr images = %d\n", test_data->nr_images);
    for (int i = 0; i < 28; i++) {
    for (int j = 0; j < 28; j++)
        printf("%3hhu ", test_data->images[0][j * 28 + i]); // image is transposed
    printf("\n");
}
    return 0;
}
