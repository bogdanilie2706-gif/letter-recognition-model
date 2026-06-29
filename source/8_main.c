#include "../headers/8_main.h"

int main(void)
{
    FILE *test_images_file = fopen("data/emnist-letters-test-images-idx3-ubyte", "rb");
    FILE *test_labels_file = fopen("data/emnist-letters-test-labels-idx1-ubyte", "rb");
    FILE *train_images_file = fopen("data/emnist-letters-train-images-idx3-ubyte", "rb");
    FILE *train_labels_file = fopen("data/emnist-letters-train-labels-idx1-ubyte", "rb");
    if(!test_images_file || !test_labels_file ||
        !train_images_file || !train_labels_file) { // 

        perror("couldn't open the data files");
        return 1;
    }

    dataset_t test_data = malloc(sizeof(dataset));
    dataset_t train_data = malloc(sizeof(dataset));
    if (!test_data || !train_data) { // checking if the reading went well
        fclose(test_images_file);
        fclose(test_labels_file);
        fclose(train_images_file);
        fclose(train_labels_file);
        perror("reading test didn't work");
        return 1;
    }
    
    int test_idx3_ret = load_idx3_file(test_images_file, test_data);
    int test_idx1_ret = load_idx1_file(test_labels_file, test_data);

    int train_idx3_ret = load_idx3_file(train_images_file, train_data);
    int train_idx1_ret = load_idx1_file(train_labels_file, train_data);

    if (test_idx1_ret || test_idx3_ret || train_idx1_ret || train_idx3_ret) { // checking if the loading went well
        fclose(test_images_file);
        fclose(test_labels_file);
        fclose(train_images_file);
        fclose(train_labels_file);
        destroy_dataset(&test_data);
        destroy_dataset(&train_data);
        perror("Couldn't load data");
    }


    print_image(test_data, 21);
    print_image(train_data, 35);

    fclose(test_images_file);
    fclose(test_labels_file);
    fclose(train_images_file);
    fclose(train_labels_file);
    destroy_dataset(&test_data);
    destroy_dataset(&train_data);
    return 0;
}
