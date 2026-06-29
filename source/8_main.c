#include "../headers/8_main.h"

int main(void)
{
    srand(42);
    dataset_t test_data = malloc(sizeof(dataset));
    dataset_t train_data = malloc(sizeof(dataset));
    if (!test_data || !train_data) { // checking if the mallocs went well
        perror("reading test didn't work");
        return 1;
    }
    
    int test_ret = load_dataset(test_data, "data/test_labels_idx1",
        "data/test_images_idx3");
    int train_ret = load_dataset(train_data, "data/train_labels_idx1",
        "data/train_images_idx3");

    if (test_ret || train_ret) {
        perror("error loading data");
        destroy_dataset(&train_data);
        destroy_dataset(&test_data);
    }

    print_image(test_data, 21);
    print_image(train_data, 35);

    destroy_dataset(&test_data);
    destroy_dataset(&train_data);
    return 0;
}
