#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int min = 1, max = 10000000;

void bubble_sort(int arr[], int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j + 1] < arr[j]) {
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}

void selection_sort(int arr[], int n) {
    for (int i = 0; i < n - 1; i++) {
        int min_idx = i;
        for (int j = i + 1; j < n; j++) {
            if (arr[j] < arr[min_idx]) {
                min_idx = j;
            }
        }
        int temp = arr[min_idx];
        arr[min_idx] = arr[i];
        arr[i] = temp;
    }
}

void insertion_sort(int arr[], int n) {
    for (int i = 1; i < n; i++) {
        int key = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

void generate_random_array(int arr[], int n) {
    srand(time(0));
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % (max - min + 1) + min;
    }
}

void output_time_data(FILE *time_data, const char *algorithm_name, int n, double time_used) {
    fprintf(time_data, "%d %f %s\n", n, time_used, algorithm_name);
}

int main() {
    clock_t start, end;
    double time_used;

    FILE *time_data = fopen("time_data.txt", "a");
    if (time_data == NULL) {
        printf("Error opening file.\n");
        return 1;
    }

    int n;
    printf("Enter number of elements in the array: ");
    scanf("%d", &n);
    
    int *arr = malloc(n * sizeof(int));
    int *copy = malloc(n * sizeof(int));
    
    if (arr == NULL || copy == NULL) {
        printf("Memory allocation failed.\n");
        return 1;
    }

    generate_random_array(arr, n);
    for (int i = 0; i < n; i++) {
        copy[i] = arr[i];
    }

    // Bubble Sort
    start = clock();
    bubble_sort(arr, n);
    end = clock();
    time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Bubble Sort Time: %f seconds\n", time_used);
    output_time_data(time_data, "bubble", n, time_used);

    // Selection Sort
    for (int i = 0; i < n; i++) {
        arr[i] = copy[i];
    }
    start = clock();
    selection_sort(arr, n);
    end = clock();
    time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Selection Sort Time: %f seconds\n", time_used);
    output_time_data(time_data, "selection", n, time_used);

    // Insertion Sort
    for (int i = 0; i < n; i++) {
        arr[i] = copy[i];
    }
    start = clock();
    insertion_sort(arr, n);
    end = clock();
    time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Insertion Sort Time: %f seconds\n", time_used);
    output_time_data(time_data, "insertion", n, time_used);

    fclose(time_data);
    free(arr);
    free(copy);

    return 0;
}
