#include "lenet.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <mpi.h> 

#define FILE_TRAIN_IMAGE "train-images-idx3-ubyte"
#define FILE_TRAIN_LABEL "train-labels-idx1-ubyte"
#define FILE_TEST_IMAGE "t10k-images-idx3-ubyte"
#define FILE_TEST_LABEL "t10k-labels-idx1-ubyte"
#define LENET_FILE "model.dat"
#define COUNT_TRAIN 60000
#define COUNT_TEST 10000

// Load the model from a file
int load(LeNet5 *lenet, const char *filename)
{
    FILE *file = fopen(filename, "rb");
    if (!file)
    {
        return 1; // File could not be opened
    }

    // Read weights and biases for all layers
    if (fread(lenet, sizeof(LeNet5), 1, file) != 1)
    {
        fclose(file);
        return 1; // Failed to read the file correctly
    }

    fclose(file);
    return 0; // Successfully loaded the model
}


int read_data(unsigned char (*data)[28][28], unsigned char label[], const int count, const char data_file[], const char label_file[])
{
	FILE *fp_image = fopen(data_file, "rb");
	FILE *fp_label = fopen(label_file, "rb");
	if (!fp_image || !fp_label)
		return 1;
	fseek(fp_image, 16, SEEK_SET);
	fseek(fp_label, 8, SEEK_SET);
	fread(data, sizeof(*data) * count, 1, fp_image);
	fread(label, count, 1, fp_label);
	fclose(fp_image);
	fclose(fp_label);
	return 0;
}

void training(LeNet5 *lenet, image *train_data, uint8 *train_label, int batch_size, int total_size, int rank, int num_procs)
{
	int start = rank * (total_size / num_procs);
	int end = (rank == num_procs - 1) ? total_size : start + (total_size / num_procs);

	for (int i = start; i < end; i += batch_size)
	{
		TrainBatch(lenet, train_data + i, train_label + i, batch_size);
	}
	MPI_Allreduce(MPI_IN_PLACE, lenet, sizeof(LeNet5) / sizeof(double), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
}

int testing(LeNet5 *lenet, image *test_data, uint8 *test_label, int total_size, int rank, int num_procs)
{
	int start = rank * (total_size / num_procs);
	int end = (rank == num_procs - 1) ? total_size : start + (total_size / num_procs);
	int local_right = 0;

	for (int i = start; i < end; ++i)
	{
		uint8 l = test_label[i];
		int p = Predict(lenet, test_data[i], 10);
		local_right += (l == p);
	}

	int total_right = 0;
	MPI_Reduce(&local_right, &total_right, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	return total_right;
}

int save(LeNet5 *lenet, char filename[])
{
	FILE *fp = fopen(filename, "wb");
	if (!fp)
		return 1;
	fwrite(lenet, sizeof(LeNet5), 1, fp);
	fclose(fp);
	return 0;
}

int main(int argc, char *argv[])
{
	MPI_Init(&argc, &argv);

	int rank, num_procs;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

	image *train_data = (image *)calloc(COUNT_TRAIN, sizeof(image));
	uint8 *train_label = (uint8 *)calloc(COUNT_TRAIN, sizeof(uint8));
	image *test_data = (image *)calloc(COUNT_TEST, sizeof(image));
	uint8 *test_label = (uint8 *)calloc(COUNT_TEST, sizeof(uint8));

	if (rank == 0)
	{
		if (read_data(train_data, train_label, COUNT_TRAIN, FILE_TRAIN_IMAGE, FILE_TRAIN_LABEL))
		{
			printf("ERROR: Dataset not found!\n");
			MPI_Abort(MPI_COMM_WORLD, 1);
		}

		if (read_data(test_data, test_label, COUNT_TEST, FILE_TEST_IMAGE, FILE_TEST_LABEL))
		{
			printf("ERROR: Dataset not found!\n");
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
	}

	MPI_Bcast(train_data, COUNT_TRAIN * sizeof(image), MPI_BYTE, 0, MPI_COMM_WORLD);
	MPI_Bcast(train_label, COUNT_TRAIN * sizeof(uint8), MPI_BYTE, 0, MPI_COMM_WORLD);
	MPI_Bcast(test_data, COUNT_TEST * sizeof(image), MPI_BYTE, 0, MPI_COMM_WORLD);
	MPI_Bcast(test_label, COUNT_TEST * sizeof(uint8), MPI_BYTE, 0, MPI_COMM_WORLD);

	LeNet5 *lenet = (LeNet5 *)malloc(sizeof(LeNet5));
	if (rank == 0 && load(lenet, LENET_FILE))
		Initial(lenet);

	MPI_Bcast(lenet, sizeof(LeNet5) / sizeof(double), MPI_DOUBLE, 0, MPI_COMM_WORLD);

	clock_t start = clock();
	int batches[] = {300};

	// for (int i = 0; i < sizeof(batches) / sizeof(*batches); ++i)
	// 	training(lenet, train_data, train_label, batches[i], COUNT_TRAIN, rank, num_procs);

	int right = testing(lenet, test_data, test_label, COUNT_TEST, rank, num_procs);

	if (rank == 0)
	{
		printf("%d/%d\n", right, COUNT_TEST);
		printf("Time:%u\n", (unsigned)(clock() - start));
		save(lenet, LENET_FILE);
	}

	free(lenet);
	free(train_data);
	free(train_label);
	free(test_data);
	free(test_label);

	MPI_Finalize();
	return 0;
}
