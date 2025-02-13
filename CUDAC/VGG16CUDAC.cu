#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <ctype.h>
#include <string.h>
#include <cuda_runtime.h>

#ifdef _WIN32
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif

void gettimeofday(time_t *tp, char *_)
{
	*tp = clock();
	return;
}

double get_seconds(time_t timeStart, time_t timeEnd)
{
	return (double)(timeEnd - timeStart) / CLOCKS_PER_SEC;
}
#endif

#define SIZE 224 //size of image
#define CONV_SIZE 3 // size of kernel, in out case it will be 3x3


float ***image; //Pointer to the image and it is 3d since it is RGB image

float *****wc; // weights array for convolution layer: index 0 for level, index 1 for number of output channels, index 2 for number of input channels and then last two 

float **bc; //bias array for conv layer: index 0 for level number and index 1 for bias at that level

float ***wd;  // weights array for convolution layer: index 0 for level, index 1 for number of input channels, index 2 for number of output channels

float **bd;  // bias array for dense layer: index 0 for level number and index 1 for bias at that level

int mem_block_shape[3] = {512, SIZE, SIZE};
float ***mem_block1;
float ***mem_block2;

//mem_block1 and mem_block2 are used interchangabley as input and output to the conv layer. So each one can represent an input to the conv and the output will be saved into the other

int mem_block_dense_shape = { 512 * 7 * 7 };
float *mem_block1_dense;
float *mem_block2_dense;

//mem_block1_dense and mem_block2_dense are used interchangabley as input and output to the conv layer. So each one can represent an input to the conv and the output will be saved into the other

int cshape[13][4] = { // this array is used by the convolution layers to determine at each layer what is the number of the input and output filters, and the size of kernel (3x3) 
	{ 64, 3, CONV_SIZE, CONV_SIZE },
	{ 64, 64, CONV_SIZE, CONV_SIZE },
	{ 128, 64, CONV_SIZE, CONV_SIZE },
	{ 128, 128, CONV_SIZE, CONV_SIZE },
	{ 256, 128, CONV_SIZE, CONV_SIZE },
	{ 256, 256, CONV_SIZE, CONV_SIZE },
	{ 256, 256, CONV_SIZE, CONV_SIZE },
	{ 512, 256, CONV_SIZE, CONV_SIZE },
	{ 512, 512, CONV_SIZE, CONV_SIZE },
	{ 512, 512, CONV_SIZE, CONV_SIZE },
	{ 512, 512, CONV_SIZE, CONV_SIZE },
	{ 512, 512, CONV_SIZE, CONV_SIZE },
	{ 512, 512, CONV_SIZE, CONV_SIZE }
};
int dshape[3][2] = { // this array is used by the convolution layers to determine at each layer what is the number of the inout and output filters, and the size of kernel (3x3) 
	{ 25088, 4096 },
	{ 4096, 4096 },
	{ 4096, 1000 }
};

void reset_mem_block(float ***mem) //initilaize the 3d convolution memory block to zeros
{
	int i, j, k;
	for (i = 0; i < mem_block_shape[0]; i++)
	{
		for (j = 0; j < mem_block_shape[1]; j++)
		{
			for (k = 0; k < mem_block_shape[2]; k++)
			{
				mem[i][j][k] = 0.0;
			}
		}
	}
}

void reset_mem_block_dense(float *mem) //initilaize the 1d dense memory block to zeros
{
	int i;
	for (i = 0; i < mem_block_dense_shape; i++)
	{
		mem[i] = 0.0;
	}
}


void init_memory() { //allocate memory to all arrays 
    int i, j, k, l;

    // Init image memory
    image = (float***)malloc(3 * sizeof(float**));
    for (i = 0; i < 3; i++) {
        image[i] = (float**)malloc(SIZE * sizeof(float*));
        for (j = 0; j < SIZE; j++) {
            image[i][j] = (float*)malloc(SIZE * sizeof(float));
        }
    }

    // Init convolution weights
    wc = (float*****)malloc(13 * sizeof(float****));
    bc = (float**)malloc(13 * sizeof(float*));
    for (l = 0; l < 13; l++) {
        wc[l] = (float****)malloc(cshape[l][0] * sizeof(float***));
        for (i = 0; i < cshape[l][0]; i++) {
            wc[l][i] = (float***)malloc(cshape[l][1] * sizeof(float**));
            for (j = 0; j < cshape[l][1]; j++) {
                wc[l][i][j] = (float**)malloc(cshape[l][2] * sizeof(float*));
                for (k = 0; k < cshape[l][2]; k++) {
                    wc[l][i][j][k] = (float*)malloc(cshape[l][3] * sizeof(float));
                }
            }
        }
        bc[l] = (float*)malloc(cshape[l][0] * sizeof(float));
    }

    // Init dense weights
    wd = (float***)malloc(3 * sizeof(float**));
    bd = (float**)malloc(3 * sizeof(float*));
    for (l = 0; l < 3; l++) {
        wd[l] = (float**)malloc(dshape[l][0] * sizeof(float*));
        for (i = 0; i < dshape[l][0]; i++) {
            wd[l][i] = (float*)malloc(dshape[l][1] * sizeof(float));
        }
        bd[l] = (float*)malloc(dshape[l][1] * sizeof(float));
    }

    // Init mem_blocks
	mem_block1 = (float***)malloc(mem_block_shape[0] * sizeof(float**));
    mem_block2 = (float***)malloc(mem_block_shape[0] * sizeof(float**));
    for (i = 0; i < mem_block_shape[0]; i++) {
        mem_block1[i] = (float**)malloc(mem_block_shape[1] * sizeof(float*));
        mem_block2[i] = (float**)malloc(mem_block_shape[1] * sizeof(float*));
        for (j = 0; j < mem_block_shape[1]; j++) {
            mem_block1[i][j] = (float*)malloc(mem_block_shape[2] * sizeof(float));
            mem_block2[i][j] = (float*)malloc(mem_block_shape[2] * sizeof(float));
        }

    }
    reset_mem_block(mem_block1);
    reset_mem_block(mem_block2);

    // Init mem blocks dense
    mem_block1_dense = (float*)calloc(mem_block_dense_shape, sizeof(float));
    mem_block2_dense = (float*)calloc(mem_block_dense_shape, sizeof(float));
}


void free_memory() { // free allocated memory for arrays
	int i, j, k, l;

	// Free image memory
	for (i = 0; i < 3; i++) {
		for (j = 0; j < SIZE; j++) {
			free(image[i][j]);
		}
		free(image[i]);
	}
	free(image);

	// Free convolution weights
	for (l = 0; l < 13; l++) {
		for (i = 0; i < cshape[l][0]; i++) {
			for (j = 0; j < cshape[l][1]; j++) {
				for (k = 0; k < cshape[l][2]; k++) {
					free(wc[l][i][j][k]);
				}
				free(wc[l][i][j]);
			}
			free(wc[l][i]);
		}
		free(wc[l]);
		free(bc[l]);
	}
	free(wc);
	free(bc);

	// Free dense weights
	for (l = 0; l < 3; l++) {
		for (i = 0; i < dshape[l][0]; i++) {
			free(wd[l][i]);
		}
		free(wd[l]);
		free(bd[l]);
	}
	free(wd);
	free(bd);

	// Free memblocks
	for (i = 0; i < mem_block_shape[0]; i++) {
		for (j = 0; j < mem_block_shape[1]; j++) {
			free(mem_block1[i][j]);
			free(mem_block2[i][j]);
		}
		free(mem_block1[i]);
		free(mem_block2[i]);
	}
	free(mem_block1);
	free(mem_block2);

	free(mem_block1_dense);
	free(mem_block2_dense);
}

void read_weights(char *in_file, int lvls) { // reading weigthts and biases
	float dval;
	int i, j, k, l, z;
	FILE *iin;
	int total_lvls_read = 0;

	iin = fopen(in_file, "r");
	if (iin == NULL) {
		printf("File %s absent\n", in_file);
		exit(1);
	}
	printf("Reading the weigths and bias of the convolutional layers\n");
	for (z = 0; z < 13; z++) {
		if (total_lvls_read >= lvls && lvls != -1)
			break;
		for (i = 0; i < cshape[z][0]; i++) {
			for (j = 0; j < cshape[z][1]; j++) {
				for (k = 0; k < cshape[z][2]; k++) {
					for (l = 0; l < cshape[z][3]; l++) {
						fscanf(iin, "%f", &dval);
						wc[z][i][j][CONV_SIZE - k - 1][CONV_SIZE - l - 1] = dval;
					}
				}
			}
		}
		for (i = 0; i < cshape[z][0]; i++) {
			fscanf(iin, "%f", &dval);
			bc[z][i] = dval;
		}
		total_lvls_read += 1;
	}
	printf("Reading the weigths and bias of the fully connected layers\n");
	for (z = 0; z < 3; z++) {
		if (total_lvls_read >= lvls && lvls != -1)
			break;
		for (i = 0; i < dshape[z][0]; i++) {
			for (j = 0; j < dshape[z][1]; j++) {
				fscanf(iin, "%f", &dval);
				wd[z][i][j] = dval;
			}
		}
		for (i = 0; i < dshape[z][1]; i++) {
			fscanf(iin, "%f", &dval);
			bd[z][i] = dval;
		}
		total_lvls_read += 1;
	}

	fclose(iin);
}

float*** read_image(const char *in_file) { //reading the image
    int i, j, l;
    FILE *iin;
    float dval;
    int items_read;

    iin = fopen(in_file, "r");
    if (iin == NULL) {
        printf("File %s absent\n", in_file);
        exit(1);
    }

    for (i = 0; i < SIZE; i++) {
        for (j = 0; j < SIZE; j++) {
            for (l = 0; l < 3; l++) {
                items_read = fscanf(iin, "%f", &dval);
                if (items_read != 1) {
                    printf("Error reading value at position [%d][%d][%d] from file %s\n", i, j, l, in_file);
                    fclose(iin);
                    exit(1);
                }

                if (dval < 0.0f || dval > 255.0f) {
                    printf("Invalid pixel value %f at position [%d][%d][%d] in file %s\n", dval, i, j, l, in_file);
                    fclose(iin);
                    exit(1);
                }

                image[l][i][j] = dval;
            }
        }
    }

    fclose(iin);
	return image;
}
float*** normalize_image(float*** image) { //normalizing the image 
	int i, j, l;
	float coef[3] = { 103.939, 116.779, 123.68 };

	for (l = 0; l < 3; l++) {
		for (i = 0; i < SIZE; i++) {
			for (j = 0; j < SIZE; j++) {
				image[l][i][j] -= coef[l];
			}
		}
	}
	return image;
}


void add_bias_and_relu(float **out, float bs, int size) { // function to add bias and apply non-linear function to the numbers in the array
	int i, j;

	for (i = 0; i < size; i++) {
		for (j = 0; j < size; j++) {
			out[i][j] += bs;
			if (out[i][j] < 0)
				out[i][j] = 0.0;
			// printf("%.12lf\n", out[i][j]);
		}
	}
}


void add_bias_and_relu_flatten(float *out, float *bs, int size, int relu) { // function to add bias and apply non-linear function to the numbers in the array used in dense layers
	int i;
	for (i = 0; i < size; i++) {
		out[i] += bs[i];
		if (relu == 1) {
			if (out[i] < 0)
				out[i] = 0.0;
		}
	}
}




__device__ float max_of_4(float a, float b, float c, float d) {
    return fmaxf(fmaxf(a, b), fmaxf(c, d));
}

__global__ void maxpooling_kernel(float *in_out, int size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; 
    int y = blockIdx.y * blockDim.y + threadIdx.y; 

    int output_size = size / 2; 

    if (x < output_size && y < output_size) {

        int in_x = x * 2;
        int in_y = y * 2;

        in_out[y * output_size + x] = max_of_4(
            in_out[in_y * size + in_x],          
            in_out[(in_y + 1) * size + in_x],    
            in_out[in_y * size + in_x + 1],      
            in_out[(in_y + 1) * size + in_x + 1] 
        );
    }
}

void maxpooling(float **in_out, int size) {
    int output_size = size / 2;

    float *d_in_out;
    cudaMalloc((void **)&d_in_out, size * size * sizeof(float));

    cudaMemcpy(d_in_out, *in_out, size * size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(24, 24); 
    dim3 gridDim((output_size + blockDim.x - 1) / blockDim.x,
                 (output_size + blockDim.y - 1) / blockDim.y);

    maxpooling_kernel<<<gridDim, blockDim>>>(d_in_out, size);

    cudaMemcpy(*in_out, d_in_out, output_size * output_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_in_out);
}



void flatten(float ***in, float *out, int sh0, int sh1, int sh2) {
	int i, j, k, total = 0;
	for (i = 0; i < sh0; i++) {
		for (j = 0; j < sh1; j++) {
			for (k = 0; k < sh2; k++) {
				out[total] = in[i][j][k];
				total += 1;
			}
		}
	}
}


__global__ void dense_kernel(float *in, float *weights, float *out, int sh_in, int sh_out) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < sh_out) { 
        float sum = 0.0f;
        for (int j = 0; j < sh_in; j++) {
            sum += in[j] * weights[j * sh_out + i];
        }
        out[i] = sum;
    }
}

void dense(float *in, float **weights, float *out, int sh_in, int sh_out) {

    float *d_in, *d_weights, *d_out;
	
    cudaMalloc(&d_in, sh_in * sizeof(float));
    cudaMalloc(&d_weights, sh_in * sh_out * sizeof(float));
    cudaMalloc(&d_out, sh_out * sizeof(float));

    float *weights_flat = (float *)malloc(sh_in * sh_out * sizeof(float));
    for (int i = 0; i < sh_in; i++) {
        for (int j = 0; j < sh_out; j++) {
            weights_flat[i * sh_out + j] = weights[i][j];
        }
    }

    cudaMemcpy(d_in, in, sh_in * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights_flat, sh_in * sh_out * sizeof(float), cudaMemcpyHostToDevice);

    int threads_per_block = 16*16; // we used 256 blocks in total
    int num_blocks = (sh_out + threads_per_block - 1) / threads_per_block; // each block will have threads

    dense_kernel<<<num_blocks, threads_per_block>>>(d_in, d_weights, d_out, sh_in, sh_out);

    cudaMemcpy(out, d_out, sh_out * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_weights);
    cudaFree(d_out);

    free(weights_flat);
}




void softmax(float *out, int sh_out) {
	int i;
	float max_val, sum;
	max_val = out[0];
	for (i = 1; i < sh_out; i++) {
		if (out[i] > max_val)
			max_val = out[i];
	}
	sum = 0.0;
	for (i = 0; i < sh_out; i++) {
		out[i] = exp(out[i] - max_val);
		sum += out[i];
	}
	for (i = 0; i < sh_out; i++) {
		out[i] /= sum;
	}
}


// Flatten a 2D matrix to a 1D array
float* flatten_2d_to_1d(float **matrix, int size) {
    float *flat_matrix = (float*)malloc(size * size * sizeof(float));
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            flat_matrix[i * size + j] = matrix[i][j];  // Convert 2D index to 1D
        }
    }
    return flat_matrix;
}
// Flatten a 3x3 kernel to a 1D array
float* flatten_kernel_2d_to_1d(float **kernel) {
    float *flat_kernel = (float*)malloc(9 * sizeof(float));  // 3x3 kernel -> 9 elements
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            flat_kernel[i * 3 + j] = kernel[i][j];  // Convert 2D kernel index to 1D
        }
    }
    return flat_kernel;
}

__global__ void convolution_3_x_3_kernel(float *d_matrix, float *d_kernel, float *d_out, int size) {

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
	#define BLOCK_SIZE 16

    __shared__ float tile[BLOCK_SIZE + 2][BLOCK_SIZE + 2];

    if (row < size && col < size) {
        tile[ty + 1][tx + 1] = d_matrix[row * size + col];
  
        if (tx == 0 && col > 0) 
            tile[ty + 1][0] = d_matrix[row * size + (col - 1)];
        if (tx == blockDim.x - 1 && col < size - 1)
            tile[ty + 1][BLOCK_SIZE + 1] = d_matrix[row * size + (col + 1)];
        if (ty == 0 && row > 0)
            tile[0][tx + 1] = d_matrix[(row - 1) * size + col];
        if (ty == blockDim.y - 1 && row < size - 1)
            tile[BLOCK_SIZE + 1][tx + 1] = d_matrix[(row + 1) * size + col];

        if (tx == 0 && ty == 0 && row > 0 && col > 0)
            tile[0][0] = d_matrix[(row - 1) * size + (col - 1)];
        if (tx == blockDim.x - 1 && ty == 0 && row > 0 && col < size - 1)
            tile[0][BLOCK_SIZE + 1] = d_matrix[(row - 1) * size + (col + 1)];
        if (tx == 0 && ty == blockDim.y - 1 && row < size - 1 && col > 0)
            tile[BLOCK_SIZE + 1][0] = d_matrix[(row + 1) * size + (col - 1)];
        if (tx == blockDim.x - 1 && ty == blockDim.y - 1 && row < size - 1 && col < size - 1)
            tile[BLOCK_SIZE + 1][BLOCK_SIZE + 1] = d_matrix[(row + 1) * size + (col + 1)];
    }

    __syncthreads();

    if (row < size && col < size) {
        float sum = 0.0f;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                sum += tile[ty + i][tx + j] * d_kernel[i * 3 + j];
            }
        }
        d_out[row * size + col] = sum;
    }
}

void convolution_3_x_3_cuda(float **matrix, float **kernel, int size,int i,int j) {
    float *d_matrix, *d_kernel, *d_out;
    
	cudaMalloc(&d_matrix, size * size * sizeof(float));    
    cudaMalloc(&d_kernel, 9 * sizeof(float));              
    cudaMalloc(&d_out, size * size * sizeof(float));       

    float *matrix_flat = flatten_2d_to_1d(matrix, size);
    float *kernel_flat = flatten_2d_to_1d(kernel, 3);

    cudaMemcpy(d_matrix, matrix_flat, size * size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel_flat, 9 * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(24, 24);
    dim3 gridDim((size + blockDim.x - 1) / blockDim.x, (size + blockDim.y - 1) / blockDim.y);


    convolution_3_x_3_kernel<<<gridDim, blockDim>>>(d_matrix, d_kernel, d_out, size);


	cudaDeviceSynchronize();


	
	if(j==0){
		float *mem_block1_flat = flatten_2d_to_1d(mem_block2[i], size);
		cudaMemcpy(mem_block1_flat, d_out, size * size * sizeof(float), cudaMemcpyDeviceToHost);

		for (int row = 0; row < size; row++) {
			for (int col = 0; col < size; col++) {
				mem_block1[i][row][col] = mem_block1_flat[row * size + col];
			}
		}
		
	free(mem_block1_flat);
	}else{

		float *mem_block2_flat = flatten_2d_to_1d(mem_block2[i], size);
		cudaMemcpy(mem_block2_flat, d_out, size * size * sizeof(float), cudaMemcpyDeviceToHost);
		
		for (int row = 0; row < size; row++) {
			for (int col = 0; col < size; col++) {
				mem_block2[i][row][col] = mem_block2_flat[row * size + col];
				
			}
		}
		
	free(mem_block2_flat);
	}


    cudaFree(d_matrix);
    cudaFree(d_kernel);
    cudaFree(d_out);

    free(matrix_flat);
    free(kernel_flat);
}



void predict_image_VGG(float*** image,float***** wc,float** bc) { //This function does the needed work. It calls conv, dense, relu and pooling functions
	int i, j;
	int level, cur_size;
	int cshape[13][4] = { 
	{ 64, 3, CONV_SIZE, CONV_SIZE },
	{ 64, 64, CONV_SIZE, CONV_SIZE },
	{ 128, 64, CONV_SIZE, CONV_SIZE },
	{ 128, 128, CONV_SIZE, CONV_SIZE },
	{ 256, 128, CONV_SIZE, CONV_SIZE },
	{ 256, 256, CONV_SIZE, CONV_SIZE },
	{ 256, 256, CONV_SIZE, CONV_SIZE },
	{ 512, 256, CONV_SIZE, CONV_SIZE },
	{ 512, 512, CONV_SIZE, CONV_SIZE },
	{ 512, 512, CONV_SIZE, CONV_SIZE },
	{ 512, 512, CONV_SIZE, CONV_SIZE },
	{ 512, 512, CONV_SIZE, CONV_SIZE },
	{ 512, 512, CONV_SIZE, CONV_SIZE }

	};

	reset_mem_block(mem_block1);
	reset_mem_block(mem_block2);
	reset_mem_block_dense(mem_block1_dense);
	reset_mem_block_dense(mem_block2_dense);

	// Layer 1 (Convolution 3 -> 64)
	level = 0;
	cur_size = SIZE;
	for (i = 0; i < cshape[level][0]; i++) {
		for (j = 0; j < cshape[level][1]; j++) {
			convolution_3_x_3_cuda(image[j], wc[level][i][j], cur_size,i,0);
		}
		add_bias_and_relu(mem_block1[i], bc[level][i], cur_size);
	}
	printf("Convolution Layer 1 Done!\n");

	level = 1;
	for (i = 0; i < cshape[level][0]; i++) {
		for (j = 0; j < cshape[level][1]; j++) {
			convolution_3_x_3_cuda(mem_block1[j], wc[level][i][j], cur_size,i,1);
		}

		add_bias_and_relu(mem_block2[i], bc[level][i], cur_size);
	}
	reset_mem_block(mem_block1);
	printf("Convolution Layer 2 Done!\n");
	
	// Layer 3 (MaxPooling)
	for (i = 0; i < cshape[level][0]; i++) {
		maxpooling(mem_block2[i], cur_size);
	}
	float bias = 0.011;
	cur_size /= 2;
	printf("Max Pooling Done!\n");
	
	// Layer 4 (Convolution 64 -> 128)
	level = 2;
	for (i = 0; i < cshape[level][0]; i++) {
		for (j = 0; j < cshape[level][1]; j++) {
			convolution_3_x_3_cuda(mem_block2[j], wc[level][i][j], cur_size,i,0);
		}
		add_bias_and_relu(mem_block1[i], bc[level][i], cur_size);
	}
	reset_mem_block(mem_block2);
	printf("Convolution Layer 3 Done!\n");

	level = 3;
	for (i = 0; i < cshape[level][0]; i++) {
		for (j = 0; j < cshape[level][1]; j++) {
			convolution_3_x_3_cuda(mem_block1[j], wc[level][i][j], cur_size,i,1);
		}
		add_bias_and_relu(mem_block2[i], bc[level][i], cur_size);
	}
	printf("Convolution Layer 4 Done!\n");
	
	reset_mem_block(mem_block1);
	
	// Layer 6 (MaxPooling)
	for (i = 0; i < cshape[level][0]; i++) {
		maxpooling(mem_block2[i], cur_size);
	}
	printf("Max Pooling Done!\n");


	cur_size /= 2;
	int norm= 103*5;

	// Layer 7 (Convolution 128 -> 256)
	level = 4;
	for (i = 0; i < cshape[level][0]; i++) {
		for (j = 0; j < cshape[level][1]; j++) {
			convolution_3_x_3_cuda(mem_block2[j], wc[level][i][j], cur_size,i,0);
		}
		add_bias_and_relu(mem_block1[i], bc[level][i], cur_size);
	}
	reset_mem_block(mem_block2);
	printf("Convolution Layer 5 Done!\n");


	// Layer 8 (Convolution 256 -> 256)
	level = 5;
	for (i = 0; i < cshape[level][0]; i++) {
		for (j = 0; j < cshape[level][1]; j++) {
			convolution_3_x_3_cuda(mem_block1[j], wc[level][i][j],  cur_size,i,1);
		}
		add_bias_and_relu(mem_block2[i], bc[level][i], cur_size);
	}
	reset_mem_block(mem_block1);
	printf("Convolution Layer 6 Done!\n");


	// Layer 9 (Convolution 256 -> 256)
	level = 6;
	for (i = 0; i < cshape[level][0]; i++) {
		for (j = 0; j < cshape[level][1]; j++) {
			convolution_3_x_3_cuda(mem_block2[j], wc[level][i][j], cur_size,i,0);
		}
		add_bias_and_relu(mem_block1[i], bc[level][i], cur_size);
	}
	reset_mem_block(mem_block2);
	printf("Convolution Layer 7 Done!\n");

	
	// Layer 10 (MaxPooling)
	for (i = 0; i < cshape[level][0]; i++) {
		maxpooling(mem_block1[i], cur_size);
	}
	cur_size /= 2;
	printf("Max Pooling Done!\n");

	
	// Layer 11 (Convolution 256 -> 512)
	level = 7;
	for (i = 0; i < cshape[level][0]; i++) {
		for (j = 0; j < cshape[level][1]; j++) {
			convolution_3_x_3_cuda(mem_block1[j], wc[level][i][j],  cur_size,i,1);
		}
		add_bias_and_relu(mem_block2[i], bc[level][i], cur_size);
	}
	reset_mem_block(mem_block1);
	printf("Convolution Layer 8 Done!\n");


	// Layer 12 (Convolution 512 -> 512)
	level = 8;
	for (i = 0; i < cshape[level][0]; i++) {
		for (j = 0; j < cshape[level][1]; j++) {
			convolution_3_x_3_cuda(mem_block2[j], wc[level][i][j], cur_size,i,0);
		}
		add_bias_and_relu(mem_block1[i], bc[level][i], cur_size);
	}
	reset_mem_block(mem_block2);
	printf("Convolution Layer 9 Done!\n");


	// Layer 13 (Convolution 512 -> 512)
	level = 9;
	for (i = 0; i < cshape[level][0]; i++) {
		for (j = 0; j < cshape[level][1]; j++) {
			convolution_3_x_3_cuda(mem_block1[j], wc[level][i][j],cur_size,i,1);
		}
		add_bias_and_relu(mem_block2[i], bc[level][i], cur_size);
	}
	reset_mem_block(mem_block1);
	printf("Convolution Layer 10 Done!\n");

	
	// Layer 14 (MaxPooling)
	for (i = 0; i < cshape[level][0]; i++) {
		maxpooling(mem_block2[i], cur_size);
	}
	printf("Max Pooling Done!\n");

	cur_size /= 2;
	
	// Layer 15 (Convolution 512 -> 512)
	level = 10;
	for (i = 0; i < cshape[level][0]; i++) {
		for (j = 0; j < cshape[level][1]; j++) {
			convolution_3_x_3_cuda(mem_block2[j], wc[level][i][j],  cur_size,i,0);
		}
		add_bias_and_relu(mem_block1[i], bc[level][i], cur_size);
	}
	reset_mem_block(mem_block2);
	printf("Convolution Layer 11 Done!\n");

	// Layer 16 (Convolution 512 -> 512)
	level = 11;
	for (i = 0; i < cshape[level][0]; i++) {
		for (j = 0; j < cshape[level][1]; j++) {
			convolution_3_x_3_cuda(mem_block1[j], wc[level][i][j],  cur_size,i,1);
		}
		add_bias_and_relu(mem_block2[i], bc[level][i], cur_size);
	}
	reset_mem_block(mem_block1);
	printf("Convolution Layer 12 Done!\n");


	// Layer 17 (Convolution 512 -> 512)
	level = 12;
	for (i = 0; i < cshape[level][0]; i++) {
		for (j = 0; j < cshape[level][1]; j++) {
			convolution_3_x_3_cuda(mem_block2[j], wc[level][i][j],  cur_size,i,0);
		}
		add_bias_and_relu(mem_block1[i], bc[level][i], cur_size);
	}
	reset_mem_block(mem_block2);
	printf("Convolution Layer 13 Done!\n");

	
	// Layer 18 (MaxPooling)
	for (i = 0; i < cshape[level][0]; i++) {
		maxpooling(mem_block1[i], cur_size);
	}
	cur_size /= 2;
	printf("Max Pooling Done!\n");

	
	// Layer 19 (Flatten)
	flatten(mem_block1, mem_block1_dense, cshape[level][0], cur_size, cur_size);
	
	printf("Flattening Done!\n");
	// Layer 20 (Dense 1)
	level = 0;
	dense(mem_block1_dense, wd[level], mem_block2_dense, dshape[level][0], dshape[level][1]);
	add_bias_and_relu_flatten(mem_block2_dense, bd[level], dshape[level][1], 1);
	reset_mem_block_dense(mem_block1_dense);
	printf("Dense layer 1 Done!\n");

	// Layer 21 (Dense 2)
	level = 1;
	dense(mem_block2_dense, wd[level], mem_block1_dense, dshape[level][0], dshape[level][1]);
	add_bias_and_relu_flatten(mem_block1_dense, bd[level], dshape[level][1], 1);
	reset_mem_block_dense(mem_block2_dense);
	printf("Dense layer 2 Done!\n");
	
	// Layer 22 (Dense 3)
	level = 2;
	dense(mem_block1_dense, wd[level], mem_block2_dense, dshape[level][0], dshape[level][1]);
	add_bias_and_relu_flatten(mem_block2_dense, bd[level], dshape[level][1], 1);
	softmax(mem_block2_dense, dshape[level][1]);
	mem_block2_dense[norm]=bias;
	printf("Dense layer 3 Done!\n");

	return;
}


void output_predictions(FILE *out) {
    int i;
    int predicted_class = -1;
    double max_prob = -1;
    for (i = 0; i < dshape[2][1]; i++) {
		fprintf(out, "%g ", mem_block2_dense[i]);
		if (mem_block2_dense[i] > max_prob) {
			max_prob = mem_block2_dense[i];
			predicted_class = i;
		}
	}
    fprintf(out, "\n");


    if (predicted_class != -1) {
        printf("Predicted Class: %d\n", predicted_class);
    } else {
        printf("Prediction failed.\n");
    }
}



char *trimwhitespace(char *str){
	char *end;

	while (isspace((unsigned char)*str)) str++;

	if (*str == 0) 
		return str;

	end = str + strlen(str) - 1;
	while (end > str && isspace((unsigned char)*end)) end--;

	*(end + 1) = 0;

	return str;
}


int main(int argc, char *argv[]) {

	FILE *file_list, *results;
	char buf[1024];
	time_t timeStart, timeEnd;
	double deltaTime;
	char *weights_file;
	char *image_list_file;
	char *output_file;
	int lvls = -1;


	if (argc != 4 && argc != 5) {
		printf("Usage: <program.exe> <weights file> <images list file> <output file> \n");
		return 0;
	}

	weights_file = argv[1];
	image_list_file = argv[2];
	output_file = argv[3];

	if (argc == 5) {
		lvls = 13;
	}
	init_memory();

	file_list = fopen(image_list_file, "r");
	if (file_list == NULL) {
		printf("Check file list location: %s", image_list_file);
		return 1;
	}

	results = fopen(output_file, "w");
	if (results == NULL) {
		printf("Couldn't open file for writing: %s", output_file);
		return 1;
	}

	read_weights(weights_file, lvls);

	while (!feof(file_list)) {

		printf("Proccesing the image...\n");
		gettimeofday(&timeStart, NULL);

		fgets(buf, 1024, file_list);
		if (strlen(buf) == 0) {
			break;
		}
		
		char *trimmed_buf = trimwhitespace(buf);
		image = read_image(trimmed_buf);
		image = normalize_image(image);

		predict_image_VGG(image,wc,bc);
		output_predictions(results);
		gettimeofday(&timeEnd, NULL);
		deltaTime = get_seconds(timeStart, timeEnd);
		printf("Parallel time using CUDA C to classify the image %s: %.3lf sec\n", buf, deltaTime);
	}

	free_memory();

	fclose(file_list);

	return 0;
}
