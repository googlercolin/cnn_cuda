// Very minimal skeleton for the kernel

#include <stdio.h>

//int **zip(int *arr1, int *arr2, int length)
//{
//    int **ret = new int*[length];
//    for(int i = 0; i<length; i++)
//    {
//        ret[i] = new int[2];
//        ret[i][0] = arr1[i];
//        ret[i][1] = arr2[i];
//    }
//    return ret;
//}

extern "C" __global__ void convolution_layer(double input[100][100],
                                            double conv_filters[10][5][5],
                                            double outputs[10][20][20]) {
    int neuron = blockIdx.x;
    int x = threadIdx.x;
    int y = threadIdx.y;
    int input_x_idx = x*5;
    int input_y_idx = y*5;

    int sum = 0;

    for(int i = 0; i < 5; i++) {
        for(int j = 0; j < 5; j++) {
            sum = sum + input[input_x_idx+i][input_y_idx+j] * conv_filters[neuron][i][j];
        }
    }

    outputs[neuron][x][y] = sum;

    if (outputs[neuron][x][y] < 0.0) {
        outputs[neuron][x][y] = 0.0;
    }
}

//extern "C" __global__ void relu_layer(double conv_out[10][20][20]) {
//    int neuron = blockIdx.x;
//    int x = threadIdx.x;
//    int y = threadIdx.y;
//
//    if (conv_out[neuron][x][y] < 0.0) {
//        conv_out[neuron][x][y] = 0.0;
//    }
//
//}