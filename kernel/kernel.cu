#include <stdio.h>

extern "C" __global__ void convolution_relu_layer(double input[100][100],
                                            double conv_filters[10][5][5],
                                            double outputs[10][20][20]) {
    // Convolution

    int neuron = blockIdx.x;
    int x = threadIdx.x;
    int y = threadIdx.y;
    int input_x_idx = x*5;
    int input_y_idx = y*5;

    double sum = 0;

    for(int i = 0; i < 5; i++) {
        for(int j = 0; j < 5; j++) {
            sum = sum + input[input_x_idx+i][input_y_idx+j] * conv_filters[neuron][i][j];
        }
    }

    outputs[neuron][x][y] = sum;

    // ReLU

    if (outputs[neuron][x][y] < 0.0) {
        outputs[neuron][x][y] = 0.0;
    }
}