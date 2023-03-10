# [PR] Using CUDA GPU to parallelize execution of Convolution and ReLU layers

# Summary
To improve the performance of the original Convolutional Neural Network (CNN) which was running on the 
CPU, we utilized CUDA to invoke a kernel on the GPU. This allows us to parallelize the convolution, ReLU, and output 
operations. The kernel was written is CUDA C++ in `kernel.cu` and the host code is written in `cuda.rs`.

# Tech details
We use the [Rustacuda library](https://github.com/bheisler/RustaCUDA) that allows us to write our host in 
Rust, and interface with the GPU. We first implement a `CudaContext` based on the struct defined in 
`cuda.rs`. This implementation initializes the fields in the `CudaContext` struct, and contains a 
`compute` function for computation of all the layers in the CNN.

In the `compute` function, we first create buffers for the input and the output of the convolution using
DeviceBox. We name this `input_box` and `conv_output_box` respectively. Thereafter, we launch the kernel 
to the GPU with 10 blocks of (20x20) threads, no dynamic shared memory on `stream`. This corresponds to the 
10 neurons giving (20x20) output after convolution is performed. The launch has to be done in an unsafe 
block because the launch macro is unsafe. Each buffer is converted using the as_device_ptr() 
so that the contents of the device buffer are provided. This kernel only performs the convolution and 
ReLU operations in parallel. 

For the GPU kernel, `kernel.cu`, we let the block index (`blockIdx.x`) define the neuron number, and the
thread indices (`threadIdx.x`, `threadIdx.y`) define the (x, y) coordinate of the output convolution 
matrix. To operate on the (100x100) input matrix, we use (x * 5, y * 5) to define the first pixel of the 
section to start computing the dot product between the input and the filter. We then loop through each
pixel in each (5x5) section of input and multiply it with the corresponding pixel in the filter. The sum of 
all these multiplications form the dot product. This will result in a (20x20) output
matrix of products. Thereafter, the ReLU of each value in output of the convolution layer is performed, 
to set any negative values to 0. 

Following the kernel execution, we copy the `conv_output_box` results back to host memory (`conv_output`). 
Lastly, for the Output layer, we now launch the kernel to the GPU with 10 blocks, no dynamic shared memory on `stream`,
in a similar fashion as above. Each block index (`blockIdx.x`) corresponds to the 10 sets of weights we use 
to perform the dot product with the `conv_output`. The sum of the 4000 multiplications of each set of weights
and the `conv_output` form the dot product for one of the ten elements in the `output` vector.
Since kernel launches are asynchronous, we wait for the kernels to finish executing using `self.stream.synchronize()?;`.

Note: the build.rs file will be run automatically at build time to compile the kernel.cu file into a 
kernel.ptx file, which gets downloaded to the GPU.

# Testing for correctness
We test our outputs from the CUDA optimization using the `compare.py` script provided. We first execute
the original CPU-run CNN and save the output as `out.csv`, then execute the new CUDA-optimized CNN and
save the output as `out_cuda.csv`. Thereafter, we execute `python3 compare.py` to check if the 
CUDA-optimized CNN provides correct outputs. We observe a `Comparison finished` printout, indicating
that the outputs from both CPU and GPU versions of the CNN are identical.

# Testing for performance 
We are unable to report consistently faster performance of the CUDA-optimized CNN over the CPU-run CNN
largely due to the extremely busy `ecetesla` servers. As such, we evaluate performance based on the 
bandwidth, i.e. the amount of work that can be done simultaneously. Unlike the CPU-run CNN, which performs
convolution for each element in the input and filter, and ReLU for each element in `conv_output` 
sequentially, the CUDA-optimized CNN allocates each filter / neuron to a block in the GPU, and assigns
(20x20) threads to each block. This allows for one element in each (5x5) section of the input to be 
computed every cycle for the convolution layer. As for the ReLU layer, all the elements in the `conv_output`
can be operated on at once. Lastly, for the Output layer, we are able to perform 10 dot products of the 10 flattened
weight vectors with all the elements in the `conv_output` simultaneously using the 10 threads, 
instead of performing them sequentially. In essence, we use Single Instruction Multiple Thread to improve the bandwidth.