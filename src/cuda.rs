// This is the skeleton for the CUDA implementation

use crate::cnn::*;
use rustacuda::function::BlockSize;
use rustacuda::launch;
use rustacuda::memory::DeviceBox;
use rustacuda::prelude::*;
use std::error::Error;
use std::ffi::CString;

// Fields need to be ordered this way so the DeviceBoxes are
// dropped before the Context. Otherwise the drop will panic.

pub struct CudaContext {
    conv_layer: DeviceBox<ConvLayer>,
    output_layer: DeviceBox<OutputLayer>,
    module: Module,
    stream: Stream,
    _context: Context,
}

impl CudaContext {
    pub fn init(cnn: &Cnn) -> Result<Self, Box<dyn Error>> {
        rustacuda::init(CudaFlags::empty())?;
        self::conv_layer = DeviceBox::new(&cnn.conv_layer)?;
        self::output_layer = DeviceBox::new(&cnn.output_layer);
        let device = Device::get_device(0)?;
        self::_context = Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;
        let ptx = CString::new(include_str!("../kernel/kernel.ptx"))?;
        self::module = Module::load_from_string(&ptx)?;
        self::stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        Ok(Self)
    }

    pub fn compute(&mut self, input: &InputMatrix) -> Result<OutputVec, Box<dyn Error>> {
        let mut conv_output = ConvOutput([[[0.0; CONV_OUT_DIM]; CONV_OUT_DIM]; CONV_LAYER_SIZE]);
        let mut output = OutputVec([0.0; OUT_LAYER_SIZE]);

        // Create buffers for data
        let mut input_buf = DeviceBuffer::from_slice(input.as_slice())?;
        let mut conv_layer_buf = DeviceBuffer::from_slice(conv_output.as_slice())?;
        let mut conv_output_buf = DeviceBuffer::from_slice(&cnn.conv_layer.as_slice())?;
        let mut output_buf = DeviceBuffer::from_slice(output.as_slice())?;

        unsafe {
            // Launch the kernel with one block of one thread, no dynamic shared memory on `stream`.
            let result = launch!(self.module.convolution_layer<<<10, (20, 20), 0, self.stream>>>(
                input_buf.as_device_ptr(),
                conv_layer_buf.as_device_ptr(),
                conv_output_buf.as_device_ptr()
            ));
            result?;
        }

        // Kernel launches are asynchronous, so we wait for the kernels to finish executing.
        self.stream.synchronize()?;

        // Copy the results back to host memory
        conv_output_buf.copy_to(&mut conv_output)?;

        unsafe {
            // Launch the kernel with one block of one thread, no dynamic shared memory on `stream`.
            let result = launch!(self.module.relu_layer<<<10, (20, 20), 0, self.stream>>>(
                conv_output_buf.as_device_ptr()
            ));
            result?;
        }

        // Kernel launches are asynchronous, so we wait for the kernels to finish executing.
        self.stream.synchronize()?;

        // Copy the results back to host memory
        conv_output_buf.copy_to(&mut conv_output)?;

        output_layer(&conv_output, &cnn.output_layer, &mut output);

        Ok(output)
    }
}

fn output_layer(input: &ConvOutput, weights: &OutputLayer, output: &mut OutputVec) {
    // Go thru each output neuron
    for (weight, out) in weights.0.iter().zip(output.0.iter_mut()) {
        // Flatten the output of the previous layer into a 4000x1 vector, then dot product it with
        // the weight vector to produce a single value
        let flattened = input.0.iter().flat_map(|n| n.iter().flat_map(|r| r.iter()));
        let prod: f64 = flattened.zip(weight.iter()).map(|(a, b)| a * b).sum();
        *out = prod;
    }
}