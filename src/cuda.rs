// This is the skeleton for the CUDA implementation

use crate::cnn::*;
// use rustacuda::function::BlockSize;
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
        let ptx = CString::new(include_str!("../kernel/kernel.ptx"))?;
        let device = Device::get_device(0)?;
        let _ctx = Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;
        let cuda_ctx = CudaContext {
            conv_layer: DeviceBox::new(&cnn.conv_layer)?,
            output_layer: DeviceBox::new(&cnn.output_layer)?,
            module: Module::load_from_string(&ptx)?,
            stream: Stream::new(StreamFlags::NON_BLOCKING, None)?,
            _context: _ctx,
        };
        Ok(cuda_ctx)
    }

    pub fn compute(&mut self, input: &InputMatrix) -> Result<OutputVec, Box<dyn Error>> {
        let conv_output = ConvOutput([[[0.0; CONV_OUT_DIM]; CONV_OUT_DIM]; CONV_LAYER_SIZE]);
        let mut output = OutputVec([0.0; OUT_LAYER_SIZE]);

        // Create buffers for data
        let mut input_box = DeviceBox::new(input)?;
        let mut conv_output_box = DeviceBox::new(&conv_output)?;
        let mut output_box = DeviceBox::new(&output)?;

        unsafe {
            // Launch the kernel with 10 blocks of 20*20 threads, no dynamic shared memory on `stream`.
            let module = &self.module;
            let stream = &self.stream;
            let result = launch!(module.convolution_relu_layer<<<10, (20, 20), 0, stream>>>(
                input_box.as_device_ptr(),
                self.conv_layer.as_device_ptr(),
                conv_output_box.as_device_ptr()
            ));
            result?;
        }

        unsafe {
            // Launch the kernel with 10 blocks of 1 threads, no dynamic shared memory on `stream`.
            let module = &self.module;
            let stream = &self.stream;
            let result = launch!(module.output_layer<<<10, 1, 0, stream>>>(
                conv_output_box.as_device_ptr(),
                self.output_layer.as_device_ptr(),
                output_box.as_device_ptr()
            ));
            result?;
        }

        // Kernel launches are asynchronous, so we wait for the kernels to finish executing.
        self.stream.synchronize()?;

        // Copy the results back to host memory
        output_box.copy_to(&mut output)?;

        Ok(output)
    }
}