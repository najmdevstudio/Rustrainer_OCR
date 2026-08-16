//! Backend/device selection, shared by the CLI commands and the GUI wizard.
//!
//! Exactly one of the `rocm`, `vulkan` or `cpu` features must be active (see `Cargo.toml`).

#[cfg(feature = "rocm")]
mod imp {
    pub type TrainBackend = burn::backend::Autodiff<burn::backend::Rocm>;
    pub type InferBackend = burn::backend::Rocm;

    pub fn device() -> burn::tensor::Device<InferBackend> {
        burn::backend::rocm::RocmDevice::default()
    }
}

#[cfg(all(feature = "vulkan", not(feature = "rocm")))]
mod imp {
    pub type TrainBackend = burn::backend::Autodiff<burn::backend::Vulkan>;
    pub type InferBackend = burn::backend::Vulkan;

    pub fn device() -> burn::tensor::Device<InferBackend> {
        Default::default()
    }
}

#[cfg(all(feature = "cpu", not(feature = "rocm"), not(feature = "vulkan")))]
mod imp {
    pub type TrainBackend = burn::backend::Autodiff<burn::backend::NdArray>;
    pub type InferBackend = burn::backend::NdArray;

    pub fn device() -> burn::tensor::Device<InferBackend> {
        Default::default()
    }
}

pub use imp::*;
