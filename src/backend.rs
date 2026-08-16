// Rustrainer-OCR A GUI Utility to train/fine tune OCR Models written in Rust.
// Copyright (C) 2026 Mohammad Najm
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.
//
// Contact: Mohammad Najm <najm.devops@gmail.com>
// https://github.com/najmdevstudio/Rustrainer_OCR

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
