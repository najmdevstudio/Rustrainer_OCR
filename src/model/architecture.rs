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

//! Which of plate-ocr's supported OCR model architectures a checkpoint/pretrained file uses.
//!
//! Fine-tuning from an external file (`.pt`/`.pth`/`.onnx`) auto-detects the architecture by
//! inspecting the file itself (see `crate::interop`); this project's own Burn checkpoints carry
//! no such structure to inspect ahead of time, so a small sidecar file records it instead (see
//! [`Architecture::write_sidecar`]). Starting a brand-new training run (nothing to detect from)
//! lets the user pick one explicitly (CLI `--architecture`, GUI parameters screen).

use std::path::PathBuf;
use std::str::FromStr;

/// One of plate-ocr's supported OCR model architectures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Architecture {
    /// 4x (Conv2d + BatchNorm) -> bidirectional LSTM -> Linear -> CTC. The original, higher-
    /// capacity architecture (see [`crate::model::crnn::CrnnOcr`]).
    CrnnBiLstm,
    /// 4x (Conv2d + BatchNorm) -> 2-layer MLP -> CTC, no recurrent layer. A lighter/faster
    /// alternative sharing the same CNN backbone (see [`crate::model::conv_ctc::ConvCtcOcr`]).
    ConvCtc,
}

impl Architecture {
    /// All supported architectures, in a stable order — used to populate GUI dropdowns/CLI help.
    pub const ALL: [Architecture; 2] = [Architecture::CrnnBiLstm, Architecture::ConvCtc];

    /// Short, human-readable label shown in the GUI and log/CLI output.
    pub fn label(self) -> &'static str {
        match self {
            Architecture::CrnnBiLstm => "CRNN (Conv + BiLSTM + CTC)",
            Architecture::ConvCtc => "Conv-CTC (Conv-only, no recurrent layer)",
        }
    }

    /// Stable machine-readable identifier: used in the checkpoint sidecar file, the ONNX/PyTorch
    /// export manifest, and accepted by the CLI's `--architecture` flag.
    pub fn id(self) -> &'static str {
        match self {
            Architecture::CrnnBiLstm => "crnn_bilstm",
            Architecture::ConvCtc => "conv_ctc",
        }
    }

    /// Path of the small sidecar file this project writes next to every Burn checkpoint it
    /// saves, recording which architecture it is. Burn's own checkpoint format has no room for
    /// custom metadata, and unlike `.pt`/`.onnx` there's no tensor structure that can be
    /// inspected without already knowing which config to load it with.
    pub fn sidecar_path(model_path: &str) -> PathBuf {
        PathBuf::from(format!("{model_path}.architecture"))
    }

    /// Writes [`Self::sidecar_path`] for `model_path`.
    pub fn write_sidecar(self, model_path: &str) -> std::io::Result<()> {
        std::fs::write(Self::sidecar_path(model_path), self.id())
    }

    /// Reads back the architecture written by [`Self::write_sidecar`], defaulting to
    /// [`Architecture::CrnnBiLstm`] — this project's only architecture before this file existed —
    /// if the sidecar is missing/unreadable, so every checkpoint saved by earlier versions of
    /// this tool keeps loading exactly as before.
    pub fn read_sidecar(model_path: &str) -> Architecture {
        std::fs::read_to_string(Self::sidecar_path(model_path))
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(Architecture::CrnnBiLstm)
    }
}

impl std::fmt::Display for Architecture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

impl FromStr for Architecture {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim() {
            "crnn_bilstm" | "crnn" => Ok(Architecture::CrnnBiLstm),
            "conv_ctc" | "conv-ctc" => Ok(Architecture::ConvCtc),
            other => Err(format!("Unknown architecture '{other}' (expected 'crnn' or 'conv-ctc').")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sidecar_round_trips() {
        let path = std::env::temp_dir().join(format!("plate_ocr_arch_sidecar_test_{}", std::process::id()));
        let path_str = path.to_string_lossy().to_string();

        Architecture::ConvCtc.write_sidecar(&path_str).expect("failed to write sidecar");
        assert_eq!(Architecture::read_sidecar(&path_str), Architecture::ConvCtc);

        let _ = std::fs::remove_file(Architecture::sidecar_path(&path_str));
    }

    #[test]
    fn missing_sidecar_defaults_to_crnn_for_backward_compatibility() {
        let path = std::env::temp_dir().join("plate_ocr_definitely_missing_sidecar_xyz");
        let _ = std::fs::remove_file(Architecture::sidecar_path(&path.to_string_lossy()));
        assert_eq!(Architecture::read_sidecar(&path.to_string_lossy()), Architecture::CrnnBiLstm);
    }

    #[test]
    fn from_str_accepts_ids_and_aliases() {
        assert_eq!("crnn_bilstm".parse::<Architecture>().unwrap(), Architecture::CrnnBiLstm);
        assert_eq!("crnn".parse::<Architecture>().unwrap(), Architecture::CrnnBiLstm);
        assert_eq!("conv_ctc".parse::<Architecture>().unwrap(), Architecture::ConvCtc);
        assert_eq!("conv-ctc".parse::<Architecture>().unwrap(), Architecture::ConvCtc);
        assert!("nonsense".parse::<Architecture>().is_err());
    }
}
