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

//! GPLv3 compliance text shared by the CLI (the startup notice printed in interactive
//! terminals, and the `show w` / `show c` commands) and the GUI's About box. See the "How to
//! Apply These Terms to Your New Programs" section at the end of [`LICENSE`](../LICENSE) for
//! the requirements this module satisfies.

/// Display name used in the copyright/license notices and the GUI's About box. The Cargo
/// package/binary itself keeps the `plate-ocr` name for backwards-compatible CLI invocation.
pub const PROGRAM_NAME: &str = "Rustrainer-OCR";

/// Copyright line shared by every notice.
pub const COPYRIGHT_LINE: &str = "Copyright (C) 2026 Mohammad Najm";

/// How to reach the author, as requested by the GPL's "How to Apply These Terms" section.
pub const CONTACT: &str =
    "Mohammad Najm <najm.devops@gmail.com> — https://github.com/najmdevstudio/Rustrainer_OCR";

/// Full text of the GNU GPLv3, embedded at compile time so `show w`/`show c` and the GUI's
/// About box can quote the exact sections they refer to without depending on a LICENSE file
/// being present next to the running binary.
const FULL_LICENSE_TEXT: &str = include_str!("../LICENSE");

/// The short notice the GPL asks an interactively-run program to print at startup; printed by
/// `main` whenever stdout is a terminal.
pub fn short_notice() -> String {
    format!(
        "{PROGRAM_NAME} {COPYRIGHT_LINE}\n\
This program comes with ABSOLUTELY NO WARRANTY; for details run `plate-ocr show w`.\n\
This is free software, and you are welcome to redistribute it\n\
under certain conditions; run `plate-ocr show c` for details."
    )
}

/// GPLv3 sections 15-17 (Disclaimer of Warranty / Limitation of Liability / Interpretation of
/// Sections 15 and 16), shown by `plate-ocr show w` and the GUI's About box.
pub fn warranty_section() -> &'static str {
    extract_section("15. Disclaimer of Warranty.", "END OF TERMS AND CONDITIONS")
}

/// GPLv3 sections 4-6 (Conveying Verbatim Copies / Conveying Modified Source Versions /
/// Conveying Non-Source Forms), shown by `plate-ocr show c` and the GUI's About box.
pub fn conditions_section() -> &'static str {
    extract_section("4. Conveying Verbatim Copies.", "7. Additional Terms.")
}

/// Slices [`FULL_LICENSE_TEXT`] between (and including) `start_marker` and just before
/// `end_marker`, trimming trailing whitespace.
fn extract_section(start_marker: &str, end_marker: &str) -> &'static str {
    let start = FULL_LICENSE_TEXT
        .find(start_marker)
        .expect("LICENSE is missing an expected GPLv3 section marker");
    let end = FULL_LICENSE_TEXT[start..]
        .find(end_marker)
        .map(|offset| start + offset)
        .expect("LICENSE is missing an expected GPLv3 section marker");
    FULL_LICENSE_TEXT[start..end].trim_end()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn warranty_section_contains_disclaimer_and_liability() {
        let text = warranty_section();
        assert!(text.contains("THERE IS NO WARRANTY"));
        assert!(text.contains("Limitation of Liability"));
    }

    #[test]
    fn conditions_section_contains_conveying_rules() {
        let text = conditions_section();
        assert!(text.contains("Conveying Verbatim Copies"));
        assert!(text.contains("Conveying Modified Source Versions"));
        assert!(text.contains("Conveying Non-Source Forms"));
        assert!(!text.contains("Additional Terms"));
    }

    #[test]
    fn short_notice_mentions_warranty_and_conditions_commands() {
        let notice = short_notice();
        assert!(notice.contains(PROGRAM_NAME));
        assert!(notice.contains("NO WARRANTY"));
        assert!(notice.contains("show w"));
        assert!(notice.contains("show c"));
    }
}
