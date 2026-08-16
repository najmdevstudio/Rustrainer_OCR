/// Character vocabulary for license plate OCR.
/// Index 0 is reserved for the CTC blank token.
pub const VOCAB: &str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";
pub const BLANK_IDX: usize = 0;
#[allow(dead_code)]
pub const NUM_CLASSES: usize = 37; // 36 alphanumeric + 1 blank

/// Encode a plate text string into a sequence of class indices (1-indexed, 0 = blank).
pub fn encode(plate_text: &str) -> Vec<usize> {
    plate_text
        .chars()
        .filter_map(|c| {
            let c = c.to_ascii_uppercase();
            VOCAB.find(c).map(|i| i + 1)
        })
        .collect()
}

/// CTC greedy decode: collapse repeated indices and remove blanks.
pub fn decode(indices: &[usize]) -> String {
    let mut result = String::new();
    let mut prev = BLANK_IDX;
    for &idx in indices {
        if idx != prev && idx != BLANK_IDX {
            if let Some(c) = VOCAB.chars().nth(idx - 1) {
                result.push(c);
            }
        }
        prev = idx;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode() {
        let text = "AB12CD";
        let encoded = encode(text);
        assert_eq!(encoded, vec![11, 12, 2, 3, 13, 14]);
        // Simulate CTC output with repeats and blanks
        let ctc_output = vec![11, 11, 0, 12, 0, 0, 2, 3, 3, 13, 14, 0];
        let decoded = decode(&ctc_output);
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_num_classes() {
        assert_eq!(VOCAB.len() + 1, NUM_CLASSES);
    }
}
