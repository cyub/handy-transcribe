//! Post-processing: the non-filler cleanup and filler-word removal Handy
//! applies after transcription. Ported from Handy's `audio_toolkit/text.rs`.
//! Custom-word fuzzy correction (which needs `natural`/`strsim`) is omitted;
//! custom words are instead injected as model vocabulary via RunOptions.

use once_cell::sync::Lazy;
use regex::Regex;

/// Filler tokens that are not lexical words in any language.
const UNIVERSAL_FILLER_WORDS: &[&str] = &[
    "uh", "uhm", "umm", "uhh", "uhhh", "ehh", "ehm", "ahm", "hmm", "hm", "mmm", "хм", "ммм",
];

/// Filler words only safe to remove with evidence for the output language.
fn gated_filler_words_for_language(lang: &str) -> &'static [&'static str] {
    let base_lang = lang.split(&['-', '_'][..]).next().unwrap_or(lang);
    match base_lang {
        "en" => &["um", "ah", "eh"],
        "de" => &["äh", "ähm"],
        "fr" => &["euh"],
        _ => &[],
    }
}

static MULTI_SPACE_PATTERN: Lazy<Regex> = Lazy::new(|| Regex::new(r"\s{2,}").unwrap());

/// Collapses repeated words (3+ repetitions) to a single instance.
fn collapse_stutters(text: &str) -> String {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return text.to_string();
    }

    let mut result: Vec<&str> = Vec::new();
    let mut i = 0;

    while i < words.len() {
        let word = words[i];
        let word_lower = word.to_lowercase();

        if word_lower.chars().all(|c| c.is_alphabetic()) {
            let mut count = 1;
            while i + count < words.len() && words[i + count].to_lowercase() == word_lower {
                count += 1;
            }
            if count >= 3 {
                result.push(word);
                i += count;
            } else {
                result.push(word);
                i += 1;
            }
        } else {
            result.push(word);
            i += 1;
        }
    }

    result.join(" ")
}

/// Whether a word appended to `kept` would open a sentence.
fn opens_sentence(kept: &str) -> bool {
    kept.trim_end()
        .chars()
        .next_back()
        .is_none_or(|c| matches!(c, '.' | '!' | '?' | '…'))
}

fn push_restoring_capital(kept: &mut String, segment: &str, capital_owed: &mut bool) {
    if *capital_owed {
        if let Some((index, first)) = segment.char_indices().find(|(_, c)| c.is_alphanumeric()) {
            *capital_owed = false;
            kept.push_str(&segment[..index]);
            kept.extend(first.to_uppercase());
            kept.push_str(&segment[index + first.len_utf8()..]);
            return;
        }
    }
    kept.push_str(segment);
}

/// Deletes every match of one filler pattern, handing a capitalized filler's
/// capital to the word that takes its place.
fn remove_filler_matches(text: &str, pattern: &Regex) -> String {
    let mut kept = String::with_capacity(text.len());
    let mut resume = 0;
    let mut capital_owed = false;

    for filler in pattern.find_iter(text) {
        push_restoring_capital(&mut kept, &text[resume..filler.start()], &mut capital_owed);
        let capitalized = filler.as_str().starts_with(char::is_uppercase);
        capital_owed |= capitalized && opens_sentence(&kept);
        resume = filler.end();
    }
    push_restoring_capital(&mut kept, &text[resume..], &mut capital_owed);

    kept
}

/// Removes filler words when enabled. `language` gates language-specific
/// filler tokens (`um`/`ah`/`eh` for English, etc.).
pub fn remove_filler_words(text: &str, language: Option<&str>, enabled: bool) -> String {
    if !enabled {
        return text.to_string();
    }

    let patterns: Vec<Regex> = UNIVERSAL_FILLER_WORDS
        .iter()
        .chain(
            language
                .map(gated_filler_words_for_language)
                .unwrap_or_default(),
        )
        .map(|word| Regex::new(&format!(r"(?i)\b{}\b[,.]?", regex::escape(word))).unwrap())
        .collect();

    let mut filtered = text.to_string();
    for pattern in &patterns {
        filtered = remove_filler_matches(&filtered, pattern);
    }
    filtered
}

/// Applies non-filler transcription cleanup (repeated words + whitespace).
pub fn normalize_transcription_output(text: &str) -> String {
    let mut normalized = collapse_stutters(text);
    normalized = MULTI_SPACE_PATTERN
        .replace_all(&normalized, " ")
        .to_string();
    normalized.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_collapses_multiple_spaces_and_trims() {
        assert_eq!(
            normalize_transcription_output("  hello   world  "),
            "hello world"
        );
    }

    #[test]
    fn normalize_collapses_stutters() {
        assert_eq!(normalize_transcription_output("I I I I think"), "I think");
    }

    #[test]
    fn filler_words_are_removed_by_default() {
        let out = remove_filler_words("Um, so I think", Some("en"), true);
        assert_eq!(normalize_transcription_output(&out), "So I think");
    }

    #[test]
    fn filler_words_kept_when_disabled() {
        let out = remove_filler_words("Um, so I think", Some("en"), false);
        assert_eq!(out, "Um, so I think");
    }
}
