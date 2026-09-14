#!/opt/miniconda3/bin/python3
import os
import re

# Load reference vocabulary from Tiny Shakespeare input corpus
corpus_file = os.path.join(os.path.dirname(__file__), 'input.txt')
with open(corpus_file, 'r', encoding='utf-8') as f:
    corpus_text = f.read()

# Build valid English word set (case-insensitive)
valid_words = set(re.findall(r"\b[a-zA-Z]{2,}\b", corpus_text.lower()))

def evaluate_gibberish_rate(sample_text):
    """
    Extracts words from sample_text and calculates the percentage of invalid/gibberish words.
    Returns (gibberish_rate_pct, total_words, invalid_words_list).
    """
    words = re.findall(r"\b[a-zA-Z]{2,}\b", sample_text.lower())
    if not words:
        return 0.0, 0, []
    
    invalid_words = [w for w in words if w not in valid_words]
    rate = (len(invalid_words) / len(words)) * 100.0
    return rate, len(words), invalid_words

if __name__ == "__main__":
    test_sample = """
    GLOUCESTER:
    What never done?
    MENENIUS:
    Sopp'd visition, to prison!
    Nave the freiss, whome of it but, the dett shat cove hens
    """
    rate, total, invalids = evaluate_gibberish_rate(test_sample)
    print(f"🧪 Test Gibberish Audit: {rate:.2f}% non-word rate ({len(invalids)} invalid out of {total} words)")
    print(f"Invalid words found: {invalids}")
