"""
Text Post-processing Utilities
==============================

Cleans up generated text from the model.
"""

import re
from typing import List, Optional


def remove_repetitive_chars(text: str, max_repeat: int = 3) -> str:
    """Remove excessive repetitive characters."""
    # Replace any character repeated more than max_repeat times
    pattern = r'(.)\1{' + str(max_repeat) + ',}'
    cleaned = re.sub(pattern, lambda m: m.group(1) * max_repeat, text)
    return cleaned


def clean_whitespace(text: str) -> str:
    """Clean up excessive whitespace."""
    # Replace multiple spaces with single space
    text = re.sub(r' +', ' ', text)
    # Replace multiple newlines with double newline
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Strip leading/trailing whitespace
    return text.strip()


def fix_common_patterns(text: str) -> str:
    """Fix common generation issues."""
    # Fix repeated "As" pattern
    text = re.sub(r'(As\s*){3,}', 'As ', text)
    
    # Fix repeated punctuation
    text = re.sub(r'([.!?])\1+', r'\1', text)
    
    # Fix repeated commas and semicolons
    text = re.sub(r'([,;])\1+', r'\1', text)
    
    # Fix repeated quotes
    text = re.sub(r'(["\'])\1+', r'\1', text)
    
    return text


def extract_coherent_segments(text: str, min_word_length: int = 3) -> List[str]:
    """Extract coherent word segments from text."""
    # Split by various delimiters
    words = re.split(r'[\s\n.,;:!?]+', text)
    
    # Filter for coherent words
    coherent_words = []
    for word in words:
        # Remove non-alphabetic characters from edges
        cleaned = re.sub(r'^[^a-zA-Z]+|[^a-zA-Z]+$', '', word)
        # Keep if it's a reasonable word
        if len(cleaned) >= min_word_length and cleaned.isalpha():
            coherent_words.append(cleaned)
    
    return coherent_words


def reconstruct_sentences(words: List[str]) -> str:
    """Reconstruct sentences from word list."""
    if not words:
        return ""
    
    # Simple reconstruction with capitalization
    sentences = []
    current_sentence = []
    
    for i, word in enumerate(words):
        if i == 0 or (i > 0 and words[i-1].lower() in ['i', 'a']):
            current_sentence.append(word.capitalize())
        else:
            current_sentence.append(word.lower())
        
        # End sentence at certain words or every 10-15 words
        if (word.lower() in ['is', 'are', 'was', 'were', 'can', 'will', 'would'] and 
            len(current_sentence) > 5) or len(current_sentence) > 12:
            sentences.append(' '.join(current_sentence) + '.')
            current_sentence = []
    
    # Add remaining words
    if current_sentence:
        sentences.append(' '.join(current_sentence) + '.')
    
    return ' '.join(sentences)


def post_process_generated_text(text: str, aggressive: bool = False) -> str:
    """
    Post-process generated text to make it more coherent.
    
    Args:
        text: Raw generated text
        aggressive: If True, applies more aggressive cleaning
        
    Returns:
        Cleaned text
    """
    # Basic cleaning
    text = remove_repetitive_chars(text, max_repeat=2 if aggressive else 3)
    text = clean_whitespace(text)
    text = fix_common_patterns(text)
    
    if aggressive:
        # Extract and reconstruct
        words = extract_coherent_segments(text)
        if words:
            text = reconstruct_sentences(words)
    
    # Final cleanup
    text = clean_whitespace(text)
    
    # If text is too short or still problematic, return a default
    if len(text.split()) < 3:
        return "I am still learning to express myself clearly."
    
    return text


def clean_consciousness_response(text: str) -> str:
    """Special cleaning for consciousness-related responses."""
    # Remove excessive philosophical terms
    text = re.sub(r'(conscious|awareness|philosophical?|essence|being)\s*', r'\1 ', text, flags=re.IGNORECASE)
    
    # Apply standard cleaning
    text = post_process_generated_text(text, aggressive=True)
    
    # Add consciousness-specific formatting
    if text and not text[0].isupper():
        text = text[0].upper() + text[1:]
    
    return text


# Test the functions
if __name__ == "__main__":
    # Test cases
    test_texts = [
        "Ascicusnscposnessesponsesescoscessscccccccccc",
        "AsssssmasssssccAsssscousssssscccccouscone icouscosss",
        "hicespousccccciciosn\nAscascccccchiousscouscessscasphicesscosonesscascescouscespouscasscicancanssssce",
        "I am am am am doing well well well thank you you you",
        "Hello!!! How are you????....."
    ]
    
    print("Text Post-processing Tests")
    print("=" * 60)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\nTest {i}:")
        print(f"Original: {repr(text)}")
        print(f"Basic:    {repr(post_process_generated_text(text))}")
        print(f"Aggressive: {repr(post_process_generated_text(text, aggressive=True))}")
        if 'conscious' in text.lower():
            print(f"Consciousness: {repr(clean_consciousness_response(text))}")