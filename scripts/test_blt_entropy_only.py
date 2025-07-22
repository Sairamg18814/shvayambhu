"""Test BLT entropy calculation directly."""

import sys
sys.path.append('.')

import numpy as np
from collections import Counter
import math

def calculate_byte_entropy(byte_sequence):
    """Calculate Shannon entropy of byte sequence."""
    if len(byte_sequence) == 0:
        return 0.0
    
    # Count byte frequencies
    byte_counts = Counter(byte_sequence)
    total_bytes = len(byte_sequence)
    
    # Calculate entropy
    entropy = 0.0
    for count in byte_counts.values():
        if count > 0:
            probability = count / total_bytes
            entropy -= probability * math.log2(probability)
    
    return entropy

def test_multilingual_entropy():
    """Test entropy calculation for multiple languages."""
    print("=== BLT Multilingual Entropy Analysis ===")
    
    # Test languages with sample texts
    language_samples = {
        "English": "The quick brown fox jumps over the lazy dog.",
        "Chinese": "快速的棕色狐狸跳过了懒狗。",
        "Arabic": "الثعلب البني السريع يقفز فوق الكلب الكسول.",
        "Russian": "Быстрая коричневая лиса прыгает через ленивую собаку.",
        "Japanese": "速い茶色のキツネが怠け者の犬を飛び越える。",
        "Korean": "빠른 갈색 여우가 게으른 개를 뛰어넘는다.",
        "Hindi": "तेज़ भूरी लोमड़ी आलसी कुत्ते के ऊपर कूदती है।",
        "Hebrew": "השועל החום המהיר קופץ מעל הכלב העצלן.",
        "Thai": "สุนัขจิ้งจอกสีน้ำตาลที่รวดเร็วกระโดดข้ามสุนัขขี้เกียจ",
        "Greek": "Η γρήγορη καφέ αλεπού πηδά πάνω από το τεμπέλικο σκυλί.",
        "Emoji": "🦊🏃‍♂️ ➡️ 🦮😴",
        "Mixed": "Hello 世界! مرحبا мир! 🌍"
    }
    
    print("\nLanguage entropy analysis:")
    print("-" * 90)
    print(f"{'Language':15} | {'Chars':>6} | {'Bytes':>6} | {'B/C':>4} | {'Entropy':>7} | {'Max Ent':>7} | {'Efficiency':>10}")
    print("-" * 90)
    
    results = []
    
    for language, text in language_samples.items():
        # Get byte representation
        byte_seq = text.encode('utf-8')
        byte_count = len(byte_seq)
        char_count = len(text)
        
        # Calculate entropy
        entropy = calculate_byte_entropy(byte_seq)
        
        # Count unique bytes
        unique_bytes = len(set(byte_seq))
        
        # Maximum possible entropy for this many unique bytes
        max_entropy = math.log2(unique_bytes) if unique_bytes > 0 else 0
        
        # Efficiency (how close to maximum entropy)
        efficiency = entropy / max_entropy if max_entropy > 0 else 0
        
        # Bytes per character
        bytes_per_char = byte_count / char_count
        
        results.append({
            'language': language,
            'chars': char_count,
            'bytes': byte_count,
            'bytes_per_char': bytes_per_char,
            'entropy': entropy,
            'max_entropy': max_entropy,
            'efficiency': efficiency,
            'unique_bytes': unique_bytes
        })
        
        print(f"{language:15} | {char_count:6} | {byte_count:6} | {bytes_per_char:4.1f} | {entropy:7.3f} | {max_entropy:7.3f} | {efficiency:10.2%}")
    
    # Test patch size recommendations based on entropy
    print("\n\nRecommended patch sizes based on entropy:")
    print("-" * 60)
    
    for r in results:
        # Simple heuristic: higher entropy = smaller patches
        if r['entropy'] > 5.5:
            patch_size = 12
        elif r['entropy'] > 5.0:
            patch_size = 16
        elif r['entropy'] > 4.5:
            patch_size = 20
        else:
            patch_size = 24
        
        print(f"{r['language']:15} | Entropy: {r['entropy']:5.3f} | Recommended patch: {patch_size} bytes")
    
    # Test special Unicode cases
    print("\n\nSpecial Unicode cases:")
    print("-" * 80)
    
    special_cases = {
        "Zero-width space": "Hello\u200bWorld",
        "Combining chars": "é = e\u0301",
        "RTL-LTR mix": "Hello שלום World",
        "Surrogate pairs": "𝐇𝐞𝐥𝐥𝐨 𝕎𝕠𝕣𝕝𝕕",
        "Emoji ZWJ": "👨‍👩‍👧‍👦",
        "Mixed scripts": "English中文عربيрусский",
        "Control chars": "Line1\nLine2\tTab\rReturn",
        "Math symbols": "∀x∈ℝ: x² ≥ 0",
    }
    
    for case_name, text in special_cases.items():
        byte_seq = text.encode('utf-8')
        entropy = calculate_byte_entropy(byte_seq)
        unique = len(set(byte_seq))
        print(f"{case_name:20} | Bytes: {len(byte_seq):3} | Entropy: {entropy:6.3f} | Unique: {unique:3}")
    
    # Summary analysis by script type
    print("\n\nSummary by script type:")
    print("-" * 70)
    
    script_groups = {
        "Latin": ["English"],
        "CJK": ["Chinese", "Japanese", "Korean"],
        "RTL": ["Arabic", "Hebrew"],
        "Cyrillic": ["Russian"],
        "Indic": ["Hindi", "Thai"],
        "Other": ["Greek", "Emoji", "Mixed"]
    }
    
    for script_type, languages in script_groups.items():
        script_results = [r for r in results if r['language'] in languages]
        if script_results:
            avg_entropy = np.mean([r['entropy'] for r in script_results])
            avg_bytes_per_char = np.mean([r['bytes_per_char'] for r in script_results])
            avg_efficiency = np.mean([r['efficiency'] for r in script_results])
            
            print(f"{script_type:12} | Avg Entropy: {avg_entropy:5.3f} | Avg B/C: {avg_bytes_per_char:4.2f} | Avg Eff: {avg_efficiency:6.2%}")
    
    # Test entropy-based patch boundaries
    print("\n\nEntropy-based patch boundary detection:")
    print("-" * 60)
    
    test_text = "Hello world! 你好世界！ مرحبا بالعالم!"
    byte_seq = test_text.encode('utf-8')
    
    # Calculate sliding window entropy
    window_size = 8
    entropies = []
    
    for i in range(len(byte_seq) - window_size + 1):
        window = byte_seq[i:i+window_size]
        entropies.append(calculate_byte_entropy(window))
    
    # Find entropy changes (potential boundaries)
    entropy_changes = [abs(entropies[i] - entropies[i-1]) for i in range(1, len(entropies))]
    
    # Find peaks in entropy change
    threshold = np.mean(entropy_changes) + np.std(entropy_changes)
    boundaries = [i for i, change in enumerate(entropy_changes) if change > threshold]
    
    print(f"Test text: {test_text}")
    print(f"Byte length: {len(byte_seq)}")
    print(f"Average window entropy: {np.mean(entropies):.3f}")
    print(f"Entropy change threshold: {threshold:.3f}")
    print(f"Detected boundaries at positions: {boundaries[:10]}...")  # Show first 10
    
    print("\n✅ BLT entropy analysis complete!")
    print("   Entropy-based patching adapts to different languages and scripts.")
    print("   No tokenization needed - works directly with UTF-8 bytes.")

if __name__ == "__main__":
    test_multilingual_entropy()
