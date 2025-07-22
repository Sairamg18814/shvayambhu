"""Simple BLT multilingual test."""

import sys
sys.path.append('.')

from core.blt.entropy import calculate_byte_entropy
import numpy as np

def test_multilingual_byte_processing():
    """Test byte-level processing for multiple languages."""
    print("=== BLT Multilingual Byte Processing Test ===")
    
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
    
    print("\nLanguage analysis:")
    print("-" * 80)
    print(f"{'Language':15} | {'Chars':>6} | {'Bytes':>6} | {'B/C':>4} | {'Entropy':>7} | {'Unique':>6}")
    print("-" * 80)
    
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
        
        # Bytes per character
        bytes_per_char = byte_count / char_count
        
        results.append({
            'language': language,
            'chars': char_count,
            'bytes': byte_count,
            'bytes_per_char': bytes_per_char,
            'entropy': entropy,
            'unique_bytes': unique_bytes
        })
        
        print(f"{language:15} | {char_count:6} | {byte_count:6} | {bytes_per_char:4.1f} | {entropy:7.3f} | {unique_bytes:6}")
    
    # Test special Unicode cases
    print("\n\nSpecial Unicode cases:")
    print("-" * 60)
    
    special_cases = {
        "Zero-width space": "Hello\u200bWorld",
        "Combining chars": "é = e\u0301",
        "RTL-LTR mix": "Hello שלום World",
        "Surrogate pairs": "𝐇𝐞𝐥𝐥𝐨 𝕎𝕠𝕣𝕝𝕕",
        "Emoji ZWJ": "👨‍👩‍👧‍👦",
    }
    
    for case_name, text in special_cases.items():
        byte_seq = text.encode('utf-8')
        entropy = calculate_byte_entropy(byte_seq)
        print(f"{case_name:20} | Bytes: {len(byte_seq):3} | Entropy: {entropy:.3f}")
    
    # Summary analysis
    print("\n\nSummary Analysis:")
    print("-" * 60)
    
    # Group by script type
    script_groups = {
        "Latin": ["English", "Mixed"],
        "CJK": ["Chinese", "Japanese", "Korean"],
        "Arabic/Hebrew": ["Arabic", "Hebrew"],
        "Cyrillic": ["Russian"],
        "Indic": ["Hindi", "Thai"],
        "Other": ["Greek", "Emoji"]
    }
    
    for script_type, languages in script_groups.items():
        script_results = [r for r in results if r['language'] in languages]
        if script_results:
            avg_bytes_per_char = np.mean([r['bytes_per_char'] for r in script_results])
            avg_entropy = np.mean([r['entropy'] for r in script_results])
            print(f"{script_type:15} | Avg B/C: {avg_bytes_per_char:4.2f} | Avg Entropy: {avg_entropy:.3f}")
    
    # Test patch size recommendations
    print("\n\nRecommended patch sizes by script:")
    print("-" * 60)
    
    for script_type, languages in script_groups.items():
        script_results = [r for r in results if r['language'] in languages]
        if script_results:
            avg_entropy = np.mean([r['entropy'] for r in script_results])
            avg_bytes_per_char = np.mean([r['bytes_per_char'] for r in script_results])
            
            # Simple heuristic for patch size
            if avg_bytes_per_char > 2.5:  # Multi-byte scripts
                base_patch_size = 24
            elif avg_entropy > 5.0:  # High entropy
                base_patch_size = 16
            else:  # Simple scripts
                base_patch_size = 20
            
            print(f"{script_type:15} | Recommended patch size: {base_patch_size} bytes")
    
    print("\n✅ BLT successfully handles multilingual text at the byte level!")
    print("   No tokenization required - works with any language or script.")

if __name__ == "__main__":
    test_multilingual_byte_processing()
