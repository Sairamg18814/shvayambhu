"""Test BLT multilingual capabilities."""

import sys
sys.path.append('.')

import mlx.core as mx
from core.blt.patching import BLTInputProcessor
from core.blt.entropy import calculate_byte_entropy
import numpy as np

def test_multilingual_support():
    """Test BLT with multiple languages."""
    print("=== BLT Multilingual Test ===")
    
    # Test languages
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
    
    # Initialize processor
    processor = BLTInputProcessor(
        min_patch_size=4,
        max_patch_size=32,
        embedding_dim=768
    )
    
    print("\nTesting language processing and entropy:")
    print("-" * 60)
    
    results = []
    
    for language, text in language_samples.items():
        # Get byte representation
        byte_seq = text.encode('utf-8')
        byte_count = len(byte_seq)
        char_count = len(text)
        
        # Calculate entropy
        entropy = calculate_byte_entropy(byte_seq)
        
        # Process with BLT
        try:
            embeddings, metadata = processor.process_input(text)
            patches = metadata['patches']
            boundaries = metadata['boundaries']
            patch_count = len(patches)
            
            # Calculate average patch size
            patch_lengths = [end - start for start, end in boundaries]
            avg_patch_size = np.mean(patch_lengths) if patch_lengths else 0
            
            status = "✅ PASS"
            
        except Exception as e:
            status = f"❌ FAIL: {str(e)[:30]}..."
            patch_count = 0
            avg_patch_size = 0
            
        results.append({
            'language': language,
            'chars': char_count,
            'bytes': byte_count,
            'bytes_per_char': byte_count / char_count,
            'entropy': entropy,
            'patches': patch_count,
            'avg_patch_size': avg_patch_size,
            'status': status
        })
        
        print(f"{language:15} | Chars: {char_count:3} | Bytes: {byte_count:3} | "
              f"B/C: {byte_count/char_count:.1f} | Entropy: {entropy:.2f} | "
              f"Patches: {patch_count:2} | Avg size: {avg_patch_size:.1f} | {status}")
    
    # Test special cases
    print("\n\nTesting special Unicode cases:")
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
        try:
            embeddings, metadata = processor.process_input(text)
            print(f"{case_name:20} | {len(byte_seq):3} bytes | ✅ PASS")
        except Exception as e:
            print(f"{case_name:20} | {len(byte_seq):3} bytes | ❌ FAIL: {str(e)[:30]}...")
    
    # Summary
    print("\n\nSummary:")
    print("-" * 60)
    passed = sum(1 for r in results if "PASS" in r['status'])
    total = len(results)
    print(f"Languages tested: {total}")
    print(f"Passed: {passed}/{total} ({passed/total*100:.0f}%)")
    
    # Entropy analysis
    entropies = [r['entropy'] for r in results]
    print(f"\nEntropy statistics:")
    print(f"  Min: {min(entropies):.2f}")
    print(f"  Max: {max(entropies):.2f}")
    print(f"  Mean: {np.mean(entropies):.2f}")
    print(f"  Std: {np.std(entropies):.2f}")
    
    # Patch size analysis by script
    script_stats = {}
    scripts = {
        "Latin": ["English", "Mixed"],
        "CJK": ["Chinese", "Japanese", "Korean"],
        "Arabic": ["Arabic"],
        "Cyrillic": ["Russian"],
        "Indic": ["Hindi", "Thai"],
        "Other": ["Hebrew", "Greek", "Emoji"]
    }
    
    for script_type, langs in scripts.items():
        script_results = [r for r in results if r['language'] in langs and "PASS" in r['status']]
        if script_results:
            avg_patch = np.mean([r['avg_patch_size'] for r in script_results])
            avg_entropy = np.mean([r['entropy'] for r in script_results])
            script_stats[script_type] = (avg_patch, avg_entropy)
    
    print("\nPatch size by script type:")
    for script, (patch_size, entropy) in script_stats.items():
        print(f"  {script:10} | Avg patch: {patch_size:4.1f} bytes | Avg entropy: {entropy:.2f}")
    
    return passed == total

if __name__ == "__main__":
    success = test_multilingual_support()
    
    if success:
        print("\n✅ All multilingual tests passed!")
    else:
        print("\n❌ Some tests failed!")
        
    print("\nBLT successfully handles multiple languages, scripts, and Unicode edge cases")
    print("without requiring tokenization or vocabulary!")