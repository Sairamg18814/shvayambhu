"""Multilingual testing for BLT architecture.

This module tests BLT's ability to handle multiple languages, scripts,
and mixed-language documents without tokenization.
"""

import pytest
import torch
import numpy as np
from typing import List, Dict, Tuple
import unicodedata

from shvayambhu.core.blt.pipeline import BLTPipeline
from shvayambhu.core.blt.entropy import calculate_byte_entropy
from shvayambhu.core.blt.patching import BLTInputProcessor


class MultilingualTestSuite:
    """Comprehensive multilingual testing for BLT."""
    
    def __init__(self):
        self.config = {
            "vocab_size": 256,
            "hidden_dim": 768,
            "num_layers": 6,
            "num_heads": 12,
            "patch_embedding_dim": 768,
            "max_patch_size": 32,
            "device": torch.device("cpu")
        }
        self.pipeline = BLTPipeline(self.config)
        self.pipeline.eval()
        
        # Define test languages with sample texts
        self.language_samples = {
            # Major languages
            "English": {
                "text": "The quick brown fox jumps over the lazy dog.",
                "script": "Latin",
                "direction": "LTR"
            },
            "Chinese (Simplified)": {
                "text": "快速的棕色狐狸跳过了懒狗。",
                "script": "Han",
                "direction": "LTR"
            },
            "Chinese (Traditional)": {
                "text": "快速的棕色狐狸跳過了懶狗。",
                "script": "Han",
                "direction": "LTR"
            },
            "Spanish": {
                "text": "El rápido zorro marrón salta sobre el perro perezoso.",
                "script": "Latin",
                "direction": "LTR"
            },
            "Hindi": {
                "text": "तेज़ भूरी लोमड़ी आलसी कुत्ते के ऊपर कूदती है।",
                "script": "Devanagari",
                "direction": "LTR"
            },
            "Arabic": {
                "text": "الثعلب البني السريع يقفز فوق الكلب الكسول.",
                "script": "Arabic",
                "direction": "RTL"
            },
            "Russian": {
                "text": "Быстрая коричневая лиса прыгает через ленивую собаку.",
                "script": "Cyrillic",
                "direction": "LTR"
            },
            "Japanese": {
                "text": "速い茶色のキツネが怠け者の犬を飛び越える。",
                "script": "Mixed (Hiragana/Katakana/Kanji)",
                "direction": "LTR"
            },
            "Korean": {
                "text": "빠른 갈색 여우가 게으른 개를 뛰어넘는다.",
                "script": "Hangul",
                "direction": "LTR"
            },
            "Hebrew": {
                "text": "השועל החום המהיר קופץ מעל הכלב העצלן.",
                "script": "Hebrew",
                "direction": "RTL"
            },
            "Thai": {
                "text": "สุนัขจิ้งจอกสีน้ำตาลที่รวดเร็วกระโดดข้ามสุนัขขี้เกียจ",
                "script": "Thai",
                "direction": "LTR"
            },
            "Greek": {
                "text": "Η γρήγορη καφέ αλεπού πηδά πάνω από το τεμπέλικο σκυλί.",
                "script": "Greek",
                "direction": "LTR"
            },
            "Turkish": {
                "text": "Hızlı kahverengi tilki tembel köpeğin üzerinden atlar.",
                "script": "Latin",
                "direction": "LTR"
            },
            "Vietnamese": {
                "text": "Con cáo nâu nhanh nhẹn nhảy qua con chó lười.",
                "script": "Latin",
                "direction": "LTR"
            },
            "Bengali": {
                "text": "দ্রুত বাদামী শিয়াল অলস কুকুরের উপর দিয়ে লাফ দেয়।",
                "script": "Bengali",
                "direction": "LTR"
            }
        }
        
        # Special test cases
        self.special_cases = {
            "Emoji": "🦊🏃‍♂️🦮😴",
            "Math": "∀x∈ℝ: x² ≥ 0",
            "Mixed Script": "Hello世界مرحبا",
            "Zero-width": "Hello\u200bWorld",  # Zero-width space
            "Combining": "é = e\u0301",  # Combining acute accent
            "RTL-LTR Mix": "Hello שלום World",
            "Surrogate Pairs": "𝐇𝐞𝐥𝐥𝐨 𝕎𝕠𝕣𝕝𝕕",  # Mathematical alphanumeric
        }
    
    def test_language_preservation(self):
        """Test that each language is preserved correctly."""
        print("\n=== Language Preservation Test ===")
        
        results = {}
        
        for language, data in self.language_samples.items():
            text = data["text"]
            byte_seq = text.encode('utf-8')
            
            # Process through pipeline
            output = self.pipeline.process_bytes(byte_seq, mode='inference')
            reconstructed = output['reconstructed_text']
            
            # Check preservation
            preserved = reconstructed == text
            results[language] = preserved
            
            print(f"{language}: {'PASS' if preserved else 'FAIL'}")
            if not preserved:
                print(f"  Original: {text}")
                print(f"  Got: {reconstructed}")
        
        # All languages should be preserved
        assert all(results.values()), f"Failed languages: {[k for k, v in results.items() if not v]}"
    
    def test_script_entropy_patterns(self):
        """Test entropy patterns across different scripts."""
        print("\n=== Script Entropy Patterns ===")
        
        entropy_by_script = {}
        
        for language, data in self.language_samples.items():
            text = data["text"]
            script = data["script"]
            byte_seq = text.encode('utf-8')
            
            # Calculate entropy
            entropy = calculate_byte_entropy(byte_seq)
            
            if script not in entropy_by_script:
                entropy_by_script[script] = []
            entropy_by_script[script].append((language, entropy))
        
        # Print entropy patterns
        for script, entries in entropy_by_script.items():
            avg_entropy = np.mean([e[1] for e in entries])
            print(f"\n{script}:")
            for lang, entropy in entries:
                print(f"  {lang}: {entropy:.3f}")
            print(f"  Average: {avg_entropy:.3f}")
    
    def test_mixed_language_documents(self):
        """Test documents containing multiple languages."""
        print("\n=== Mixed Language Documents ===")
        
        mixed_documents = [
            # Code switching
            "Hello world! 你好世界！ مرحبا بالعالم!",
            
            # Technical document
            "The function f(x) = x² is called 二次函数 in Chinese.",
            
            # Social media style
            "Just arrived in Tokyo! 東京に着きました！🗼",
            
            # Academic citation
            "As noted by García (2023), the phenomenon (现象) is widespread.",
            
            # Product description
            "Premium quality | 高品質 | جودة عالية | Высокое качество",
        ]
        
        for i, doc in enumerate(mixed_documents):
            byte_seq = doc.encode('utf-8')
            output = self.pipeline.process_bytes(byte_seq, mode='inference')
            reconstructed = output['reconstructed_text']
            
            preserved = reconstructed == doc
            print(f"Document {i+1}: {'PASS' if preserved else 'FAIL'}")
            if not preserved:
                print(f"  Original: {doc}")
                print(f"  Got: {reconstructed}")
    
    def test_special_unicode_cases(self):
        """Test special Unicode cases."""
        print("\n=== Special Unicode Cases ===")
        
        for case_name, text in self.special_cases.items():
            byte_seq = text.encode('utf-8')
            output = self.pipeline.process_bytes(byte_seq, mode='inference')
            reconstructed = output['reconstructed_text']
            
            preserved = reconstructed == text
            print(f"{case_name}: {'PASS' if preserved else 'FAIL'}")
            
            # Special validation for certain cases
            if case_name == "Combining":
                # Check both forms are handled
                nfc_form = unicodedata.normalize('NFC', text)
                nfd_form = unicodedata.normalize('NFD', text)
                print(f"  NFC preserved: {reconstructed == nfc_form}")
                print(f"  NFD preserved: {reconstructed == nfd_form}")
    
    def test_patching_across_scripts(self):
        """Test how patching adapts to different scripts."""
        print("\n=== Patching Across Scripts ===")
        
        processor = BLTInputProcessor(
            min_patch_size=4,
            max_patch_size=32,
            embedding_dim=768
        )
        
        patch_stats = {}
        
        for language, data in self.language_samples.items():
            text = data["text"]
            script = data["script"]
            
            # Get patches
            embeddings, metadata = processor.process_input(text)
            patches = metadata['patches']
            boundaries = metadata['boundaries']
            
            # Calculate statistics
            patch_lengths = [end - start for start, end in boundaries]
            avg_patch_size = np.mean(patch_lengths)
            
            if script not in patch_stats:
                patch_stats[script] = []
            patch_stats[script].append((language, avg_patch_size, len(patches)))
        
        # Print patch statistics
        for script, stats in patch_stats.items():
            avg_size = np.mean([s[1] for s in stats])
            print(f"\n{script}:")
            for lang, size, count in stats:
                print(f"  {lang}: {size:.1f} bytes/patch, {count} patches")
            print(f"  Average patch size: {avg_size:.1f} bytes")
    
    def test_direction_handling(self):
        """Test handling of different text directions."""
        print("\n=== Text Direction Handling ===")
        
        # Test RTL languages
        rtl_languages = [
            (lang, data) for lang, data in self.language_samples.items() 
            if data["direction"] == "RTL"
        ]
        
        for language, data in rtl_languages:
            text = data["text"]
            byte_seq = text.encode('utf-8')
            
            # Process
            output = self.pipeline.process_bytes(byte_seq, mode='inference')
            reconstructed = output['reconstructed_text']
            
            # Check preservation
            preserved = reconstructed == text
            print(f"{language} (RTL): {'PASS' if preserved else 'FAIL'}")
            
            # Test mixed direction
            mixed = f"Start {text} End"
            mixed_output = self.pipeline.process_bytes(mixed.encode('utf-8'), mode='inference')
            mixed_preserved = mixed_output['reconstructed_text'] == mixed
            print(f"{language} (Mixed): {'PASS' if mixed_preserved else 'FAIL'}")
    
    def test_normalization_consistency(self):
        """Test Unicode normalization consistency."""
        print("\n=== Normalization Consistency ===")
        
        # Test different normalization forms
        test_strings = [
            "café",  # Combining characters
            "ñ",     # Precomposed
            "가",    # Korean syllable
            "é",     # Accented character
        ]
        
        for text in test_strings:
            # Test all normalization forms
            forms = {
                'NFC': unicodedata.normalize('NFC', text),
                'NFD': unicodedata.normalize('NFD', text),
                'NFKC': unicodedata.normalize('NFKC', text),
                'NFKD': unicodedata.normalize('NFKD', text),
            }
            
            print(f"\nTesting: {text}")
            for form_name, normalized in forms.items():
                byte_seq = normalized.encode('utf-8')
                output = self.pipeline.process_bytes(byte_seq, mode='inference')
                reconstructed = output['reconstructed_text']
                
                preserved = reconstructed == normalized
                print(f"  {form_name}: {'PASS' if preserved else 'FAIL'}")
    
    def test_byte_efficiency_by_language(self):
        """Test byte-level efficiency across languages."""
        print("\n=== Byte Efficiency by Language ===")
        
        # Compare byte usage
        for language, data in self.language_samples.items():
            text = data["text"]
            
            # Get byte representation
            byte_seq = text.encode('utf-8')
            char_count = len(text)
            byte_count = len(byte_seq)
            bytes_per_char = byte_count / char_count
            
            # Process and check overhead
            output = self.pipeline.process_bytes(byte_seq, mode='inference')
            
            print(f"{language}:")
            print(f"  Characters: {char_count}")
            print(f"  Bytes: {byte_count}")
            print(f"  Bytes/char: {bytes_per_char:.2f}")
            print(f"  Patches: {len(output.get('patches', []))}")


class TestMultilingual:
    """Pytest test cases for multilingual support."""
    
    @pytest.fixture
    def test_suite(self):
        """Create multilingual test suite."""
        return MultilingualTestSuite()
    
    def test_all_languages_preserved(self, test_suite):
        """Test that all languages are preserved correctly."""
        for language, data in test_suite.language_samples.items():
            text = data["text"]
            byte_seq = text.encode('utf-8')
            
            output = test_suite.pipeline.process_bytes(byte_seq, mode='inference')
            reconstructed = output['reconstructed_text']
            
            assert reconstructed == text, f"Failed to preserve {language}"
    
    def test_mixed_scripts(self, test_suite):
        """Test mixed script handling."""
        mixed_text = "Hello 世界 مرحبا мир"
        byte_seq = mixed_text.encode('utf-8')
        
        output = test_suite.pipeline.process_bytes(byte_seq, mode='inference')
        reconstructed = output['reconstructed_text']
        
        assert reconstructed == mixed_text
    
    def test_emoji_preservation(self, test_suite):
        """Test emoji handling."""
        emoji_text = "Hello 👋 World 🌍!"
        byte_seq = emoji_text.encode('utf-8')
        
        output = test_suite.pipeline.process_bytes(byte_seq, mode='inference')
        reconstructed = output['reconstructed_text']
        
        assert reconstructed == emoji_text
    
    def test_rtl_languages(self, test_suite):
        """Test right-to-left languages."""
        rtl_samples = [
            ("Arabic", test_suite.language_samples["Arabic"]["text"]),
            ("Hebrew", test_suite.language_samples["Hebrew"]["text"]),
        ]
        
        for lang, text in rtl_samples:
            byte_seq = text.encode('utf-8')
            output = test_suite.pipeline.process_bytes(byte_seq, mode='inference')
            reconstructed = output['reconstructed_text']
            assert reconstructed == text, f"Failed RTL language: {lang}"
    
    def test_zero_width_characters(self, test_suite):
        """Test zero-width and invisible characters."""
        # Zero-width space
        text_zws = "Hello\u200bWorld"
        output = test_suite.pipeline.process_bytes(text_zws.encode('utf-8'), mode='inference')
        assert output['reconstructed_text'] == text_zws
        
        # Zero-width joiner
        text_zwj = "👨‍👩‍👧‍👦"  # Family emoji with ZWJ
        output = test_suite.pipeline.process_bytes(text_zwj.encode('utf-8'), mode='inference')
        assert output['reconstructed_text'] == text_zwj
    
    def test_surrogate_pairs(self, test_suite):
        """Test surrogate pair handling."""
        # Mathematical alphanumeric symbols
        text = "𝐇𝐞𝐥𝐥𝐨 𝕎𝕠𝕣𝕝𝕕"
        output = test_suite.pipeline.process_bytes(text.encode('utf-8'), mode='inference')
        assert output['reconstructed_text'] == text
    
    def test_language_batch_processing(self, test_suite):
        """Test batch processing with multiple languages."""
        texts = [data["text"] for data in test_suite.language_samples.values()][:5]
        byte_sequences = [text.encode('utf-8') for text in texts]
        
        outputs = test_suite.pipeline.process_batch(byte_sequences, mode='inference')
        
        for i, text in enumerate(texts):
            assert outputs['reconstructed_texts'][i] == text


def run_multilingual_tests():
    """Run complete multilingual test suite."""
    suite = MultilingualTestSuite()
    
    print("Running Multilingual Test Suite...")
    print("=" * 60)
    
    suite.test_language_preservation()
    suite.test_script_entropy_patterns()
    suite.test_mixed_language_documents()
    suite.test_special_unicode_cases()
    suite.test_patching_across_scripts()
    suite.test_direction_handling()
    suite.test_normalization_consistency()
    suite.test_byte_efficiency_by_language()
    
    print("\n" + "=" * 60)
    print("Multilingual tests completed!")


if __name__ == "__main__":
    run_multilingual_tests()