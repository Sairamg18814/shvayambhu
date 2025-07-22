"""Test BLT module imports."""

import sys
sys.path.append('.')

try:
    from core.blt.pipeline import BLTPipeline
    print("✅ BLT pipeline imported successfully")
    
    from core.blt.encoder import LocalEncoder
    print("✅ LocalEncoder imported successfully")
    
    from core.blt.transformer import LatentTransformer
    print("✅ LatentTransformer imported successfully")
    
    from core.blt.decoder import LocalDecoder
    print("✅ LocalDecoder imported successfully")
    
    # Test creating a simple model
    model = BLTPipeline(
        embedding_dim=256,
        hidden_dim=512,
        transformer_layers=4,
        transformer_heads=8
    )
    print("✅ BLT model created successfully")
    
    # Get model info
    info = model.get_model_info()
    print(f"\nModel info:")
    print(f"- Total parameters: {info['total_parameters']:,}")
    print(f"- Model size: {info['model_size_mb']:.1f} MB")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
