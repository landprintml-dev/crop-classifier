"""
🧪 Local Test Script
Quick test to verify everything works before deployment
"""

import sys
import os

def test_imports():
    """Test all required imports"""
    print("📦 Testing imports...")
    try:
        import flask
        print("  ✅ Flask")
        import torch
        print("  ✅ PyTorch")
        import ee
        print("  ✅ Earth Engine API")
        import numpy
        print("  ✅ NumPy")
        import pickle
        print("  ✅ Pickle")
        return True
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        print("\n  Run: pip install -r requirements.txt")
        return False


def test_files():
    """Test required files exist"""
    print("\n📁 Checking required files...")
    required_files = [
        'app.py',
        'crop_classifier_best.pt',
        'scaler.pkl',
        'templates/index.html',
        'requirements.txt'
    ]
    
    all_exist = True
    for file in required_files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"  ✅ {file} ({size:,} bytes)")
        else:
            print(f"  ❌ {file} - NOT FOUND")
            all_exist = False
    
    return all_exist


def test_gee_auth():
    """Test GEE authentication"""
    print("\n🌍 Testing Google Earth Engine...")
    try:
        import ee
        try:
            ee.Initialize()
            print("  ✅ GEE authenticated")
            return True
        except Exception as e:
            print(f"  ❌ GEE not authenticated: {e}")
            print("\n  Run: earthengine authenticate")
            return False
    except ImportError:
        print("  ❌ earthengine-api not installed")
        return False


def test_model_load():
    """Test model can be loaded"""
    print("\n🤖 Testing model loading...")
    try:
        import torch
        checkpoint = torch.load('crop_classifier_best.pt', map_location='cpu')
        
        print(f"  ✅ Model loaded")
        print(f"     - Classes: {checkpoint.get('class_names', 'N/A')}")
        print(f"     - Model type: {checkpoint.get('model_name', 'N/A')}")
        print(f"     - Features: {checkpoint.get('n_features', 'N/A')}")
        print(f"     - Timesteps: {checkpoint.get('n_timesteps', 'N/A')}")
        return True
    except Exception as e:
        print(f"  ❌ Model loading failed: {e}")
        return False


def main():
    """Run all tests"""
    print("="*60)
    print("🧪 PRE-DEPLOYMENT TEST SUITE")
    print("="*60)
    print()
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Files", test_files()))
    results.append(("GEE Auth", test_gee_auth()))
    results.append(("Model Load", test_model_load()))
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} - {name}")
    
    all_passed = all(r[1] for r in results)
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("="*60)
        print("\n✅ Ready to deploy!")
        print("\nNext steps:")
        print("  1. Test locally: python app.py")
        print("  2. Read: DEPLOYMENT.md")
        print("  3. Deploy to Railway or Vercel")
    else:
        print("⚠️  SOME TESTS FAILED")
        print("="*60)
        print("\n❌ Fix the issues above before deploying")
    
    print()
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())

