#!/usr/bin/env python3
"""
Test script for enhanced data preparation functionality
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all required modules can be imported"""
    try:
        from data_prep import (
            validate_data_schema, 
            validate_data_quality, 
            advanced_imputation,
            EXPECTED_SCHEMA
        )
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_schema_definition():
    """Test that the expected schema is properly defined"""
    try:
        from data_prep import EXPECTED_SCHEMA
        
        # Check that schema has expected fields
        expected_fields = [
            "PassengerId", "Survived", "Pclass", "Name", "Sex", 
            "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"
        ]
        
        actual_fields = [field.name for field in EXPECTED_SCHEMA.fields]
        
        if set(actual_fields) == set(expected_fields):
            print("✅ Schema definition is correct")
            return True
        else:
            print(f"❌ Schema mismatch. Expected: {expected_fields}, Got: {actual_fields}")
            return False
            
    except Exception as e:
        print(f"❌ Schema test failed: {e}")
        return False

def test_function_definitions():
    """Test that all required functions are defined"""
    try:
        from data_prep import (
            preprocess_titanic,
            validate_data_schema,
            validate_data_quality,
            advanced_imputation
        )
        
        # Check if functions are callable
        functions = [
            preprocess_titanic,
            validate_data_schema,
            validate_data_quality,
            advanced_imputation
        ]
        
        for func in functions:
            if not callable(func):
                print(f"❌ {func.__name__} is not callable")
                return False
        
        print("✅ All functions are properly defined and callable")
        return True
        
    except Exception as e:
        print(f"❌ Function definition test failed: {e}")
        return False

def test_directory_structure():
    """Test that required directories exist or can be created"""
    try:
        required_dirs = [
            "data/raw",
            "data/processed", 
            "models",
            "reports/data_validation"
        ]
        
        for dir_path in required_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            if Path(dir_path).exists():
                print(f"✅ Directory {dir_path} exists")
            else:
                print(f"❌ Failed to create directory {dir_path}")
                return False
        
        print("✅ All required directories are accessible")
        return True
        
    except Exception as e:
        print(f"❌ Directory structure test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Enhanced Data Preparation Module")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Schema Definition Test", test_schema_definition),
        ("Function Definition Test", test_function_definitions),
        ("Directory Structure Test", test_directory_structure)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The enhanced data preparation module is ready.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
