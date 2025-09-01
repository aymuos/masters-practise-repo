# Enhanced Data Preparation for Titanic MLOps Project

## Overview

This enhanced data preparation module provides comprehensive data validation, quality checks, and sophisticated imputation strategies for the Titanic dataset. It goes beyond basic preprocessing to ensure data integrity and create meaningful features for machine learning.

## Features

### 🔍 **Data Validation**
- **Schema Validation**: Ensures data structure matches expected format
- **Data Quality Checks**: Comprehensive analysis of missing values, outliers, and data integrity
- **Value Range Validation**: Checks for unrealistic values (e.g., negative ages, extreme fares)
- **Categorical Value Validation**: Ensures categorical variables contain expected values

### 🛠️ **Advanced Imputation**
- **Contextual Age Imputation**: Uses median by Sex + Pclass (more accurate than simple median)
- **Stratified Fare Imputation**: Imputes by passenger class
- **Smart Categorical Imputation**: Mode-based imputation for categorical variables
- **Feature Engineering**: Creates new meaningful features from existing data

### 📊 **Simplified Encoding Strategy**
- **Label Encoding**: Uses StringIndexer for categorical variables (simpler than one-hot encoding)
- **Feature Assembly**: Combines numerical and indexed categorical features
- **Pipeline Persistence**: Saves preprocessing pipeline for consistent transformations

## Key Improvements

### **1. Data Type Safety**
- Explicit casting of Age and Fare columns to DoubleType
- Prevents string data type errors during imputation
- Ensures numerical operations work correctly

### **2. Simplified Feature Engineering**
- **Cabin_known**: Binary indicator (0/1) for cabin availability
- **FamilySize**: SibSp + Parch + 1 (total family members)
- **IsAlone**: Binary indicator for single passengers
- **Title**: Extracted and grouped passenger titles

### **3. Robust Imputation Pipeline**
- **Age**: Median by Sex + Pclass → Overall median fallback
- **Fare**: Median by Pclass
- **Embarked**: Mode (most frequent value)
- **Cabin**: Binary feature + "Unknown" for missing values

## Usage

### **Basic Usage**
```python
from src.data_prep import preprocess_titanic

# Run complete preprocessing pipeline
df_processed, preproc_model = preprocess_titanic()
```

### **Custom Paths**
```python
# Custom input/output paths
df_processed, preproc_model = preprocess_titanic(
    raw_csv="path/to/raw.csv",
    out_parquet="path/to/output.parquet",
    pipeline_out_dir="path/to/pipeline"
)
```

## Output Files

### **Processed Data**
- `data/processed/train_processed.parquet`: Clean, imputed dataset
- `models/preprocess_pipeline`: Fitted preprocessing pipeline

### **Validation Reports**
- `reports/data_validation/data_quality_report.json`: Comprehensive quality analysis
- `reports/data_validation/processing_summary.json`: Processing statistics

## Data Quality Report

The quality report includes:
- Missing value counts and percentages
- Value range statistics (min, max, mean, stddev)
- Data type validation
- Quality issue identification
- Categorical value validation

## Processing Summary

The processing summary tracks:
- Input/output row counts
- Feature counts (categorical vs numerical)
- Quality issues found
- Imputation status
- Encoding strategy used

## Error Handling

- **Schema Validation**: Continues processing even if schema doesn't match exactly
- **Data Type Issues**: Automatically casts problematic columns to correct types
- **Missing Values**: Handles missing values gracefully with appropriate imputation
- **Pipeline Failures**: Comprehensive error logging and graceful degradation

## Testing

Run the test suite to verify functionality:
```bash
python test_simple_data_prep.py
```

## Dependencies

- **PySpark**: For distributed data processing
- **Pathlib**: For file path handling
- **JSON**: For report generation
- **Logging**: For comprehensive logging

## Benefits of Simplified Approach

### **1. Performance**
- Faster processing without one-hot encoding
- Reduced memory usage
- Simpler pipeline execution

### **2. Interpretability**
- Label encoding preserves ordinal relationships
- Easier to understand feature importance
- More intuitive for business stakeholders

### **3. Robustness**
- Fewer failure points in the pipeline
- Better handling of edge cases
- More stable across different datasets

## Future Enhancements

1. **Feature Selection**: Add correlation-based feature selection
2. **Outlier Detection**: Implement statistical outlier detection
3. **Cross-Validation**: Add cross-validation for imputation strategies
4. **Automated Tuning**: Auto-tune imputation parameters
5. **Additional Features**: Create more domain-specific features

## Troubleshooting

### **Common Issues**

1. **Data Type Errors**: Ensure Age and Fare columns are numeric
2. **Missing Dependencies**: Install PySpark and required packages
3. **Memory Issues**: Reduce batch size for large datasets
4. **Schema Mismatches**: Check column names and types in raw data

### **Debug Mode**
Enable detailed logging by setting log level to DEBUG in your logging configuration.

## Contributing

When adding new features:
1. Follow the existing code structure
2. Add comprehensive logging
3. Include error handling
4. Update validation reports
5. Document new features
6. Update DVC pipeline if new outputs are added
7. Test thoroughly with the test suite
