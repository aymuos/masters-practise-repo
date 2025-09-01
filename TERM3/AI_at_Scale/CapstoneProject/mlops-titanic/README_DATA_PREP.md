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
- **Contextual Age Imputation**: Uses median age by Sex and Pclass for more accurate estimates
- **Stratified Fare Imputation**: Imputes missing fares based on passenger class
- **Mode-based Categorical Imputation**: Fills missing categorical values with most frequent values
- **Feature Engineering from Missing Data**: Creates meaningful features from missing value patterns

### 🚀 **Feature Engineering**
- **Title Extraction**: Extracts and groups passenger titles from names
- **Family Size Features**: Creates family size and "alone" indicators
- **Cabin Information**: Preserves cabin availability information as a feature
- **Enhanced Categorical Encoding**: Comprehensive one-hot encoding with proper handling of rare categories

## File Structure

```
src/
├── data_prep.py              # Enhanced data preparation module
├── utils/
│   └── logging.py            # Logging utilities
reports/
└── data_validation/          # Data quality and validation reports
    ├── data_quality_report.json
    └── processing_summary.json
```

## Usage

### Basic Usage

```python
from src.data_prep import preprocess_titanic

# Run complete preprocessing pipeline
df_processed, preproc_model = preprocess_titanic(
    raw_csv="data/raw/train.csv",
    out_parquet="data/processed/train_processed.parquet",
    pipeline_out_dir="models/preprocess_pipeline"
)
```

### Individual Functions

```python
from src.data_prep import (
    validate_data_schema,
    validate_data_quality,
    advanced_imputation
)

# Validate schema
schema_valid = validate_data_schema(df, EXPECTED_SCHEMA)

# Check data quality
quality_report = validate_data_quality(df)

# Apply advanced imputation
df_imputed = advanced_imputation(df)
```

## Data Validation Details

### Schema Validation
The module expects the following schema for the Titanic dataset:

```python
EXPECTED_SCHEMA = StructType([
    StructField("PassengerId", IntegerType(), True),
    StructField("Survived", IntegerType(), True),
    StructField("Pclass", IntegerType(), True),
    StructField("Name", StringType(), True),
    StructField("Sex", StringType(), True),
    StructField("Age", DoubleType(), True),
    StructField("SibSp", IntegerType(), True),
    StructField("Parch", IntegerType(), True),
    StructField("Ticket", StringType(), True),
    StructField("Fare", DoubleType(), True),
    StructField("Cabin", StringType(), True),
    StructField("Embarked", StringType(), True)
])
```

### Quality Checks Performed

1. **Missing Value Analysis**
   - Count and percentage of missing values per column
   - Warning for columns with >50% missing values

2. **Value Range Validation**
   - Age: Checks for values outside 0-120 range
   - Fare: Checks for negative values or values >1000
   - Categorical: Validates expected values (e.g., Sex should be "male" or "female")

3. **Statistical Summary**
   - Min, max, mean, standard deviation for numerical columns
   - Unique value counts for categorical columns

## Imputation Strategies

### Age Imputation
1. **Primary Strategy**: Median age by Sex and Pclass combination
2. **Fallback Strategy**: Overall median age if group-specific median is unavailable

### Fare Imputation
1. **Primary Strategy**: Median fare by Pclass
2. **Contextual**: Considers passenger class for more accurate estimates

### Categorical Imputation
1. **Embarked**: Mode (most frequent value)
2. **Cabin**: "Unknown" + binary indicator for cabin availability

## Feature Engineering

### New Features Created

1. **Title**: Extracted from Name, grouped into meaningful categories
   - Common: Mr, Miss, Mrs, Master
   - Officer: Dr, Rev, Col, Major, Capt
   - Royalty: Jonkheer, Don, Sir, Countess, Lady
   - Other: Rare titles

2. **FamilySize**: SibSp + Parch + 1 (passenger themselves)

3. **IsAlone**: Binary indicator (1 if FamilySize == 1, 0 otherwise)

4. **Cabin_known**: Binary indicator for cabin availability

## Output Reports

### Data Quality Report (`data_quality_report.json`)
```json
{
  "total_rows": 891,
  "total_columns": 12,
  "missing_values": {
    "Age": {"count": 177, "percentage": 19.87},
    "Cabin": {"count": 687, "percentage": 77.10}
  },
  "quality_issues": [
    "High missing values in Cabin: 77.10%"
  ]
}
```

### Processing Summary (`processing_summary.json`)
```json
{
  "input_rows": 891,
  "output_rows": 891,
  "input_columns": 12,
  "output_columns": 2,
  "categorical_features": 6,
  "numerical_features": 5,
  "quality_issues_found": 1,
  "imputation_applied": true
}
```

## Testing

Run the test suite to validate the module:

```bash
python test_data_prep.py
```

## DVC Pipeline Integration

The enhanced preprocessing is integrated into the DVC pipeline:

```yaml
preprocess:
  cmd: python src/data_prep.py
  deps:
    - src/data_prep.py
    - src/utils/logging.py
    - data/raw/train.csv
  outs:
    - data/processed/train_processed.parquet
    - models/preprocess_pipeline
    - reports/data_validation/data_quality_report.json
    - reports/data_validation/processing_summary.json
```

## Error Handling

The module includes comprehensive error handling:

- **Schema Mismatch**: Logs warning but continues processing
- **Data Quality Issues**: Logs warnings and continues with processing
- **Imputation Failures**: Falls back to simpler strategies
- **File I/O Errors**: Graceful error handling with detailed logging

## Performance Considerations

- **Efficient Imputation**: Uses Spark's groupBy operations for contextual imputation
- **Memory Management**: Processes data in chunks for large datasets
- **Pipeline Optimization**: Single-pass preprocessing pipeline for efficiency

## Future Enhancements

1. **Advanced Outlier Detection**: Statistical methods for outlier identification
2. **Automated Feature Selection**: ML-based feature importance ranking
3. **Cross-Validation**: K-fold validation for imputation strategies
4. **Real-time Validation**: Streaming data validation capabilities

## Dependencies

- PySpark 3.5.1+
- Python 3.8+
- Standard libraries: json, pathlib

## Contributing

When adding new validation rules or imputation strategies:

1. Add comprehensive logging
2. Include error handling
3. Update test suite
4. Document new features
5. Update DVC pipeline if new outputs are added
