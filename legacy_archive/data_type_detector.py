import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union

try:
    import dask.dataframe as dd
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

class DataTypeDetector:
    """Advanced data type detection with confidence scoring (Dask Supported)"""
    
    def __init__(self):
        self.confidence_threshold = 0.8
        self.numeric_threshold = 0.95
    
    def detect_column_types(self, df) -> Dict[str, Dict]:
        """Detect column types with confidence scores and recommendations"""
        results = {}
        
        # Dukungan Out-of-Core Dask:
        # Untuk efisiensi mendeteksi tipe data pada dataset masif, 
        # kita mengambil 10,000 baris pertama sebagai representasi statistik.
        is_dask = DASK_AVAILABLE and isinstance(df, dd.DataFrame)
        
        if is_dask:
            df_sample = df.head(10000) # .head() pada Dask me-return Pandas DataFrame
        else:
            df_sample = df
            
        for column in df_sample.columns:
            results[column] = self.analyze_series(df_sample[column], column)
        
        return results

    def analyze_series(self, series: pd.Series, column_name: str = None) -> Dict:
        """Analyze a single series and return results"""
        # Basic type detection
        basic_type = self._get_basic_type(series)
        
        # Advanced analysis
        advanced_analysis = self._analyze_column_characteristics(series)
        
        # Confidence scoring
        confidence = self._calculate_confidence_score(series, basic_type, advanced_analysis)
        
        # Recommendations
        recommendations = self._generate_recommendations(series, basic_type, advanced_analysis)
        
        return {
            'detected_type': basic_type,
            'confidence': confidence,
            'analysis': advanced_analysis,
            'recommendations': recommendations,
            'sample_values': series.dropna().head(10).tolist(),
            'null_percentage': advanced_analysis.get('null_percentage', 0),
            'unique_count': advanced_analysis.get('unique_count', 0),
            'null_count': advanced_analysis.get('null_count', 0),
            'column_name': column_name
        }
    
    def _get_basic_type(self, series: pd.Series) -> str:
        """Get basic pandas dtype"""
        dtype = str(series.dtype)
        
        if 'int' in dtype or 'float' in dtype:
            return 'numeric'
        elif 'bool' in dtype:
            return 'boolean'
        elif 'datetime' in dtype:
            return 'datetime'
        elif 'category' in dtype:
            return 'categorical'
        else:
            return 'object'
    
    def _analyze_column_characteristics(self, series: pd.Series) -> Dict:
        """Analyze detailed column characteristics"""
        analysis = {}
        
        # Basic statistics
        analysis['total_values'] = len(series)
        analysis['null_count'] = series.isnull().sum()
        analysis['null_percentage'] = analysis['null_count'] / len(series) if len(series) > 0 else 0
        analysis['unique_count'] = series.nunique()
        analysis['unique_ratio'] = analysis['unique_count'] / len(series) if len(series) > 0 else 0
        
        if analysis['unique_count'] > 0:
            # Value distribution
            value_counts = series.value_counts()
            analysis['most_frequent'] = value_counts.index[0] if len(value_counts) > 0 else None
            analysis['frequency_ratio'] = value_counts.iloc[0] / len(series) if len(value_counts) > 0 else 0
            
            # Pattern analysis for object columns
            if str(series.dtype) == 'object':
                analysis['avg_length'] = series.astype(str).str.len().mean()
                analysis['has_numbers'] = series.astype(str).str.contains(r'\d').any()
                analysis['has_special_chars'] = series.astype(str).str.contains(r'[^\w\s]').any()
                
                # Try to parse as datetime
                try:
                    pd.to_datetime(series.dropna())
                    analysis['is_datetime'] = True
                except:
                    analysis['is_datetime'] = False
            
            # Check for datetime dtype
            elif 'datetime' in str(series.dtype):
                analysis['is_datetime'] = True
            
            # Numeric analysis
            if str(series.dtype) in ['int64', 'float64']:
                analysis['min'] = series.min()
                analysis['max'] = series.max()
                analysis['mean'] = series.mean()
                analysis['std'] = series.std()
                analysis['skewness'] = series.skew()
                analysis['kurtosis'] = series.kurtosis()
                
                # Check for potential categorical encoding
                if analysis['unique_count'] <= 20 and analysis['unique_ratio'] < 0.1:
                    analysis['potential_categorical'] = True
        
        return analysis
    
    def _calculate_confidence_score(self, series: pd.Series, basic_type: str, analysis: Dict) -> float:
        """Calculate confidence score for type detection"""
        confidence = 0.7  # Base confidence (increased from 0.5)
        
        # Special handling for datetime columns
        if analysis.get('is_datetime', False):
            confidence += 0.4  # High bonus for successful datetime parsing
            # Don't penalize high unique ratio for datetime (it's expected)
            if analysis['unique_ratio'] > 0.9:
                confidence += 0.2  # Bonus instead of penalty for datetime
            return max(0.0, min(1.0, confidence))
        
        # Special handling for numeric columns with high unique ratio
        if basic_type == 'numeric' and analysis['unique_ratio'] > 0.9:
            # For numeric columns, high unique ratio is actually good (not ID-like)
            confidence += 0.2  # Bonus instead of penalty
        elif analysis['unique_ratio'] > 0.9:
            confidence -= 0.2  # Penalty for non-numeric high unique ratio
        
        # Low unique ratio with few unique values suggests categorical
        if analysis['unique_ratio'] < 0.1 and analysis['unique_count'] <= 20:
            confidence += 0.2  # Reduced bonus
        
        # High frequency of one value suggests categorical
        if analysis.get('frequency_ratio', 0) > 0.5:
            confidence += 0.1  # Reduced bonus
        
        # Object type with numbers might be mixed
        if basic_type == 'object' and analysis.get('has_numbers', False):
            confidence -= 0.1  # Reduced penalty
        
        # For numeric types, check if values are reasonable
        if basic_type == 'numeric':
            confidence += 0.2  # Base bonus for numeric types
            if analysis.get('std', 0) > 0:  # Has variation
                confidence += 0.1
            if analysis.get('skewness', 0) < 10:  # Not extremely skewed
                confidence += 0.1
            # Additional checks for good numeric data
            if analysis.get('min', float('inf')) != analysis.get('max', float('-inf')):  # Has range
                confidence += 0.1
            if analysis['null_percentage'] < 0.1:  # Low null percentage
                confidence += 0.1
        
        # For categorical, check if it makes sense
        if basic_type == 'object' and analysis['unique_count'] <= 20 and analysis['unique_ratio'] < 0.5:
            confidence += 0.2
        
        return max(0.0, min(1.0, confidence))
    
    def _generate_recommendations(self, series: pd.Series, basic_type: str, analysis: Dict) -> List[str]:
        """Generate data type recommendations"""
        recommendations = []
        
        # High null percentage
        if analysis['null_percentage'] > 0.5:
            recommendations.append("Consider dropping this column due to high missing values")
        
        # Potential categorical encoding
        if analysis.get('potential_categorical', False):
            recommendations.append("Consider treating as categorical variable")
        
        # High unique ratio (possible ID column)
        if analysis['unique_ratio'] > 0.9:
            recommendations.append("This appears to be an ID column - consider excluding from analysis")
        
        # Low unique count with high frequency
        if analysis['unique_count'] <= 2 and analysis.get('frequency_ratio', 0) > 0.9:
            recommendations.append("Consider as binary/boolean variable")
        
        # Object type recommendations
        if basic_type == 'object':
            if analysis.get('is_datetime', False):
                recommendations.append("Convert to datetime format")
            elif not analysis.get('has_special_chars', False) and analysis.get('avg_length', 0) < 20:
                recommendations.append("Consider categorical encoding")
        
        return recommendations
    
    def get_column_classification(self, df: pd.DataFrame, confidence_threshold: float = None) -> Tuple[List[str], List[str], List[str]]:
        """Get final column classification with filtering"""
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold
        
        results = self.detect_column_types(df)
        
        numerical = []
        categorical = []
        datetime_cols = []
        
        for column, analysis in results.items():
            detected_type = analysis['detected_type']
            confidence = analysis['confidence']
            
            # Skip low confidence columns
            if confidence < confidence_threshold:
                continue
            
            if detected_type == 'numeric' and not analysis['analysis'].get('potential_categorical', False):
                numerical.append(column)
            elif detected_type == 'datetime' or analysis['analysis'].get('is_datetime', False):
                datetime_cols.append(column)
            elif detected_type in ['object', 'categorical'] or analysis['analysis'].get('potential_categorical', False):
                categorical.append(column)
        
        return numerical, categorical, datetime_cols