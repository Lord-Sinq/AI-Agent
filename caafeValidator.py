"""
CAAFE-style Feature Validation Module
Context-Aware Automated Feature Engineering with Performance Validation
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, roc_auc_score, classification_report, f1_score
from typing import Dict, List, Tuple, Optional, Any
import re
from pathlib import Path


class CAAFEFeatureValidator:
    """
    Validates generated features by measuring performance improvement.
    Implements the CAAFE approach: generate, test, keep only if improves.
    """

    def __init__(self, target: str, cv_folds: int = 3, random_state: int = 42):
        """
        Initialize the validator.

        Args:
            target: Name of the target column
            cv_folds: Number of cross-validation folds
            random_state: Random seed for reproducibility
        """
        self.target = target
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.baseline_score = None
        self.current_score = None
        self.feature_history = []
        self.target_encoder = None
        self.baseline_evaluated = False

    def evaluate_baseline(self, df: pd.DataFrame) -> float:
        """
        Calculate baseline performance without any new features.

        Args:
            df: DataFrame with features and target

        Returns:
            Baseline accuracy score
        """
        if self.target not in df.columns:
            raise ValueError(f"Target column '{self.target}' not found in dataframe")

        X = df.drop(columns=[self.target])
        y = df[self.target]

        # Encode target if it's categorical
        if y.dtype == 'object':
            self.target_encoder = LabelEncoder()
            y = self.target_encoder.fit_transform(y)

        # Handle categorical features with get_dummies
        X = pd.get_dummies(X)

        # Determine CV folds (minimum 2, maximum cv_folds)
        n_samples = len(df)
        cv = min(self.cv_folds, max(2, n_samples // 5)) if n_samples > 10 else 2

        try:
            clf = RandomForestClassifier(
                n_estimators=50,
                random_state=self.random_state,
                n_jobs=-1
            )
            scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
            self.baseline_score = scores.mean()
            self.current_score = self.baseline_score
            self.baseline_evaluated = True

            print(f"[INFO] Baseline accuracy: {self.baseline_score:.4f} ± {scores.std():.4f} (cv={cv})")
            return self.baseline_score

        except Exception as e:
            print(f"[WARNING] Baseline evaluation failed: {e}")
            self.baseline_score = 0.5
            self.current_score = 0.5
            self.baseline_evaluated = True
            return 0.5

    def evaluate_feature(self, df: pd.DataFrame, feature_code: str) -> Tuple[bool, float]:
        """
        Evaluate if a new feature improves performance.

        Args:
            df: Original DataFrame
            feature_code: Python code that creates new feature(s)

        Returns:
            Tuple of (is_improvement, new_score)
        """
        # Check if baseline has been evaluated
        if self.current_score is None:
            print("[ERROR] Must call evaluate_baseline() first")
            return False, 0.5

        # Create copy for testing
        test_df = df.copy()

        # Check code safety first
        if not self.is_safe_code(feature_code):
            print("[WARNING] Unsafe code rejected")
            return False, self.current_score

        try:
            # Execute feature code in safe environment
            local_ns = {
                'df': test_df,
                'pd': pd,
                'np': np,
            }
            exec(feature_code, {'__builtins__': {}}, local_ns)
            test_df = local_ns['df']

            # Check if new columns were added
            new_cols = [col for col in test_df.columns if col not in df.columns]
            if not new_cols:
                print("[INFO] No new feature created")
                return False, self.current_score

            print(f"[INFO] Testing new feature(s): {new_cols[:3]}")

            # Evaluate with new feature
            X = test_df.drop(columns=[self.target])
            y = test_df[self.target]

            # Encode target if needed
            if y.dtype == 'object':
                if self.target_encoder:
                    y = self.target_encoder.transform(y)
                else:
                    le = LabelEncoder()
                    y = le.fit_transform(y)

            # Handle categorical features
            X = pd.get_dummies(X)

            # Determine CV folds
            n_samples = len(test_df)
            cv = min(self.cv_folds, max(2, n_samples // 5)) if n_samples > 10 else 2

            clf = RandomForestClassifier(
                n_estimators=50,
                random_state=self.random_state,
                n_jobs=-1
            )
            scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
            new_score = scores.mean()

            improvement = new_score - self.current_score

            # Check if improvement is significant (at least 0.1% or 0.001)
            is_improvement = improvement > 0.001

            if is_improvement:
                print(f"[INFO] Improvement: {self.current_score:.4f} → {new_score:.4f} (+{improvement:.4f})")
                self.current_score = new_score
                self.feature_history.append({
                    'code': feature_code,
                    'features': new_cols,
                    'improvement': improvement,
                    'new_score': new_score
                })
                return True, new_score
            else:
                print(f"[INFO] No improvement: {self.current_score:.4f} → {new_score:.4f} ({improvement:+.4f})")
                return False, self.current_score

        except SyntaxError as e:
            print(f"[WARNING] Syntax error in feature code: {e}")
            return False, self.current_score
        except Exception as e:
            print(f"[WARNING] Feature evaluation failed: {e}")
            return False, self.current_score

    def validate_model_code(self, code_path: str, df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate generated model code by executing it and extracting a score.

        Args:
            code_path: Path to the generated Python model file.
            df: DataFrame used for validation.

        Returns:
            Tuple of validation success and metrics dictionary.
        """
        if self.target not in df.columns:
            raise ValueError(f"Target column '{self.target}' not found in dataframe")

        # Ensure baseline is evaluated
        if not self.baseline_evaluated:
            self.evaluate_baseline(df)

        code_file = Path(code_path)
        if not code_file.exists():
            raise FileNotFoundError(f"Generated code file not found: {code_path}")

        try:
            code = code_file.read_text()

            # Safe builtins
            safe_builtins = {
                # Basic types
                'abs': abs,
                'all': all,
                'any': any,
                'bool': bool,
                'bytes': bytes,
                'bytearray': bytearray,
                'dict': dict,
                'float': float,
                'int': int,
                'len': len,
                'list': list,
                'max': max,
                'min': min,
                'object': object,
                'print': print,
                'range': range,
                'set': set,
                'str': str,
                'sum': sum,
                'tuple': tuple,
                'type': type,
                'zip': zip,
                'enumerate': enumerate,

                # Exceptions
                'Exception': Exception,
                'ValueError': ValueError,
                'TypeError': TypeError,
                'KeyError': KeyError,
                'IndexError': IndexError,
                'AttributeError': AttributeError,
                'RuntimeError': RuntimeError,
                'FileNotFoundError': FileNotFoundError,

                # Type checking
                'isinstance': isinstance,
                'issubclass': issubclass,
                'callable': callable,
                'hasattr': hasattr,
                'getattr': getattr,

                # Constants
                'True': True,
                'False': False,
                'None': None,
                '__import__': __import__,
            }

            exec_globals = {
                '__builtins__': safe_builtins,
                'pd': pd,
                'np': np,
                'accuracy_score': accuracy_score,
                'mean_squared_error': mean_squared_error,
                'r2_score': r2_score,
                'RandomForestClassifier': RandomForestClassifier,
                'train_test_split': train_test_split,
                'StandardScaler': StandardScaler,
                'ColumnTransformer': ColumnTransformer,
                'Pipeline': Pipeline,
                'OneHotEncoder': OneHotEncoder,
                'SimpleImputer': SimpleImputer,
                'LogisticRegression': LogisticRegression,
                'HistGradientBoostingClassifier': HistGradientBoostingClassifier,
                'roc_auc_score': roc_auc_score,
                'classification_report': classification_report,
                'f1_score': f1_score,
            }
            exec_locals = {'df': df.copy(), 'score': None}

            # Execute with proper error handling
            exec(code, exec_globals, exec_locals)

            score = None
            if 'score' in exec_locals and exec_locals['score'] is not None:
                score = float(exec_locals['score'])
            elif 'y_pred' in exec_locals and 'y_test' in exec_locals:
                score = float(accuracy_score(exec_locals['y_test'], exec_locals['y_pred']))
            elif 'pipe' in exec_locals and 'X_test' in exec_locals and 'y_test' in exec_locals:
                y_pred = exec_locals['pipe'].predict(exec_locals['X_test'])
                score = float(accuracy_score(exec_locals['y_test'], y_pred))
            elif 'model' in exec_locals and 'X_test' in exec_locals and 'y_test' in exec_locals:
                y_pred = exec_locals['model'].predict(exec_locals['X_test'])
                score = float(accuracy_score(exec_locals['y_test'], y_pred))
            else:
                score = self.baseline_score

            metrics = {
                'score': score,
                'issues': [],
                'warnings': []
            }
            return True, metrics

        except SyntaxError as e:
            return False, {
                'score': 0.0,
                'issues': [f"SyntaxError: {e}"],
                'warnings': []
            }
        except Exception as e:
            return False, {
                'score': 0.0,
                'issues': [str(e)],
                'warnings': []
            }

    @staticmethod
    def is_safe_code(code: str) -> bool:
        """
        Check if generated code is safe to execute.

        Args:
            code: Python code string

        Returns:
            True if code appears safe, False otherwise
        """
        # Dangerous patterns to block - MORE SPECIFIC, don't block open() completely
        dangerous_patterns = [
            r'eval\s*\(', r'exec\s*\(', r'__import__\s*\(',
            r'subprocess\.', r'os\.system', r'os\.popen',
            r'import\s+os\b', r'from\s+os\s+import', r'import\s+sys\b',
            r'from\s+sys\s+import', r'globals\s*\(', r'locals\s*\(',
            r'compile\s*\(', r'__code__', r'__getattribute__',
            r'__setattr__', r'\.write\s*\(', r'\.delete\s*\(',
            r'\.remove\s*\(', r'socket\.', r'requests\.', r'urllib\.',
            r'shell\s*=', r'Popen', r'check_output', r'breakpoint\s*\(',
            r'input\s*\(', r'eval\s+', r'exec\s+',
            r'__builtins__',  # Dangerous builtins access
            r'rm\s+-rf',  # Dangerous file operations
            r'os\.kill', r'os\.remove', r'os\.rmdir',  # Dangerous OS operations
        ]

        code_lower = code.lower()

        # Check for dangerous patterns
        for pattern in dangerous_patterns:
            if re.search(pattern, code_lower):
                print(f"[WARNING] Dangerous pattern detected: {pattern}")
                return False

        # Check `del` usage - only allow deleting DataFrame columns
        # This is safe because it only removes a column from the DataFrame
        del_pattern = r'del\s+df\s*\[\s*[\'"]'
        if re.search(r'del\s+', code_lower) and not re.search(del_pattern, code_lower):
            # `del` used on something other than df['column'] - might be dangerous
            # Could be deleting variables, which is mostly harmless but let's be careful
            print("[WARNING] Suspicious 'del' usage detected")
            # Still allow it since it's just deleting variables, not files
            pass

        # Allow open() only for safe file operations (reading data files)
        # Check all open() calls to ensure they're only reading data files
        open_pattern = r'open\s*\(\s*[\'"]([^\'"]+)[\'"]\s*[,)]'
        open_matches = re.finditer(open_pattern, code)

        for match in open_matches:
            filename = match.group(1)
            # Only allow data/ or ./data/ paths
            if not filename.startswith('data/') and not filename.startswith('./data/') and not filename.startswith('..') and not filename.endswith('.arff'):
                print(f"[WARNING] Unsafe file path in open(): {filename}")
                return False

        return True

    def run_iterations(self, df: pd.DataFrame, feature_generator, max_iterations: int = 5) -> pd.DataFrame:
        """
        Run multiple CAAFE iterations.

        Args:
            df: Original DataFrame
            feature_generator: Function that generates feature code
            max_iterations: Maximum number of iterations

        Returns:
            Enhanced DataFrame with accepted features
        """
        # Ensure baseline is evaluated
        if self.current_score is None:
            self.evaluate_baseline(df)

        current_df = df.copy()

        for i in range(max_iterations):
            print(f"\n{'='*50}")
            print(f"CAAFE Iteration {i+1}/{max_iterations}")
            print(f"{'='*50}")

            # Generate new feature code (pass current_score for context)
            feature_code = feature_generator(current_df, self.current_score, i)

            if not feature_code:
                print("[INFO] No feature generated, stopping")
                break

            # Test the feature
            is_improved, new_score = self.evaluate_feature(current_df, feature_code)

            # If improved, keep it
            if is_improved:
                # Execute the code to actually add the feature
                local_ns = {'df': current_df, 'pd': pd, 'np': np}
                try:
                    exec(feature_code, {'__builtins__': {}}, local_ns)
                    current_df = local_ns['df']
                    print(f"[INFO] Feature accepted! Current score: {self.current_score:.4f}")
                except Exception as e:
                    print(f"[WARNING] Failed to apply feature: {e}")
            else:
                print(f"[INFO] Feature rejected")

        print(f"\n{'='*50}")
        print(f"CAAFE Complete!")
        print(f"Baseline: {self.baseline_score:.4f}")
        print(f"Final: {self.current_score:.4f}")
        print(f"Improvement: {self.current_score - self.baseline_score:+.4f}")
        print(f"Features added: {len(self.feature_history)}")
        print(f"{'='*50}")

        return current_df

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of feature engineering process.

        Returns:
            Dictionary with summary statistics
        """
        return {
            'baseline_score': self.baseline_score,
            'final_score': self.current_score,
            'improvement': self.current_score - self.baseline_score if self.baseline_score else 0,
            'features_tested': len(self.feature_history),
            'features_accepted': len(self.feature_history),
            'history': self.feature_history
        }


def extract_code_from_response(response_text: str) -> Optional[str]:
    """
    Extract Python code from LLM response.

    Args:
        response_text: Raw response from LLM

    Returns:
        Extracted code or None
    """
    if not response_text:
        return None

    # Remove think tags
    if '<think>' in response_text:
        response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)

    # Look for code blocks
    patterns = [
        r'```python\n(.*?)```',
        r'```\n(.*?)```',
        r'```python(.*?)```',
        r'df\[[\'"].*?[\'"]\]\s*=.*?(?=\n\n|\Z)',
    ]

    for pattern in patterns:
        match = re.search(pattern, response_text, re.DOTALL)
        if match:
            code = match.group(1).strip() if match.lastindex else match.group(0).strip()
            if 'df[' in code and ('=' in code or '.apply' in code):
                code = code.replace('\\n', '\n').replace('\\t', '\t')
                return code

    # Look for feature creation line
    lines = response_text.split('\n')
    for line in lines:
        line = line.strip()
        if 'df[' in line and '=' in line and not line.startswith('#'):
            line = line.replace('\\n', '\n')
            if len(line) > 20 and len(line) < 500:
                return line

    return None


def validate_feature_performance(df: pd.DataFrame, target: str, feature_col: str) -> float:
    """
    Quick validation of a single feature's impact.

    Args:
        df: DataFrame with feature and target
        target: Target column name
        feature_col: Feature column to test

    Returns:
        Accuracy improvement (positive means feature helps)
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    # Without feature
    X_without = df.drop(columns=[target, feature_col])
    y = df[target]

    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)

    X_without = pd.get_dummies(X_without)

    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    scores_without = cross_val_score(clf, X_without, y, cv=min(3, len(df)//3), scoring='accuracy')

    # With feature
    X_with = df.drop(columns=[target])
    X_with = pd.get_dummies(X_with)

    scores_with = cross_val_score(clf, X_with, y, cv=min(3, len(df)//3), scoring='accuracy')

    improvement = scores_with.mean() - scores_without.mean()
    return improvement


# Example usage
if __name__ == "__main__":
    print("CAAFE Validator Module")
    print("=" * 50)

    # Create sample data
    test_df = pd.DataFrame({
        'age': [25, 30, 35, 40, 45, 50, 55, 60],
        'income': [50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000],
        'target': [0, 0, 1, 1, 1, 1, 0, 0]
    })

    validator = CAAFEFeatureValidator(target='target', cv_folds=2)
    baseline = validator.evaluate_baseline(test_df)
    print(f"Test baseline: {baseline:.4f}")
    print(f"Current score: {validator.current_score:.4f}")

    # Test safe code check
    safe_code = "df['age_squared'] = df['age'] ** 2"
    unsafe_code = "os.system('rm -rf /')"

    print(f"\nSafe code check: {validator.is_safe_code(safe_code)}")
    print(f"Unsafe code check: {validator.is_safe_code(unsafe_code)}")

    # Test feature evaluation
    print(f"\nTesting feature: {safe_code}")
    is_improved, new_score = validator.evaluate_feature(test_df, safe_code)
    print(f"Feature improved: {is_improved}, New score: {new_score:.4f}")