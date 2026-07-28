"""Code validation agent module."""

import ast
import json
from pathlib import Path
from typing import List, Optional

import sklearn
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

import scipy
import pandas as pd
import numpy as np
from base_agent import Agent
from caafeValidator import CAAFEFeatureValidator


class CodeValidationAgent(Agent):
    """Code validation agent for generated model code."""

    def _save_validation_response(
        self,
        result: dict,
        code_path: str,
        data_path: str,
        target: Optional[str] = None,
        problem_type: Optional[str] = None,
    ) -> None:
        save_response = getattr(self.llm, "_save_response", None)
        if not callable(save_response) or not getattr(self.llm, "save_responses", False):
            return

        payload = {
            "code_path": code_path,
            "data_path": data_path,
            "target": target,
            "problem_type": problem_type,
            "result": result,
        }
        prompt = (
            f"Validate code file: {code_path}\n"
            f"Data file: {data_path}\n"
            f"Target: {target or 'None'}\n"
            f"Problem type: {problem_type or 'None'}"
        )
        response_payload = {
            "text": json.dumps(payload, indent=2, ensure_ascii=False),
            "raw": payload,
        }
        save_response(
            prompt,
            response_payload,
            getattr(self.llm, "default_model", None),
            4000,
            "code_validator",
        )

    def validate_code(
        self,
        code_path: str,
        data_path: str,
        target: Optional[str] = None,
        problem_type: Optional[str] = None
    ) -> dict:
        result = {
            "code_path": code_path,
            "syntax_ok": False,
            "executed": False,
            "score": None,
            "baseline_score": None,
            "improvement": None,
            "improved": None,
            "issues": [],
            "warnings": [],
            "notes": []
        }

        code_file = Path(code_path)
        if not code_file.exists():
            result["issues"].append("Code file not found")
            self._save_validation_response(result, code_path, data_path, target, problem_type)
            return result

        try:
            code = code_file.read_text()
        except Exception as e:
            result["issues"].append(f"Unable to read code file: {e}")
            self._save_validation_response(result, code_path, data_path, target, problem_type)
            return result

        if not code.strip():
            result["issues"].append("Code file is empty")
            self._save_validation_response(result, code_path, data_path, target, problem_type)
            return result

        if not CAAFEFeatureValidator.is_safe_code(code):
            result["issues"].append("Code failed safety checks")
            self._save_validation_response(result, code_path, data_path, target, problem_type)
            return result

        try:
            ast.parse(code)
            result["syntax_ok"] = True
        except SyntaxError as e:
            result["issues"].append(f"SyntaxError: {e}")
            self._save_validation_response(result, code_path, data_path, target, problem_type)
            return result

        try:
            df = self._load_data(data_path)
        except Exception as e:
            result["issues"].append(f"Data loading failed: {e}")
            self._save_validation_response(result, code_path, data_path, target, problem_type)
            return result

        if target:
            try:
                validator = CAAFEFeatureValidator(target=target)
                result["baseline_score"] = validator.evaluate_baseline(df.copy())
            except Exception as e:
                result["warnings"].append(f"Baseline evaluation skipped: {e}")

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
            'Exception': Exception,
            'ValueError': ValueError,
            'TypeError': TypeError,
            'KeyError': KeyError,
            'IndexError': IndexError,
            'AttributeError': AttributeError,
            'RuntimeError': RuntimeError,

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

            # Iterables
            'zip': zip,
            'enumerate': enumerate,
            'map': map,
            'filter': filter,
            'sorted': sorted,
            'reversed': reversed,

            # Constants
            'True': True,
            'False': False,
            'None': None,
            '__import__': __import__,

            # Useful functions
            'property': property,
            'staticmethod': staticmethod,
            'classmethod': classmethod,
            'super': super,
            'round': round,
            'pow': pow,
            'divmod': divmod,
            'format': format,
            'repr': repr,
            'ascii': ascii,
            'ord': ord,
            'chr': chr,
            'hex': hex,
            'oct': oct,
            'bin': bin,
        }

        exec_globals = {
            '__builtins__': safe_builtins,
            'pd': pd,
            'np': np,
            'sklearn': sklearn,
            'scipy': scipy,
            'accuracy_score': accuracy_score,
            'mean_squared_error': mean_squared_error,
            'r2_score': r2_score,
            'RandomForestClassifier': RandomForestClassifier,
            'HistGradientBoostingClassifier': HistGradientBoostingClassifier,
            'LogisticRegression': LogisticRegression,
            'train_test_split': train_test_split,
            'StandardScaler': StandardScaler,
            'OneHotEncoder': OneHotEncoder,
            'ColumnTransformer': ColumnTransformer,
            'Pipeline': Pipeline,
            'SimpleImputer': SimpleImputer,
        }
        exec_locals = {
            'df': df.copy()
        }

        try:
            exec(code, exec_globals, exec_locals)
            result["executed"] = True
        except Exception as e:
            result["issues"].append(f"Execution error: {e}")
            self._save_validation_response(result, code_path, data_path, target, problem_type)

        metrics = self._extract_result_metrics(exec_locals, target, problem_type)
        result.update(metrics)

        if result.get("score") is not None and result.get("baseline_score") is not None:
            result["improvement"] = result["score"] - result["baseline_score"]
            result["improved"] = result["improvement"] > 0

        self._save_validation_response(result, code_path, data_path, target, problem_type)
        return result

    @staticmethod
    def _load_data(path: str) -> pd.DataFrame:
        if path.endswith('.arff'):
            try:
                from scipy.io import arff
            except ImportError:
                raise ImportError("scipy is required for ARFF support")
            data, meta = arff.loadarff(path)
            df = pd.DataFrame(data)
            for col in df.select_dtypes(include=['object']).columns:
                try:
                    if len(df[col]) > 0 and isinstance(df[col].iloc[0], bytes):
                        df[col] = df[col].str.decode('utf-8')
                except (IndexError, AttributeError, UnicodeDecodeError):
                    pass
            return df

        return pd.read_csv(path)

    @staticmethod
    def _extract_result_metrics(exec_locals: dict, target: Optional[str], problem_type: Optional[str]) -> dict:
        score = None
        warnings: List[str] = []
        issues: List[str] = []

        if 'score' in exec_locals:
            try:
                score = float(exec_locals['score'])
            except Exception as e:
                warnings.append(f"Unable to parse score: {e}")

        elif 'y_pred' in exec_locals and 'y_test' in exec_locals:
            try:
                from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
                y_test = exec_locals['y_test']
                y_pred = exec_locals['y_pred']
                if problem_type == 'regression':
                    score = r2_score(y_test, y_pred)
                else:
                    score = accuracy_score(y_test, y_pred)
            except Exception as e:
                issues.append(f"Unable to evaluate predictions: {e}")

        elif 'pipe' in exec_locals and 'X_test' in exec_locals and 'y_test' in exec_locals:
            try:
                from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
                y_test = exec_locals['y_test']
                y_pred = exec_locals['pipe'].predict(exec_locals['X_test'])
                if problem_type == 'regression':
                    score = r2_score(y_test, y_pred)
                else:
                    score = accuracy_score(y_test, y_pred)
            except Exception as e:
                issues.append(f"Unable to evaluate pipeline: {e}")

        elif 'model' in exec_locals and 'X_test' in exec_locals and 'y_test' in exec_locals:
            try:
                from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
                y_test = exec_locals['y_test']
                y_pred = exec_locals['model'].predict(exec_locals['X_test'])
                if problem_type == 'regression':
                    score = r2_score(y_test, y_pred)
                else:
                    score = accuracy_score(y_test, y_pred)
            except Exception as e:
                issues.append(f"Unable to evaluate model predictions: {e}")

        return {
            'score': score,
            'warnings': warnings,
            'issues': issues
        }
