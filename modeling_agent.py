"""Modeling agent module."""

import json
import re
from pathlib import Path
from typing import Any, Optional

from base_agent import Agent


class ModelingAgent(Agent):
    """Modeling agent with improved code extraction and cutoff handling."""

    def generate(self, path: str, feature_info: Optional[dict] = None,
             problem_type: Optional[str] = None, target: Optional[str] = None,
             model: Optional[str] = None) -> dict:
        content, mt, text, structure = self._read_file(path)

        headers = structure.get('headers', [])
        rows = structure.get('rows', 0)

        file_ext = Path(path).suffix.lower()
        is_arff = file_ext == '.arff'
        file_format = "ARFF" if is_arff else "CSV"

        feature_context = ""
        feature_details = ""
        feature_reasons = ""
        feature_notes = ""
        derived_features = []
        feature_metadata = {}
        if feature_info:
            features = feature_info.get('features', [])[:8]
            scale = feature_info.get('scale', [])[:4]
            encode = feature_info.get('encode', {})
            drop = feature_info.get('drop', [])[:3]
            feature_details = feature_info.get('feature_details', "") or ""
            feature_reasons = feature_info.get('feature_reasons', "") or ""
            feature_notes = feature_info.get('feature_notes', "") or ""
            derived_features = feature_info.get('derived_features', [])[:4]
            feature_metadata = feature_info.get('feature_metadata', {}) or {}
            feature_context = f"\nFEATURES: {features}\nSCALE: {scale}\nENCODE: {encode}\nDROP: {drop}"
            if feature_details:
                feature_context += f"\nFEATURE_DETAILS: {feature_details[:800]}"
            if feature_reasons:
                feature_context += f"\nFEATURE_REASONS: {feature_reasons[:800]}"
            if feature_notes:
                feature_context += f"\nFEATURE_NOTES: {feature_notes[:800]}"
            if derived_features:
                feature_context += f"\nDERIVED_FEATURES: {derived_features}"
            if feature_metadata:
                feature_context += f"\nFEATURE_METADATA: {feature_metadata}"

        sample_lines = text.splitlines()[:3]
        sample = "\n".join(sample_lines) if len(sample_lines) > 1 else ""

        prompt = f"""{Path(path).name} | {rows} rows | {len(headers)} cols | FORMAT: {file_format}
            Target: {target or 'auto'}{feature_context}
            From feature_context grab the recomended features, use it to transform the dataset before modeling.
            Use the feature engineering details to choose appropriate preprocessing and models.
            Prefer simple, robust models for datasets and use actual column names.
            CRITICAL: File format is {file_format}. Use CORRECT loader in code!
            For ARFF: from scipy.io import arff; data, meta = arff.loadarff('data/{Path(path).name}')
            For CSV: df = pd.read_csv('data/{Path(path).name}')
            IMPORTANT: Use modern scikit-learn API - avoid deprecated parameters like multi_class.
            SAMPLE:
            {sample[:300]}

            RETURN ONLY VALID JSON - NO OTHER TEXT.
            Use actual column names from the dataset and provided feature engineering guidance.
            Do not include markdown, code fences, analysis, or explanation.
            Output exactly one JSON object with keys: problem, target, models, code.
            If you cannot produce valid JSON, return {{"error":"json"}} only.
            Example output:
            {{"problem":"classification","target":"Churn","models":["RandomForestClassifier"],"code":"import pandas as pd\n..."}}

            YOUR JSON:"""

        resp = self.llm.generate(prompt, model=model, max_tokens=6000, agent="Modeling")
        response_text = resp.get("text", "")

        result = self._extract_json(response_text)

        if isinstance(result, list):
            result = {
                "problem": problem_type or "classification",
                "target": target or "target",
                "models": result,
                "code": ""
            }

        if result is None or not isinstance(result, dict):
            print("[INFO] JSON extraction failed, extracting code directly")
            result = self._extract_code_from_response(response_text)
        elif not result or not isinstance(result, dict):
            result = self._extract_code_from_response(response_text)

        if not result:
            result = self._find_json_in_response(response_text)

        code = None
        if result and isinstance(result, dict):
            code = result.get("code")

        if not code or (isinstance(code, str) and len(code) < 50):
            code = self._extract_code_block(response_text)

        if code and isinstance(code, str):
            code = self._clean_code(code)
            code = self._fix_data_loader(code, path, is_arff)

        code_path = None
        accept = False
        if code and isinstance(code, str):
            code_len = len(code)
            has_import = 'import ' in code
            has_def = 'def ' in code or 'class ' in code
            has_sklearn = 'sklearn' in code or 'from sklearn' in code
            if code_len > 100 or (code_len > 40 and (has_import or has_def or has_sklearn)):
                accept = True

        if accept and isinstance(code, str):
            code = code.replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"')
            code_filename = f"{Path(path).stem}_model.py"
            code_path = Path.cwd() / "generated_code" / code_filename
            code_path.parent.mkdir(exist_ok=True)
            code_path.write_text(code)
            print(f"[INFO] ✓ Code saved to {code_path}")
            print(f"[INFO] Code lines: {len(code.split(chr(10)))}")
        else:
            code_length = len(code) if code and isinstance(code, str) else 0
            reasons = []
            if code_length < 40:
                reasons.append(f"too short ({code_length} chars)")
            if code and isinstance(code, str) and not ('import ' in code or 'def ' in code or 'class ' in code):
                reasons.append("missing imports/defs")
            if not code:
                reasons.append("no code found")
            print(f"[WARNING] No valid code generated ({'; '.join(reasons)})")
            if code and isinstance(code, str):
                preview_path = Path.cwd() / "generated_code" / f"{Path(path).stem}_model_preview.txt"
                preview_path.parent.mkdir(exist_ok=True)
                preview_path.write_text(code[:2000])
                print(f"[INFO] Code preview saved to {preview_path}")

        return {
            "problem_type": result.get("problem") if result else None,
            "target": result.get("target") if result else None,
            "recommended_models": result.get("models", ["RandomForest"]) if result else ["RandomForest"],
            "code_generated": bool(code_path),
            "code_path": str(code_path) if code_path else None,
            "code_preview": (code[:400] + "...") if code and isinstance(code, str) and len(code) > 400 else code
        }

    def _extract_code_from_response(self, text: str) -> dict:
        code_match = re.search(r'```python\n(.*?)```', text, re.DOTALL)
        if not code_match:
            code_match = re.search(r'```\n(.*?)```', text, re.DOTALL)
        if not code_match:
            code_match = re.search(r'(import pandas.*?)(?=\n\n|\Z)', text, re.DOTALL)

        if code_match:
            code = code_match.group(1).strip()
            return {
                "problem": "classification",
                "target": "target",
                "models": ["RandomForest"],
                "code": code
            }
        return {}

    def _find_json_in_response(self, text: str) -> dict:
        json_pattern = r'\{[^{}]*"problem"[^{}]*"code"[^{}]*\}'
        matches = re.findall(json_pattern, text, re.DOTALL)

        for match in matches:
            try:
                return json.loads(match)
            except:
                continue

        return {}

    def _extract_code_block(self, text: str) -> Optional[str]:
        if not text:
            return None

        patterns = [
            r'```python\n(.*?)```',
            r'```\n(.*?)```',
            r'```python(.*?)```',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                code = match.group(1).strip()
                if code and ('import' in code or 'def ' in code):
                    return code

        match = re.search(r'(import pandas.*?)(?=\n\n[^#\s]|\Z)', text, re.DOTALL)
        if match:
            code = match.group(1).strip()
            if code:
                return code

        return ""

    def regenerate_with_feedback(self, path, feature_info, error, previous_code_path=None, problem_type=None, target=None, model=None):
        """
        Regenerate code with error feedback for fixing.
        """
        print(f"\n[Modeling Agent] Regenerating code with error feedback...")
        print(f"  Error: {error.get('error_summary', 'No details')[:200]}")

        # Read previous code if available
        previous_code = ""
        if previous_code_path and Path(previous_code_path).exists():
            try:
                previous_code = Path(previous_code_path).read_text()
                print(f"  Previous code loaded ({len(previous_code)} chars)")
            except Exception as e:
                print(f"  Failed to read previous code: {e}")

        # Read data for structure
        _, _, _, structure = self._read_file(path)
        headers = structure.get('headers', [])
        rows = structure.get('rows', 0)

        # Build error details
        error_details = error.get('issues', [error.get('error_summary', 'Unknown error')])
        error_preview = "\n".join(str(e) for e in error_details[:3])

        prompt = f"""Fix the previous code errors.
            Error: {error_preview}
            Data: {Path(path).name} | {rows} rows
            Problem Type: {problem_type or 'classification'}
            Target: {target or 'auto'}

            Previous code:
            ```python
            {previous_code[:1500] if previous_code else "No previous code available"}

            Return JSON with: problem, target, models, code (fixed version)."""

        resp = self.llm.generate(prompt, model=model, max_tokens=6000, agent="Modeling")
        response_text = resp.get("text", "")

        result = self._extract_json(response_text)
        if result is None or not isinstance(result, dict):
            result = self._extract_code_from_response(response_text)

        code = None
        if result and isinstance(result, dict):
            code = result.get("code")

        if not code or (isinstance(code, str) and len(code) < 50):
            code = self._extract_code_block(response_text)

        if code and isinstance(code, str):
            code = self._clean_code(code)
            file_ext = Path(path).suffix.lower()
            code = self._fix_data_loader(code, path, file_ext == '.arff')

        code_path = None
        if code and isinstance(code, str) and len(code) > 100:
            code = code.replace('\n', '\n').replace('\t', '\t').replace('\"', '"')
            code_filename = f"{Path(path).stem}_model_fixed.py"
            code_path = Path.cwd() / "generated_code" / code_filename
            code_path.parent.mkdir(exist_ok=True)
            code_path.write_text(code)
            print(f"[INFO] Fixed code saved to {code_path}")

        return {
            "problem_type": result.get("problem") if result else problem_type,
            "target": result.get("target") if result else target,
            "recommended_models": result.get("models", ["RandomForest"]) if result else ["RandomForest"],
            "code_generated": bool(code_path),
            "code_path": str(code_path) if code_path else None,
            "code_preview": (code[:400] + "...") if code and isinstance(code, str) and len(code) > 400 else code,
            "fixed": bool(code_path)
            }

    def _fix_data_loader(self, code: str, path: str, is_arff: bool) -> str:
        filename = Path(path).name

        if is_arff:
            csv_pattern = r"df\s*=\s*pd\.read_csv\(['\"]([^'\"]*)['\"][^)]*\)"
            arff_code = f"""# Load ARFF file
            from scipy.io import arff
            data, meta = arff.loadarff('data/{filename}')
            df = pd.DataFrame(data)
            # Decode bytes to strings for ARFF data
            for col in df.select_dtypes(include=['object']).columns:
                try:
                    if len(df[col]) > 0 and isinstance(df[col].iloc[0], bytes):
                        df[col] = df[col].str.decode('utf-8')
                except (IndexError, AttributeError, UnicodeDecodeError):
                    pass
                """
            code = re.sub(csv_pattern, arff_code, code)

            if 'arff.loadarff' not in code and 'read_csv' not in code:
                code = arff_code + "\n\n" + code
        else:
            csv_pattern = r"df\s*=\s*pd\.read_csv\(['\"]([^'\"]*)['\"][^)]*\)"
            if re.search(csv_pattern, code):
                code = re.sub(csv_pattern, f"df = pd.read_csv('data/{filename}')", code)

        return code

    def _clean_code(self, code: str) -> str:
        if not code or not isinstance(code, str):
            return ""

        match = re.search(r'(import|from|def|class|#).*', code, re.DOTALL)
        if match:
            code = match.group(0)

        lines = code.split('\n')
        last_line = lines[-1].strip() if lines else ""

        if last_line.count('(') > last_line.count(')'):
            code += ')'
        if last_line.count('[') > last_line.count(']'):
            code += ']'
        if last_line.count('{') > last_line.count('}'):
            code += '}'

        if code.strip().endswith('X[scale_cols'):
            code += '] = scaler.fit_transform(X[scale_cols])'
        if code.strip().endswith('model = RandomForestClassifier('):
            code += ')'

        code = re.sub(r',?\s*multi_class=[\'\"]?auto[\'\"]?', '', code)
        code = re.sub(r',?\s*multi_class=\'auto\'', '', code)
        code = re.sub(r',?\s*multi_class="auto"', '', code)

        return code
