"""
Agent Module for LLM-Powered Data Science Pipeline
"""

import ast
import csv
import io
import json
import mimetypes
import re
import pandas as pd

from typing import Any, Dict, List, Optional
from pathlib import Path
from llms import LLMManager
from openMLAgentLocal import OpenMLAgent
from caafeValidator import CAAFEFeatureValidator

class Agent:
    """Base agent with common utilities."""

    def __init__(self, llm_manager: LLMManager):
        self.llm = llm_manager

    @staticmethod
    def _read_file(path: str) -> tuple:
        """Read file and return content, mime type, and structure."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {path}")

        content = p.read_bytes()
        mt = Agent._detect_mimetype(p.name, content)
        text = Agent._prepare_text(content, mt)
        structure = Agent._get_structure(text, mt, p.name)

        # For ARFF files, read the full structure with metadata
        if p.suffix.lower() == '.arff' and structure.get('needs_full_read'):
            structure = Agent._read_arff_structure(str(p))
            structure["format"] = "arff"

        return content, mt, text, structure

    @staticmethod
    def _detect_mimetype(filename: str, content: bytes) -> str:
        mt, _ = mimetypes.guess_type(filename)
        if not mt:
            if content.startswith(b"%PDF"):
                return "application/pdf"
            if content.lstrip().startswith(b"{"):
                return "application/json"
            return "application/octet-stream"
        return mt

    @staticmethod
    def _prepare_text(content: bytes, mimetype: str) -> str:
        if mimetype.startswith("text/") or mimetype in ("application/json", "application/csv"):
            try:
                return content.decode("utf-8")
            except:
                return content.decode("latin-1", errors="ignore")
        return content[:1024].hex()

    @staticmethod
    def _read_arff_structure(file_path: str) -> Dict[str, Any]:
        """Extract structure from ARFF file header."""
        try:
            from scipy.io import arff
            data, meta = arff.loadarff(file_path)
            df = pd.DataFrame(data)

            # Decode bytes columns
            for col in df.select_dtypes(include=['object']).columns:
                try:
                    if df[col].iloc[0] and isinstance(df[col].iloc[0], bytes):
                        df[col] = df[col].str.decode('utf-8')
                except (IndexError, AttributeError, UnicodeDecodeError):
                    pass

            info = {"rows": len(df), "columns": len(df.columns), "headers": list(df.columns)}

            # Detect data types
            types: Dict[str, str] = {}
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    types[col] = "numeric" if df[col].dtype == 'int64' else "float"
                else:
                    types[col] = "categorical"
            if types:
                info["data_types"] = types

            return info
        except ImportError:
            return {"rows": 0, "columns": 0, "warning": "scipy not installed for ARFF support"}
        except Exception as e:
            return {"rows": 0, "columns": 0, "error": str(e)}

    @staticmethod
    def _get_structure(content: str, mimetype: str, filename: str = "") -> Dict[str, Any]:
        """Extract basic structure from CSV/ARFF/text data."""
        # Handle ARFF files
        if filename.endswith(".arff") or mimetype == "text/x-arff":
            try:
                # For ARFF, we need the file path, not just content
                # Return a placeholder that will be enhanced in _read_file
                return {"rows": 0, "columns": 0, "format": "arff", "needs_full_read": True}
            except Exception:
                pass

        lines = [l for l in content.splitlines() if l.strip()]
        if not lines:
            return {"rows": 0, "columns": 0}

        info: Dict[str, Any] = {"rows": len(lines)}

        is_csv = mimetype == "text/csv" or filename.endswith(".csv") or ("," in lines[0] and len(lines[0].split(",")) > 1)

        if is_csv:
            try:
                reader = csv.DictReader(io.StringIO("\n".join(lines[:10])))
                fieldnames = reader.fieldnames or []
                if fieldnames:
                    info["headers"] = list(fieldnames)
                    info["columns"] = len(fieldnames)

                    types: Dict[str, str] = {}
                    for row in list(reader)[:5]:
                        for h in fieldnames:
                            val = row.get(h)
                            if not val:
                                continue
                            s = val.strip()
                            if not s:
                                continue
                            if s.isdigit():
                                types[h] = "numeric"
                            elif s.replace('.', '', 1).isdigit():
                                types[h] = "float"
                            else:
                                types[h] = types.get(h, "categorical")
                    if types:
                        info["data_types"] = types
            except Exception:
                pass

        return info

    @staticmethod
    def _extract_json(text: Optional[str]) -> Any:
        """Extract JSON from LLM response - handles DeepSeek thinking text."""
        if not text:
            return None

        # Remove think tags and everything before them
        if '<think>' in text:
            think_end = text.find('</think>')
            if think_end != -1:
                text = text[think_end + 8:]

        # Look for JSON after the thinking section
        lines = text.split('\n')
        json_lines = []
        in_json = False
        brace_count = 0

        for line in lines:
            if not in_json:
                if '{' in line or '[' in line:
                    in_json = True
                    start_idx = min(
                        line.find('{') if '{' in line else len(line),
                        line.find('[') if '[' in line else len(line)
                    )
                    line = line[start_idx:]
                    brace_count = line.count('{') + line.count('[') - line.count('}') - line.count(']')
                    json_lines.append(line)
            else:
                json_lines.append(line)
                brace_count += line.count('{') + line.count('[') - line.count('}') - line.count(']')
                if brace_count == 0:
                    break

        if json_lines:
            json_str = '\n'.join(json_lines)
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r',\s*\]', ']', json_str)
            json_str = re.sub(r'```json\s*|```\s*', '', json_str)

            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"[WARNING] JSON parse error: {e}")
                json_str = re.sub(r'([^\\])\\([^"\\/bfnrtu])', r'\1\\\\\2', json_str)
                try:
                    return json.loads(json_str)
                except:
                    pass

        return None

    @staticmethod
    def _summarize_response(text: str, max_length: int = 1000) -> str:
        if not text:
            return ""
        summary = text.replace('\n', ' ').replace('\r', ' ').strip()
        if len(summary) > max_length:
            return summary[:max_length].rstrip() + "..."
        return summary


class DomainExpertAgent(Agent):
    """Domain-specific analysis agent."""

    def analyze(self, path: str, domain: str, model: Optional[str] = None) -> dict:
        content, mt, text, structure = self._read_file(path)
        sample = "\n".join(text.splitlines()[:5])

        prompt = f"""Domain: {domain}
        Data file: {path}
        Columns: {structure.get('headers', [])}
        Sample data:
        {sample}

        Return JSON:
        {{"domain_insights":"key insights",
        "key_metrics":["metric1","metric2"],
        "data_quality_issues":["issue1"],
        "business_questions":["question1"],
        "relevant_context":"context",
        "data_limitations":["limitation1"]}}"""

        resp = self.llm.generate(prompt, model=model, max_tokens=1000)
        analysis = self._extract_json(resp.get("text")) or {}

        return {"meta": {"file": Path(path).name, "domain": domain}, "structure": structure, "analysis": analysis}


class FeatureEngineerAgent(Agent):
    """Feature engineering recommendations agent."""

    def analyze(self, path: str, target: Optional[str] = None, model: Optional[str] = None) -> dict:
        content, mt, text, structure = self._read_file(path)

        headers = structure.get('headers', [])
        data_types = structure.get('data_types', {})
        rows = structure.get('rows', 0)

        prompt = f"""DATA: {path}
            ROWS: {rows}
            COLS: {headers}
            TYPES: {data_types}
            TARGET: {target if target else 'auto'}

            You are a data scientist prformming feature-engineering.
            Your final answer must be only one valid JSON object.
            Do not include any explanation, analysis, reasoning, or markdown.
            Use actual column names from COLS and select useful features, scaling, encoding, and drops.
            If the dataset contains obvious identifier or metadata columns, drop them in DROP.

            NO MODEL GENERATION ONLY FEATURE ENGINEERING SUGGESTIONS.

            RETURN ONLY VALID JSON - NO OTHER TEXT - NO CODE.

            Use actual column names from COLS; do not use placeholders like col1, numeric_col, cat_col, or id_col.
            Output exactly one JSON object with keys: features, scale, encode, drop.
            Optional keys may include: feature_details, feature_reasons, feature_notes, derived_features, feature_metadata.
            If you have extra context about why a feature was selected, include it in feature_metadata.
            Keep metadata concise and useful.
            If you cannot produce valid JSON, return {{"error":"json"}} only.

            Example of how to derive the features:
                {{example, new_feature = feature1 * feature2}}
            Example output:
                {{"features":["age","salary"],"scale":["age"],"encode":{{"gender":"label"}},"drop":["customer_id"],"feature_details":"Scale numeric columns and label encode gender.","derived_features":["age_squared"],"feature_metadata":{{"reason":"high correlation with churn"}}}}"""

        resp = self.llm.generate(prompt, model=model, max_tokens=800, agent="Feature")
        raw_response = resp.get("text", "")
        recommendations = self._normalize_feature_recommendations(self._extract_json(raw_response), headers, data_types, target)
        feature_details = self._summarize_response(raw_response)

        # If response invalid, attempt up to two retries with clearer instructions
        attempts = 0
        while not self._valid_feature_recommendations(recommendations) and attempts < 2:
            attempts += 1
            print(f"[WARNING] Feature engineer response did not contain required JSON keys. Retry #{attempts}.")
            if isinstance(recommendations, dict) and recommendations.get("error") == "json":
                # Model explicitly signaled it cannot produce JSON
                break

            retry_prompt = f"""DATA: {path}
            ROWS: {rows}
            COLS: {headers}
            TYPES: {data_types}
            TARGET: {target if target else 'auto'}

            Use actual column names from COLS only and choose useful features, scaling, encodings, and drops.
            If a column looks like an identifier or metadata, include it in DROP.
            If you can create a derived feature, include it in response.
            Optional keys may include: feature_details, feature_reasons, feature_notes, derived_features, feature_metadata.
            RETURN ONLY VALID JSON - NO OTHER TEXT - NO CODE.
            Output exactly one JSON object with keys: features, scale, encode, drop.
            If you cannot produce valid JSON, return {{"error":"json"}} only.
            IMPORTANT: do not return partial JSON. If you must, return {{"error":"json"}} instead.
            YOUR JSON:"""

            retry_resp = self.llm.generate(retry_prompt, model=model, max_tokens=800)
            recommendations = self._normalize_feature_recommendations(
                self._extract_json(retry_resp.get("text", "")), headers, data_types, target)

            # If retry returns a fragment (like encode mapping), attempt wrapping again
            if isinstance(recommendations, dict) and not self._valid_feature_recommendations(recommendations):
                if headers and set(recommendations.keys()).issubset(set(headers)) and all(isinstance(v, str) for v in recommendations.values()):
                    wrapped = {
                        "features": [h for h in headers if h not in recommendations.keys() and h != target][:10],
                        "scale": [h for h in headers if data_types.get(h) in ['numeric', 'float']][:5],
                        "encode": recommendations,
                        "drop": [h for h in headers if 'id' in h.lower() or 'date' in h.lower()],
                        # "feature_code": "",
                        "feature_metadata": {"note": "wrapped encode-only response"}
                    }
                    if self._valid_feature_recommendations(wrapped):
                        recommendations = wrapped
                        break

            # If the response contained some keys but not the required top-level ones,
            # ask the model to complete the JSON using the partial as context.
            if isinstance(recommendations, dict) and recommendations and not self._valid_feature_recommendations(recommendations):
                completion_prompt = f"""You returned a partial JSON: {json.dumps(recommendations)}\n\nUsing the DATA: {path} with COLS: {headers} and TYPES: {data_types}, produce ONLY one valid JSON object with keys: features, scale, encode, drop.\nOptional keys may include: feature_details, feature_reasons, feature_notes, derived_features, feature_metadata.\nDo not invent columns outside COLS. Fill missing keys sensibly and use actual column names.\nIf you cannot produce valid JSON, return {{"error":"json"}} only.\nRETURN ONLY VALID JSON DO NOT RETURN CODE OR CODE SNIPPETS."""
                comp_resp = self.llm.generate(completion_prompt, model=model, max_tokens=800)
                comp_json = self._normalize_feature_recommendations(
                    self._extract_json(comp_resp.get("text", "")), headers, data_types, target)
                if self._valid_feature_recommendations(comp_json):
                    recommendations = comp_json
                    break

        if not self._valid_feature_recommendations(recommendations):
            print("[WARNING] Feature engineer JSON output invalid after retries. Using default fallback.")
            recommendations = self._default_features(headers, data_types)

        if not isinstance(recommendations, dict):
            recommendations = {}

        return {
            "features": recommendations.get("features", headers[:10]),
            "scale": recommendations.get("scale", []),
            "encode": recommendations.get("encode", {}),
            "drop": recommendations.get("drop", []),
            # "feature_code": recommendations.get("feature_code", ""),
            "feature_details": recommendations.get("feature_details", feature_details),
            "feature_reasons": recommendations.get("feature_reasons", ""),
            "feature_notes": recommendations.get("feature_notes", ""),
            "derived_features": recommendations.get("derived_features", []),
            "feature_metadata": recommendations.get("feature_metadata", {}),
            "raw_feature_response": raw_response
        }

    def _normalize_feature_recommendations(
        self,
        recommendations: Any,
        headers: List[str],
        data_types: Dict[str, str],
        target: Optional[str]
    ) -> Any:
        if isinstance(recommendations, dict):
            if self._valid_feature_recommendations(recommendations):
                return recommendations

            # heuristic: if all keys look like column names and values are strings,
            # treat as an encode mapping
            if headers and set(recommendations.keys()).issubset(set(headers)) and all(isinstance(v, str) for v in recommendations.values()):
                wrapped = {
                    "features": [h for h in headers if h not in recommendations.keys() and h != target][:10],
                    "scale": [h for h in headers if data_types.get(h) in ['numeric', 'float']][:5],
                    "encode": recommendations,
                    "drop": [h for h in headers if 'id' in h.lower() or 'date' in h.lower()],
                    # "feature_code": "",
                    "feature_metadata": {"note": "wrapped encode-only response"}
                }
                if self._valid_feature_recommendations(wrapped):
                    return wrapped

            # Heuristic: if the model returned a partial feature object, fill missing keys.
            if any(key in recommendations for key in ["features", "scale", "encode", "drop"]):
                wrapped = {
                    "features": recommendations.get("features", [h for h in headers if h != target][:10]),
                    "scale": recommendations.get("scale", [h for h in headers if data_types.get(h) in ['numeric', 'float']][:5]),
                    "encode": recommendations.get("encode", {}),
                    "drop": recommendations.get("drop", [h for h in headers if 'id' in h.lower() or 'date' in h.lower()]),
                    "feature_details": recommendations.get("feature_details", ""),
                    "feature_reasons": recommendations.get("feature_reasons", ""),
                    "feature_notes": recommendations.get("feature_notes", ""),
                    "derived_features": recommendations.get("derived_features", []),
                    "feature_metadata": recommendations.get("feature_metadata", {"note": "wrapped partial feature response"})
                }
                if self._valid_feature_recommendations(wrapped):
                    return wrapped

        if isinstance(recommendations, list) and all(isinstance(item, str) for item in recommendations):
            return {
                "features": recommendations[:10],
                "scale": [h for h in headers if data_types.get(h) in ['numeric', 'float']][:5],
                "encode": {},
                "drop": [h for h in headers if 'id' in h.lower() or 'date' in h.lower()],
                # "feature_code": "",
                "feature_metadata": {"note": "wrapped feature list response"}
            }

        return recommendations

    def _valid_feature_recommendations(self, recommendations: Any) -> bool:
        if not isinstance(recommendations, dict):
            return False

        required_keys = {"features", "scale", "encode", "drop"}
        if not required_keys.issubset(recommendations.keys()):
            return False

        if not isinstance(recommendations.get("features"), list):
            return False
        if not isinstance(recommendations.get("scale"), list):
            return False
        if not isinstance(recommendations.get("encode"), dict):
            return False
        if not isinstance(recommendations.get("drop"), list):
            return False

        return True

    def _default_features(self, headers: List[str], data_types: Dict[str, str]) -> dict:
        numeric = [h for h in headers if data_types.get(h) in ['numeric', 'float']]
        categorical = [h for h in headers if data_types.get(h) == 'categorical']

        return {
            "features": headers[:10],
            "scale": numeric[:5],
            "encode": {col: "label" for col in categorical[:3]},
            "drop": [h for h in headers if 'id' in h.lower() or 'date' in h.lower()],
            # "feature_code": "",
            "feature_details": "Fallback default features used because the model response could not be parsed into valid JSON.",
            "feature_reasons": "Fallback generated defaults",
            "feature_notes": "Default feature list because JSON extraction failed.",
            "derived_features": [],
            "feature_metadata": {"note": "default fallback"},
            "raw_feature_response": ""
        }


class ModelingAgent(Agent):
    """Modeling agent with improved code extraction and cutoff handling."""

    def generate(self, path: str, feature_info: Optional[dict] = None,
             problem_type: Optional[str] = None, target: Optional[str] = None,
             model: Optional[str] = None) -> dict:
        content, mt, text, structure = self._read_file(path)

        headers = structure.get('headers', [])
        rows = structure.get('rows', 0)

        # Detect file format
        file_ext = Path(path).suffix.lower()
        is_arff = file_ext == '.arff'
        file_format = "ARFF" if is_arff else "CSV"

        # Build feature context
        feature_context = ""
        validated_code = ""
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

        # Get sample - limited to save tokens
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

        # Use 6000 tokens to avoid timeout
        resp = self.llm.generate(prompt, model=model, max_tokens=6000, agent="Modeling")
        response_text = resp.get("text", "")

        # Extract JSON
        result = self._extract_json(response_text)

        # Normalize array responses into a modeling object
        if isinstance(result, list):
            result = {
                "problem": problem_type or "classification",
                "target": target or "target",
                "models": result,
                "code": ""
            }
        # Check if result is None or not a dict before trying to check if it's in response_text
        # Fallback: extract code block directly if JSON extraction failed or returned invalid result
        if result is None or not isinstance(result, dict):
            print("[INFO] JSON extraction failed, extracting code directly")
            result = self._extract_code_from_response(response_text)
        elif not result or not isinstance(result, dict):
            result = self._extract_code_from_response(response_text)

        if not result:
            result = self._find_json_in_response(response_text)

        # Extract code from result (handle None case)
        code = None
        if result and isinstance(result, dict):
            code = result.get("code")

        # If code is still empty, try direct extraction
        if not code or (isinstance(code, str) and len(code) < 50):
            code = self._extract_code_block(response_text)
        # Fix cutoff issues in code (only if code is not None)
        if code and isinstance(code, str):
            code = self._clean_code(code)
            # Fix data loading for ARFF vs CSV
            code = self._fix_data_loader(code, path, is_arff)

        # Save code if valid. Accept shorter code if it clearly contains
        # Python constructs (imports, defs, sklearn usage). Log reasons when
        # rejecting.
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
            # Unescape the code
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
            # Save preview for debugging
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
        """Extract code directly from response when JSON parsing fails."""

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
        """Find any JSON-like structure in the response."""

        json_pattern = r'\{[^{}]*"problem"[^{}]*"code"[^{}]*\}'
        matches = re.findall(json_pattern, text, re.DOTALL)

        for match in matches:
            try:
                return json.loads(match)
            except:
                continue

        return {}

    def _extract_code_block(self, text: str) -> Optional[str]:
        """Extract Python code block from response text."""
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

        # Look for code starting with import
        match = re.search(r'(import pandas.*?)(?=\n\n[^#\s]|\Z)', text, re.DOTALL)
        if match:
            code = match.group(1).strip()
            if code:
                return code

        # Return empty string instead of None to avoid len() errors
        return ""

    def _fix_data_loader(self, code: str, path: str, is_arff: bool) -> str:
        """Fix the data loading code to match the actual file format."""
        filename = Path(path).name

        if is_arff:
            # Replace pd.read_csv with ARFF loader
            csv_pattern = r"df\s*=\s*pd\.read_csv\(['\"]([^'\"]*)['\"][^)]*\)"
            arff_code = f"""# Load ARFF file
            from scipy.io import arff
            data, meta = arff.loadarff('{filename}')
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

            # If no data loading found, add ARFF loader at start
            if 'arff.loadarff' not in code and 'read_csv' not in code:
                code = arff_code + "\n\n" + code
        else:
            # For CSV, fix the filename to match actual file
            csv_pattern = r"df\s*=\s*pd\.read_csv\(['\"]([^'\"]*)['\"][^)]*\)"
            if re.search(csv_pattern, code):
                code = re.sub(csv_pattern, f"df = pd.read_csv('{filename}')", code)

        return code

    def _clean_code(self, code: str) -> str:
        """Clean and fix cutoff issues in code."""
        if not code or not isinstance(code, str):
            return ""

        # Remove thinking text before code
        match = re.search(r'(import|from|def|class|#).*', code, re.DOTALL)
        if match:
            code = match.group(0)

        # Fix unclosed parentheses
        lines = code.split('\n')
        last_line = lines[-1].strip() if lines else ""

        if last_line.count('(') > last_line.count(')'):
            code += ')'
        if last_line.count('[') > last_line.count(']'):
            code += ']'
        if last_line.count('{') > last_line.count('}'):
            code += '}'

        # Fix incomplete scaler line
        if code.strip().endswith('X[scale_cols'):
            code += '] = scaler.fit_transform(X[scale_cols])'

        # Fix incomplete model line
        if code.strip().endswith('model = RandomForestClassifier('):
            code += ')'

        # Remove deprecated sklearn parameters
        code = re.sub(r',?\s*multi_class=[\'"]?auto[\'"]?', '', code)
        code = re.sub(r',?\s*multi_class=\'auto\'', '', code)
        code = re.sub(r',?\s*multi_class="auto"', '', code)

        return code


class CodeValidationAgent(Agent):
    """Code validation agent for generated model code."""

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
            return result

        try:
            code = code_file.read_text()
        except Exception as e:
            result["issues"].append(f"Unable to read code file: {e}")
            return result

        if not code.strip():
            result["issues"].append("Code file is empty")
            return result

        if not CAAFEFeatureValidator.is_safe_code(code):
            result["issues"].append("Code failed safety checks")
            return result

        try:
            ast.parse(code)
            result["syntax_ok"] = True
        except SyntaxError as e:
            result["issues"].append(f"SyntaxError: {e}")
            return result

        try:
            df = self._load_data(data_path)
        except Exception as e:
            result["issues"].append(f"Data loading failed: {e}")
            return result

        if target:
            try:
                validator = CAAFEFeatureValidator(target=target)
                result["baseline_score"] = validator.evaluate_baseline(df.copy())
            except Exception as e:
                result["warnings"].append(f"Baseline evaluation skipped: {e}")

        safe_builtins = {
            'abs': abs,
            'all': all,
            'any': any,
            'dict': dict,
            'float': float,
            'int': int,
            'len': len,
            'list': list,
            'max': max,
            'min': min,
            'print': print,
            'range': range,
            'set': set,
            'str': str,
            'sum': sum,
            'tuple': tuple,
            '__import__': __import__
        }

        exec_globals = {
            '__builtins__': safe_builtins,
            'pd': pd,
            'np': __import__('numpy')
        }
        exec_locals = {
            'df': df.copy()
        }

        try:
            exec(code, exec_globals, exec_locals)
            result["executed"] = True
        except Exception as e:
            result["issues"].append(f"Execution error: {e}")
            return result

        metrics = self._extract_result_metrics(exec_locals, target, problem_type)
        result.update(metrics)

        if result.get("score") is not None and result.get("baseline_score") is not None:
            result["improvement"] = result["score"] - result["baseline_score"]
            result["improved"] = result["improvement"] > 0

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


class Manager(Agent):
    """Main orchestrator."""

    def __init__(self, llm_manager: LLMManager):
        super().__init__(llm_manager)
        self.openml = OpenMLAgent()
        self.domain_agent = DomainExpertAgent(llm_manager)
        self.feature_agent = FeatureEngineerAgent(llm_manager)
        self.modeling_agent = ModelingAgent(llm_manager)
        self.code_validation_agent = CodeValidationAgent(llm_manager)

    def process(
        self,
        path: str,
        task: str,
        domain: Optional[str] = None,
        target: Optional[str] = None,
        problem_type: Optional[str] = None,
        use_openml: bool = True,
        model: Optional[str] = None,
    ) -> dict:
        """Main processing pipeline with role separation."""
        print(f"\nProcessing: {path}")

        # Read data
        _, _, _, structure = self._read_file(path)

        result = {
            "file": path,
            "task": task,
            "timestamp": __import__('datetime').datetime.now().isoformat(),
            "pipeline": {
                "domain_analysis": {},
                "feature_specs": {},
                "modeling": {},
                "code_validation": {},
                "validation": {},
            }
        }

        # ============================================
        # 1. DOMAIN EXPERT STAGE
        # ============================================

        domain_analysis = {}
        if domain:
            print(f"\n[Domain Expert] Analyzing domain: {domain}")
            domain_result = self.domain_agent.analyze(
                path=path,
                domain=domain,
                model=model
            )
            domain_analysis = domain_result.get("analysis", {})
            result["pipeline"]["domain_analysis"] = {
                "domain": domain,
                "insights": domain_analysis.get("domain_insights", ""),
                "key_metrics": domain_analysis.get("key_metrics", []),
                "business_questions": domain_analysis.get("business_questions", []),
                "data_quality_issues": domain_analysis.get("data_quality_issues", []),
                "data_limitations": domain_analysis.get("data_limitations", []),
                "relevant_context": domain_analysis.get("relevant_context", "")
            }
            print(f"  Domain insights extracted")
            print(f"  Key metrics identified: {len(domain_analysis.get('key_metrics', []))}")

        # ============================================
        # OPENML CONTEXT
        # ============================================

        if use_openml and self.openml.dataset_index:
            print("\n[OpenML] Checking similar datasets...")
            similar = self.openml.find_similar_datasets(structure, limit=3)
            if similar:
                result["pipeline"]["domain_analysis"]["openml_context"] = {
                    "similar_count": len(similar),
                    "top_match": similar[0]["dataset"]["name"] if similar else None
                }
                print(f"  Found {len(similar)} similar datasets")

        # ============================================
        # 2. FEATURE ENGINEER STAGE (NO CODE!)
        # ============================================

        print("\n[Feature Engineer] Designing features...")

        # Feature engineer returns ONLY specifications - NO executable code
        # This uses the existing analyze() method which does NOT produce code
        feature_specs = self.feature_agent.analyze(
            path=path,
            target=target,
            model=model
        )

        # Store feature specifications
        result["pipeline"]["feature_specs"] = {
            "features": feature_specs.get("features", []),
            "scale": feature_specs.get("scale", []),
            "encode": feature_specs.get("encode", {}),
            "drop": feature_specs.get("drop", []),
            "feature_details": feature_specs.get("feature_details", ""),
            "feature_reasons": feature_specs.get("feature_reasons", ""),
            "feature_notes": feature_specs.get("feature_notes", ""),
            "derived_features": feature_specs.get("derived_features", []),
            "feature_metadata": feature_specs.get("feature_metadata", {})
        }

        print(f"  Designed {len(feature_specs.get('features', []))} features")
        print(f"  Will scale: {len(feature_specs.get('scale', []))} columns")
        print(f"  Will encode: {len(feature_specs.get('encode', {}))} columns")
        print(f"  Will drop: {len(feature_specs.get('drop', []))} columns")

        # ============================================
        # 3. MODELING STAGE (Generates code)
        # ============================================

        print("\n[Model] Generating model code from feature specs...")

        # Modeling agent takes feature specifications and generates executable code
        modeling_result = self.modeling_agent.generate(
            path=path,
            feature_info=feature_specs,  # Pass the specs (NO code inside)
            problem_type=problem_type,
            target=target,
            model=model
        )

        result["pipeline"]["modeling"] = {
            "problem_type": modeling_result.get("problem_type"),
            "target": modeling_result.get("target"),
            "recommended_models": modeling_result.get("recommended_models", []),
            "code_path": modeling_result.get("code_path"),
            "code_generated": modeling_result.get("code_generated", False),
            "code_preview": modeling_result.get("code_preview", "")
        }

        print(f"  Problem type: {modeling_result.get('problem_type', 'unknown')}")
        print(f"  Target: {modeling_result.get('target', 'unknown')}")
        print(f"  Recommended models: {', '.join(modeling_result.get('recommended_models', []))}")
        if modeling_result.get('code_generated'):
            print(f"  Code saved to: {modeling_result.get('code_path')}")

        # ============================================
        # 4. CODE VALIDATION AGENT STAGE (Check generated code and run it)
        # ============================================
        print("\n[Code Validation Agent] Checking generated code execution and results...")
        code_validation_result = {}
        code_path = modeling_result.get('code_path')
        if modeling_result.get('code_generated') and code_path:
            code_validation_result = self.code_validation_agent.validate_code(
                code_path=str(code_path),
                data_path=path,
                target=target,
                problem_type=problem_type
            )
            print(f"  Syntax OK: {code_validation_result.get('syntax_ok')}")
            print(f"  Executed: {code_validation_result.get('executed')}")
            if code_validation_result.get('score') is not None:
                print(f"  Validation score: {code_validation_result.get('score'):.4f}")
            if code_validation_result.get('baseline_score') is not None:
                print(f"  Baseline score: {code_validation_result.get('baseline_score'):.4f}")
            issues = code_validation_result.get('issues') or []
            if issues:
                print(f"  Issues: {len(issues)}")
        else:
            code_validation_result = {
                "executed": False,
                "issues": ["No code generated or code path missing"],
                "notes": ["Skipped code validation"]
            }
            print("  No generated code available for validation")

        result["pipeline"]["code_validation"] = code_validation_result

        # ============================================
        # 5. CAAFE VALIDATOR STAGE (Validates model code)
        print("\n[CAAFE Validator] Validating generated model code...")

        validation_result = {
            "validated": False,
            "stage": "post_modeling",
            "metrics": {}
        }

        # Only validate if:
        # 1. Code was generated
        # 2. We have a target column
        # 3. The code file exists
        if modeling_result.get('code_generated') and target:
            code_path = modeling_result.get('code_path')
            if code_path and Path(code_path).exists():
                try:
                    print("  Loading data for validation...")
                    # Support both CSV and ARFF files
                    if path.endswith('.arff'):
                        try:
                            from scipy.io import arff
                            data, meta = arff.loadarff(path)
                            df = pd.DataFrame(data)
                            # Decode bytes to strings
                            for col in df.select_dtypes(include=['object']).columns:
                                try:
                                    if df[col].iloc[0] and isinstance(df[col].iloc[0], bytes):
                                        df[col] = df[col].str.decode('utf-8')
                                except (IndexError, AttributeError, UnicodeDecodeError):
                                    pass
                        except ImportError:
                            print(f"  [ERROR] scipy required for ARFF support. Install with: pip install scipy")
                            raise
                    else:
                        df = pd.read_csv(path)

                    # Initialize validator
                    validator = CAAFEFeatureValidator(target=target)

                    # Evaluate baseline performance
                    print("  Evaluating baseline performance...")
                    validator.evaluate_baseline(df)
                    baseline_score = validator.baseline_score
                    print(f"  Baseline score: {baseline_score:.4f}")

                    # Validate the generated model code
                    print("  Validating model code...")
                    is_valid, metrics = validator.validate_model_code(
                        code_path,
                        df
                    )

                    validation_result = {
                        "validated": is_valid,
                        "code_path": code_path,
                        "baseline_score": baseline_score,
                        "metrics": metrics,
                        "model_score": metrics.get("score", 0),
                        "improvement": metrics.get("score", 0) - baseline_score if is_valid else 0.0,
                        "issues": metrics.get("issues", []),
                        "warnings": metrics.get("warnings", [])
                    }

                    if is_valid:
                        print(f"  Validation PASSED!")
                        print(f"  Model score: {metrics.get('score', 0):.4f}")
                        print(f"  Improvement over baseline: {validation_result['improvement']:.4f}")
                    else:
                        print(f"  Validation FAILED")
                        if validation_result.get("issues"):
                            print(f"  Issues: {len(validation_result['issues'])}")

                except Exception as e:
                    validation_result = {
                        "validated": False,
                        "error": str(e),
                        "stage": "validation_error"
                    }
                    print(f"  Validation error: {e}")
            else:
                validation_result = {
                    "validated": False,
                    "note": "Code file not found",
                    "stage": "file_not_found"
                }
                print("  Code file not found for validation")
        else:
            if not modeling_result.get('code_generated'):
                validation_result["note"] = "No code generated for validation"
                print("  No code generated - skipping validation")
            elif not target:
                validation_result["note"] = "No target column specified for validation"
                print("  No target column - skipping validation")

        result["pipeline"]["validation"] = validation_result

        # ============================================

        result["summary"] = {
            "features_count": len(feature_specs.get('features', [])),
            "problem_type": modeling_result.get('problem_type'),
            "target": modeling_result.get('target'),
            "code_generated": modeling_result.get('code_generated', False),
            "models": modeling_result.get('recommended_models', []),
            "validation_passed": validation_result.get('validated', False),
            "improvement": validation_result.get('improvement') if validation_result.get('validated') else None
        }

        # Print final summary
        print(f"\n{'='*60}")
        print(f" Pipeline Complete!")
        print(f"{'='*60}")
        print(f"  Features designed: {len(feature_specs.get('features', []))}")
        print(f"  Problem type: {modeling_result.get('problem_type', 'unknown')}")
        print(f"  Models: {', '.join(modeling_result.get('recommended_models', []))}")

        if modeling_result.get('code_generated'):
            print(f"  Code generated: {modeling_result['code_path']}")

        if validation_result.get('validated'):
            print(f"  Validation passed with improvement: {validation_result['improvement']:.4f}")
        elif validation_result.get('note'):
            print(f"  Validation: {validation_result['note']}")

        if modeling_result.get('code_path'):
            print(f"\n   Run: python {modeling_result['code_path']}")
        print(f"{'='*60}")

        return result

    def quick_process(self, path: str, target: Optional[str] = None) -> dict:
        """Ultra-fast processing - minimal output."""
        _, _, _, structure = self._read_file(path)
        feature_info = self.feature_agent.analyze(path, target)
        modeling_result = self.modeling_agent.generate(path, feature_info, target=target)

        return {
            "file": path,
            "target": modeling_result.get('target'),
            "problem": modeling_result.get('problem_type'),
            "models": modeling_result.get('recommended_models', [])[:2],
            "code": modeling_result.get('code_path')
        }