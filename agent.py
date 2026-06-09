"""
Agent Module for LLM-Powered Data Science Pipeline
"""

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
    def _get_structure(content: str, mimetype: str, filename: str = "") -> Dict[str, Any]:
        """Extract basic structure from CSV/text data."""
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

            You are a pandas feature-engineering assistant.
            Your final answer must be only one valid JSON object.
            Do not include any explanation, analysis, reasoning, or markdown.
            Use actual column names from COLS and select useful features, scaling, encoding, and drops.
            If the dataset contains obvious identifier or metadata columns, drop them in DROP.
            If you can create a new derived feature, include it in FEATURE_CODE.

            RETURN ONLY VALID JSON - NO OTHER TEXT.
            Use actual column names from COLS; do not use placeholders like col1, numeric_col, cat_col, or id_col.
            Output exactly one JSON object with keys: features, scale, encode, drop, feature_code.
            Optional keys may include: feature_details, feature_reasons, feature_notes, derived_features, feature_metadata.            If you have extra context about why a feature was selected, include it in feature_metadata.
            Keep metadata concise and useful.            If you cannot produce valid JSON, return {{"error":"json"}} only.
            Example output:
                {{"features":["age","salary"],"scale":["age"],"encode":{{"gender":"label"}},"drop":["customer_id"],"feature_code":"df['age_squared'] = df['age'] ** 2","feature_details":"Scale numeric columns and label encode gender.","derived_features":["age_squared"],"feature_metadata":{{"reason":"high correlation with churn"}}}}"""

        resp = self.llm.generate(prompt, model=model, max_tokens=800)
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
            If you can create a derived feature, include it in FEATURE_CODE.
            Optional keys may include: feature_details, feature_reasons, feature_notes, derived_features, feature_metadata.
            RETURN ONLY VALID JSON - NO OTHER TEXT.
            Output exactly one JSON object with keys: features, scale, encode, drop, feature_code.
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
                        "feature_code": "",
                        "feature_metadata": {"note": "wrapped encode-only response"}
                    }
                    if self._valid_feature_recommendations(wrapped):
                        recommendations = wrapped
                        break

            # If the response contained some keys but not the required top-level ones,
            # ask the model to complete the JSON using the partial as context.
            if isinstance(recommendations, dict) and recommendations and not self._valid_feature_recommendations(recommendations):
                completion_prompt = f"""You returned a partial JSON: {json.dumps(recommendations)}\n\nUsing the DATA: {path} with COLS: {headers} and TYPES: {data_types}, produce ONLY one valid JSON object with keys: features, scale, encode, drop, feature_code.\nOptional keys may include: feature_details, feature_reasons, feature_notes, derived_features, feature_metadata.\nDo not invent columns outside COLS. Fill missing keys sensibly and use actual column names.\nIf you cannot produce valid JSON, return {{"error":"json"}} only.\nRETURN ONLY VALID JSON."""
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
            "feature_code": recommendations.get("feature_code", ""),
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
                    "feature_code": "",
                    "feature_metadata": {"note": "wrapped encode-only response"}
                }
                if self._valid_feature_recommendations(wrapped):
                    return wrapped

            # Heuristic: if the model returned a partial feature object, fill missing keys.
            if any(key in recommendations for key in ["features", "scale", "encode", "drop", "feature_code"]):
                wrapped = {
                    "features": recommendations.get("features", [h for h in headers if h != target][:10]),
                    "scale": recommendations.get("scale", [h for h in headers if data_types.get(h) in ['numeric', 'float']][:5]),
                    "encode": recommendations.get("encode", {}),
                    "drop": recommendations.get("drop", [h for h in headers if 'id' in h.lower() or 'date' in h.lower()]),
                    "feature_code": recommendations.get("feature_code", ""),
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
                "feature_code": "",
                "feature_metadata": {"note": "wrapped feature list response"}
            }

        return recommendations

    def _valid_feature_recommendations(self, recommendations: Any) -> bool:
        if not isinstance(recommendations, dict):
            return False

        required_keys = {"features", "scale", "encode", "drop", "feature_code"}
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
        if not isinstance(recommendations.get("feature_code"), str):
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
            "feature_code": "",
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
            validated_code = feature_info.get('validated_feature_code', "") or ""
            feature_details = feature_info.get('feature_details', "") or ""
            feature_reasons = feature_info.get('feature_reasons', "") or ""
            feature_notes = feature_info.get('feature_notes', "") or ""
            derived_features = feature_info.get('derived_features', [])[:4]
            feature_metadata = feature_info.get('feature_metadata', {}) or {}
            feature_context = f"\nFEATURES: {features}\nSCALE: {scale}\nENCODE: {encode}\nDROP: {drop}"
            if validated_code:
                feature_context += f"\nVALIDATED_FEATURE_CODE:\n{validated_code}"
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

        prompt = f"""{Path(path).name} | {rows} rows | {len(headers)} cols
            Target: {target or 'auto'}{feature_context}
            If VALIDATED_FEATURE_CODE is provided, use it to transform the dataset before modeling.
            Use the feature engineering details to choose appropriate preprocessing and models.
            Prefer simple, robust models for small datasets and use actual column names.
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
        resp = self.llm.generate(prompt, model=model, max_tokens=6000)
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

        # Fallback: extract code block directly
        if not result and 'import pandas' in response_text:
            print("[INFO] JSON extraction failed, extracting code directly")
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

        if accept:
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

        return code


class Manager(Agent):
    """Main orchestrator."""

    def __init__(self, llm_manager: LLMManager):
        super().__init__(llm_manager)
        self.openml = OpenMLAgent()
        self.domain_agent = DomainExpertAgent(llm_manager)
        self.feature_agent = FeatureEngineerAgent(llm_manager)
        self.modeling_agent = ModelingAgent(llm_manager)

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
        """Main processing pipeline."""
        print(f"\nProcessing: {path}")

        _, _, _, structure = self._read_file(path)
        result = {
            "file": path,
            "task": task,
            "timestamp": __import__('datetime').datetime.now().isoformat(),
            "pipeline": {}
        }

        # Check OpenML if enabled
        if use_openml and self.openml.dataset_index:
            print("\nChecking similar datasets...")
            similar = self.openml.find_similar_datasets(structure, limit=3)
            if similar:
                print(f"  Found {len(similar)} similar datasets")
                result["openml_context"] = {
                    "similar_count": len(similar),
                    "top_match": similar[0]["dataset"]["name"] if similar else None
                }

        # Feature engineering
        print("\nFeature engineering...")
        feature_info = self.feature_agent.analyze(path, target, model)
        result["pipeline"]["features"] = feature_info
        print(f"   → {len(feature_info.get('features', []))} features selected")

        # Feature validation
        validation_result = {
            "validated": False,
            "note": "No feature code available for validation"
        }
        feature_code = feature_info.get("feature_code", "")
        if target and feature_code:
            try:
                print("\nValidating generated feature code...")
                df = pd.read_csv(path)
                validator = CAAFEFeatureValidator(target=target)
                validator.evaluate_baseline(df)
                is_improved, score = validator.evaluate_feature(df, feature_code)
                validation_result = {
                    "validated": is_improved,
                    "feature_code": feature_code,
                    "score": score,
                    "improvement": score - validator.baseline_score if is_improved else 0.0
                }
                feature_info["validated_feature_code"] = feature_code if is_improved else ""
                print(f"   → Feature validation passed: {is_improved}")
            except Exception as e:
                validation_result = {"validated": False, "error": str(e)}
                print(f"   → Feature validation error: {e}")

        result["pipeline"]["feature_validation"] = validation_result

        # Modeling
        print("\nModel generation...")
        modeling_result = self.modeling_agent.generate(
            path, feature_info, problem_type, target, model
        )
        result["pipeline"]["modeling"] = modeling_result
        print(f"   → Problem type: {modeling_result.get('problem_type', 'unknown')}")
        print(f"   → Target: {modeling_result.get('target', 'unknown')}")
        print(f"   → Models: {', '.join(modeling_result.get('recommended_models', []))}")

        # Summary
        result["summary"] = {
            "features_count": len(feature_info.get('features', [])),
            "problem_type": modeling_result.get('problem_type'),
            "target": modeling_result.get('target'),
            "code_generated": modeling_result.get('code_generated', False),
            "models": modeling_result.get('recommended_models', [])
        }

        # Domain analysis if domain provided
        if domain:
            print(f"\n Adding domain context: {domain}")
            result["domain_context"] = {"domain": domain, "note": "Domain expert analysis available if needed"}

        print(f"\n Pipeline complete!")
        if modeling_result.get('code_path'):
            print(f" Ready to run: python {modeling_result['code_path']}")

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