"""
Agent Module for LLM-Powered Data Science Pipeline
"""

import csv
import io
import json
import mimetypes
import re
from typing import Any, Dict, List, Optional
from pathlib import Path
from llms import LLMManager
from openMLAgentLocal import OpenMLAgent
from caafeValidator import CAAFEFeatureValidator, extract_code_from_response


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

        import re

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
            COLS: {headers[:10]}
            TYPES: {data_types}
            TARGET: {target if target else 'auto'}

            RETURN ONLY VALID JSON - NO OTHER TEXT:
            {{"features":["col1","col2"],"scale":["numeric_col"],"encode":{{"cat_col":"label"}},"drop":["id_col"]}}"""

        resp = self.llm.generate(prompt, model=model, max_tokens=800)
        recommendations = self._extract_json(resp.get("text", ""))

        if not recommendations:
            recommendations = self._default_features(headers, data_types)

        return {
            "features": recommendations.get("features", headers[:10]),
            "scale": recommendations.get("scale", []),
            "encode": recommendations.get("encode", {}),
            "drop": recommendations.get("drop", [])
        }

    def _default_features(self, headers: List[str], data_types: Dict[str, str]) -> dict:
        numeric = [h for h in headers if data_types.get(h) in ['numeric', 'float']]
        categorical = [h for h in headers if data_types.get(h) == 'categorical']

        return {
            "features": headers[:10],
            "scale": numeric[:5],
            "encode": {col: "label" for col in categorical[:3]},
            "drop": [h for h in headers if 'id' in h.lower() or 'date' in h.lower()]
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
        if feature_info:
            features = feature_info.get('features', [])[:8]
            scale = feature_info.get('scale', [])[:4]
            encode = feature_info.get('encode', {})
            drop = feature_info.get('drop', [])[:3]
            feature_context = f"\nFEATURES: {features}\nSCALE: {scale}\nENCODE: {encode}\nDROP: {drop}"

        # Get sample - limited to save tokens
        sample_lines = text.splitlines()[:3]
        sample = "\n".join(sample_lines) if len(sample_lines) > 1 else ""

        prompt = f"""{Path(path).name} | {rows} rows | {len(headers)} cols
            Target: {target or 'auto'}{feature_context}

            SAMPLE:
            {sample[:300]}

            STRICT JSON OUTPUT - NO TEXT, NO EXPLANATION:
            {{"problem":"classification|regression","target":"col_name","models":["model1"],"code":"PYTHON CODE HERE"}}

            YOUR JSON:"""

        # Use 6000 tokens to avoid timeout
        resp = self.llm.generate(prompt, model=model, max_tokens=6000)
        response_text = resp.get("text", "")

        # Extract JSON
        result = self._extract_json(response_text)

        # Fallback: extract code block directly
        if not result and 'import pandas' in response_text:
            print("[INFO] JSON extraction failed, extracting code directly")
            result = self._extract_code_from_response(response_text)

        if not result:
            result = self._find_json_in_response(response_text)

        # Extract code from result (handle None case)
        code = None
        if result:
            code = result.get("code")

        # If code is still empty, try direct extraction
        if not code or (isinstance(code, str) and len(code) < 50):
            code = self._extract_code_block(response_text)

        # Fix cutoff issues in code (only if code is not None)
        if code and isinstance(code, str):
            code = self._clean_code(code)

        # Save code if valid
        code_path = None
        if code and isinstance(code, str) and len(code) > 100:
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
            print(f"[WARNING] No valid code generated (length: {code_length})")

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
        import re

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
        import re

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

        import re

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
        import re
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

        print(f"\n✅ Pipeline complete!")
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