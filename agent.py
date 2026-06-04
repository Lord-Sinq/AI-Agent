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
        """Extract JSON from LLM response - simplified version."""
        if not text:
            return None

        # Remove think tags and everything before them
        if '</think>' in text:
            text = text.split('</think>')[-1]

        # Remove markdown code blocks
        text = re.sub(r'```json\s*|```\s*', '', text)

        # Find JSON
        start = text.find('{')
        if start == -1:
            start = text.find('[')

        if start != -1:
            # Try to parse from start to end
            try:
                return json.loads(text[start:])
            except:
                pass

            # Try brace matching
            brace_count = 0
            for i in range(start, len(text)):
                if text[i] == '{':
                    brace_count += 1
                elif text[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        try:
                            return json.loads(text[start:i+1])
                        except:
                            break

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
        sample = "\n".join(text.splitlines()[:5])  # Only 5 rows

        prompt = f"""Data: {path}
        Target: {target if target else 'infer from data'}
        Structure: {json.dumps(structure, indent=2)}
        Sample: {sample}

        Return ONLY JSON:
        {{"recommended_features":["col1","col2"],
        "feature_transformations":{{"col":"desc"}},
        "encoding_strategies":{{"cat_col":"label_encode"}},
        "scaling_recommendations":{{"numeric_features":["col1"],
        "method":"StandardScaler"}},
        "feature_quality_issues":[]}}"""

        resp = self.llm.generate(prompt, model=model, max_tokens=1500)
        analysis_text = resp.get("text", "")
        recommendations = self._extract_json(analysis_text) or {}

        return {"structure": structure, "recommendations": recommendations}


class ModelingAgent(Agent):
    """Modeling and code generation agent."""

    def generate(self, path: str, problem_type: Optional[str] = None, target: Optional[str] = None, model: Optional[str] = None) -> dict:
        content, mt, text, structure = self._read_file(path)
        sample = "\n".join(text.splitlines()[:5])  # Only 5 rows

        headers = structure.get('headers', [])
        rows = structure.get('rows', 0)

        prompt = f"""Generate Python code for {path}
        Rows: {rows}
        Columns: {headers}
        Target: {target if target else 'infer from data'}
        Problem: {problem_type if problem_type else 'infer from data'}
        Sample: {sample}

        Return ONLY JSON:
        {{"inferred_problem_type":"classification",
        "inferred_target":"column_name",
        "recommended_models":["RandomForestClassifier"],
        "python_code":"import pandas as pd\\n# code here"}}

        Code must:
        1. Load data from '{path}'
        2. Handle preprocessing (drop IDs, handle missing, encode categoricals)
        3. Train/test split
        4. Train appropriate model
        5. Evaluate with metrics
        6. Show example prediction"""

        try:
            resp = self.llm.generate(prompt, model=model, max_tokens=4000)
            analysis_text = resp.get("text", "")
            result = self._extract_json(analysis_text) or {}
        except Exception as e:
            print(f"[ERROR] Modeling generation failed: {e}")
            result = {}

        code = result.get("python_code", "")
        code_path = None
        if code:
            code_filename = f"{Path(path).stem}_model.py"
            code_path = Path.cwd() / "generated_code" / code_filename
            code_path.parent.mkdir(exist_ok=True)
            code_path.write_text(code)
            print(f"[INFO] Code saved to {code_path}")

        return {
            "inferred_problem_type": result.get("inferred_problem_type"),
            "inferred_target": result.get("inferred_target"),
            "recommended_models": result.get("recommended_models", []),
            "code_generated": bool(code),
            "code_path": str(code_path) if code_path else None,
            "generated_code": code,
        }


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
        result = {"file": path, "task": task, "results": {}, "summary": {}}

        # Check OpenML if enabled
        if use_openml and self.openml.dataset_index:
            print("\nChecking similar datasets...")
            similar = self.openml.find_similar_datasets(structure, limit=3)
            if similar:
                print(f"  Found {len(similar)} similar datasets")
                result["openml_similar"] = [{"name": s["dataset"]["name"], "similarity": s["similarity_score"]} for s in similar]

        # Feature engineering
        print("\nFeature engineering...")
        result["results"]["features"] = self.feature_agent.analyze(path, target, model)
        result["summary"]["features_done"] = True

        # Modeling
        print("\nModel generation...")
        result["results"]["modeling"] = self.modeling_agent.generate(path, problem_type, target, model)
        result["summary"]["modeling_done"] = True
        result["summary"]["code_generated"] = result["results"]["modeling"]["code_generated"]

        # Domain analysis if domain provided
        if domain:
            print(f"\nDomain analysis ({domain})...")
            result["results"]["domain"] = self.domain_agent.analyze(path, domain, model)
            result["summary"]["domain_done"] = True

        return result