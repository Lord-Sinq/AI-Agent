"""
Agent Module for LLM-Powered Data Science Pipeline

This module provides a set of agent classes that leverage Large Language Models
(LLMs) to perform specialized tasks in a data science workflow.
"""

import csv
import io
import json
import mimetypes
from typing import Any, Dict, List, Optional, Union
from openMLAgent import OpenMLAgent
from pathlib import Path
from llms import LLMManager


class Agent:
    """Base agent for file and data processing."""

    def __init__(self, llm_manager: LLMManager):
        """Initialize an Agent with an LLM manager."""
        self.llm = llm_manager

    @staticmethod
    def detect_mimetype(filename: str, content: bytes) -> str:
        """Detect the MIME type of a file based on its name and content."""
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
        """Convert file content to a text representation for LLM processing."""
        if mimetype.startswith("text/") or mimetype in (
            "application/json",
            "application/csv",
        ):
            try:
                return content.decode("utf-8")
            except Exception:
                return content.decode("latin-1", errors="ignore")
        return content[:1024].hex()

    @staticmethod
    def _extract_json(text: Optional[str]) -> Any:
        """Extract JSON from text, handling truncated or malformed responses."""
        if text is None:
            return None

        start = text.find("{")
        if start == -1:
            start = text.find("[")
            if start == -1:
                return None

        brace_count = 0
        bracket_count = 0
        end = -1
        in_string = False
        escape_next = False

        for i in range(start, len(text)):
            char = text[i]

            if escape_next:
                escape_next = False
                continue

            if char == "\\":
                escape_next = True
                continue

            if char == '"' and not escape_next:
                in_string = not in_string
                continue

            if not in_string:
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                elif char == "[":
                    bracket_count += 1
                elif char == "]":
                    bracket_count -= 1

                if brace_count == 0 and bracket_count == 0 and (char == "}" or char == "]"):
                    end = i
                    break

        if end == -1:
            json_str = text[start:]
            if json_str.rstrip().endswith('"'):
                json_str += "}"
            elif json_str.rstrip().endswith(","):
                json_str = json_str.rstrip()[:-1] + "}"
            else:
                if json_str.rstrip().endswith("["):
                    json_str = json_str.rstrip() + "]"
                elif json_str.rstrip().endswith("{"):
                    json_str = json_str.rstrip() + "}"
                else:
                    json_str = json_str.rstrip() + '"}'

            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                return None

        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _coerce_value(value: str) -> Any:
        """Convert a string value to its appropriate data type."""
        try:
            if "." in value:
                return float(value)
            return int(value)
        except Exception:
            return value.lower() if isinstance(value, str) else value

    @staticmethod
    def _get_data_structure_info(content: str, mimetype: str, filename: str = "") -> Dict[str, Any]:
        """Extract structural information from data content."""
        lines = [l for l in content.splitlines() if l.strip()]
        if not lines:
            return {"rows": 0, "columns": 0, "headers": []}

        info: Dict[str, Any] = {"rows": len(lines), "sample_lines": min(10, len(lines))}

        is_csv = mimetype == "text/csv" or filename.lower().endswith(".csv") or ("," in lines[0] and len(lines[0].split(",")) > 1)

        if is_csv:
            try:
                csv_content = "\n".join(lines[:10])
                csv_reader = csv.DictReader(io.StringIO(csv_content))

                if csv_reader.fieldnames:
                    info["headers"] = list(csv_reader.fieldnames)
                    info["columns"] = len(csv_reader.fieldnames)

                    data_types: Dict[str, str] = {}
                    rows_checked = 0

                    for row in csv_reader:
                        if rows_checked >= 5:
                            break
                        for header in csv_reader.fieldnames:
                            if header in row and row[header] and row[header].strip():
                                val = row[header].strip()
                                if val.isdigit():
                                    data_types[header] = "numeric"
                                elif val.replace(".", "").replace("-", "").isdigit():
                                    data_types[header] = "float"
                                else:
                                    if header not in data_types:
                                        data_types[header] = "categorical"
                                    elif data_types[header] not in ["numeric", "float"]:
                                        data_types[header] = "categorical"
                        rows_checked += 1

                    info["data_types"] = data_types
                else:
                    info["headers"] = []
                    info["columns"] = 0

            except Exception as e:
                info["csv_parse_error"] = str(e)
                info["headers"] = []
                info["columns"] = 0
        else:
            info["headers"] = []
            info["columns"] = 0
            info["file_type"] = "text"

            if content.strip().startswith("{") or content.strip().startswith("["):
                info["is_json_likely"] = True
                try:
                    json_data = json.loads(content[:1000])
                    if isinstance(json_data, dict):
                        info["json_keys"] = list(json_data.keys())
                        info["columns"] = len(json_data.keys())
                    elif isinstance(json_data, list) and json_data:
                        info["array_length"] = len(json_data)
                        if isinstance(json_data[0], dict):
                            info["json_keys"] = list(json_data[0].keys())
                            info["columns"] = len(json_data[0].keys())
                except Exception:
                    pass

        return info


class DomainExpertAgent(Agent):
    """Agent that analyzes data from a domain-specific perspective."""

    def _build_domain_prompt(self, filename: str, sample: str, mimetype: str, domain: str, structure_info: Dict[str, Any]) -> str:
        """Build a prompt for the LLM to perform domain-specific analysis."""
        return (
            f"You are a {domain} domain expert and data analyst. Analyze the following data "
            f"from a {domain} perspective.\n\n"
            f"Filename: {filename}\n"
            f"Data structure: {json.dumps(structure_info, indent=2)}\n\n"
            f"Sample data:\n{sample}\n\n"
            "Provide your analysis as JSON with the following fields:\n"
            "- domain_insights (str): Key insights from a domain perspective\n"
            "- key_metrics (list): Important metrics or KPIs to track\n"
            "- data_quality_issues (list): Any data quality problems identified\n"
            "- business_questions (list): Important questions this data could answer\n"
            "- relevant_context (str): Domain-specific context or considerations\n"
            "- data_limitations (list): Limitations of this dataset for domain analysis\n"
            "Return ONLY valid JSON, no other text."
        )

    def analyze_data(self, path: str, domain: str, model: Optional[str] = None, provider: Optional[str] = None) -> dict:
        """Analyze data from a domain-specific perspective."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {path}")

        content = p.read_bytes()
        mt = self.detect_mimetype(p.name, content)
        text = self._prepare_text(content, mt)

        structure_info = self._get_data_structure_info(text, mt, p.name)
        sample = "\n".join(text.splitlines()[:30])

        prompt = self._build_domain_prompt(p.name, sample, mt, domain, structure_info)
        llm_resp = self.llm.generate(prompt, model=model, provider=provider)
        analysis_text = llm_resp.get("text", "")

        analysis = self._extract_json(analysis_text) or {}

        return {
            "meta": {"filename": p.name, "content_type": mt, "size": len(content), "domain": domain},
            "structure": structure_info,
            "analysis": analysis,
            "raw_response": llm_resp,
        }


class FeatureEngineerAgent(Agent):
    """Agent that creates and selects features for machine learning models."""

    def _build_feature_prompt(
        self,
        filename: str,
        sample: str,
        mimetype: str,
        target_variable: Optional[str],
        structure_info: Dict[str, Any],
        openml_suggestions: Optional[List[Dict]] = None,
    ) -> str:
        """Build a prompt with optional OpenML suggestions."""

        openml_context = ""
        if openml_suggestions:
            openml_context = f"\n\nOpenML best practices for similar datasets suggest these preprocessing steps:\n{json.dumps(openml_suggestions, indent=2)}\n"

        target_context = f"Target variable for prediction: {target_variable}" if target_variable else "No specific target variable provided"

        return (
            f"You are a feature engineering expert. Analyze the following data and recommend "
            f"feature engineering strategies for machine learning.\n\n"
            f"Filename: {filename}\n"
            f"{target_context}\n"
            f"Data structure: {json.dumps(structure_info, indent=2)}\n"
            f"{openml_context}"
            f"Sample data:\n{sample}\n\n"
            "Provide your analysis as JSON with the following fields:\n"
            "- recommended_features (list): Features to create or use\n"
            "- feature_transformations (dict): Suggested transformations per feature\n"
            "- feature_importance (dict): Estimated importance of each feature (1-10)\n"
            "- feature_interactions (list): Potential interaction features to create\n"
            "- encoding_strategies (dict): Recommended encoding for categorical features\n"
            "- scaling_recommendations (dict): Scaling approaches for numerical features\n"
            "- feature_quality_issues (list): Problems with existing features\n"
            "- dimensionality_reduction (str): Suggestions if dimensionality reduction is needed\n"
            "Return ONLY valid JSON, no other text."
        )

    def engineer_features(
        self, path: str, target_variable: Optional[str] = None, model: Optional[str] = None, provider: Optional[str] = None
    ) -> dict:
        """Analyze data and recommend feature engineering strategies."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {path}")

        content = p.read_bytes()
        mt = self.detect_mimetype(p.name, content)
        text = self._prepare_text(content, mt)

        structure_info = self._get_data_structure_info(text, mt, p.name)
        sample = "\n".join(text.splitlines()[:30])

        prompt = self._build_feature_prompt(p.name, sample, mt, target_variable, structure_info)
        llm_resp = self.llm.generate(prompt, model=model, provider=provider, max_tokens=2000)
        analysis_text = llm_resp.get("text", "")

        print(f"[DEBUG] Feature Engineering Response length: {len(analysis_text)}")

        recommendations = self._extract_json(analysis_text) or {}

        return {
            "meta": {"filename": p.name, "content_type": mt, "size": len(content), "target_variable": target_variable},
            "structure": structure_info,
            "recommendations": recommendations,
            "raw_response": llm_resp,
        }


class CompSciModelingExpertAgent(Agent):
    """Agent that recommends ML models AND generates executable code."""

    def _build_modeling_prompt(
        self,
        filename: str,
        sample: str,
        mimetype: str,
        problem_type: Optional[str],
        target_variable: Optional[str],
        constraints: Optional[Dict[str, Any]],
        structure_info: Dict[str, Any],
    ) -> str:
        """Build a prompt for the LLM to recommend ML models and generate code."""
        problem_context = f"Problem type: {problem_type}" if problem_type else "Problem type not specified - infer from data"
        target_context = (
            f"Target variable: {target_variable}"
            if target_variable
            else "No target variable specified - infer what would be interesting to predict"
        )
        constraints_context = f"Constraints: {json.dumps(constraints)}" if constraints else "No specific constraints provided"

        return (
            f"You are a machine learning expert who writes production-ready code. Analyze the following data "
            f"and provide both model recommendations AND executable Python code.\n\n"
            f"Filename: {filename}\n"
            f"{problem_context}\n"
            f"{target_context}\n"
            f"{constraints_context}\n"
            f"Data structure: {json.dumps(structure_info, indent=2)}\n\n"
            f"Sample data:\n{sample}\n\n"
            "IMPORTANT: Even if no target variable is specified, identify the most interesting column to predict "
            "or suggest unsupervised learning approaches.\n\n"
            "Provide your response as JSON with the following fields:\n"
            "- inferred_problem_type (str): What problem type you're solving\n"
            "- inferred_target (str): What target variable you're predicting (if any)\n"
            "- recommended_models (list): Top 3-5 models with justification\n"
            "- model_comparison (dict): Compare models on accuracy, speed, interpretability\n"
            "- evaluation_metrics (list): Appropriate metrics for this problem\n"
            "- training_strategy (str): Suggested training approach\n"
            "- validation_method (str): Recommended validation strategy\n"
            "- python_code (str): Complete, runnable Python code that:\n"
            "    * Loads and preprocesses the data\n"
            "    * Performs exploratory data analysis\n"
            "    * Performs feature engineering based on recommendations\n"
            "    * Trains the best model(s)\n"
            "    * Evaluates performance with appropriate metrics\n"
            "    * Saves the trained model\n"
            "    * Includes example predictions\n"
            "    * Has proper error handling and comments\n"
            "- potential_challenges (list): Known challenges with this data/modeling\n"
            "- optimization_tips (list): Hyperparameter tuning suggestions\n"
            "- deployment_considerations (list): Factors for production deployment\n"
            "Return ONLY valid JSON, no other text."
        )

    def recommend_models_and_generate_code(
        self,
        path: str,
        problem_type: Optional[str] = None,
        target_variable: Optional[str] = None,
        constraints: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> dict:
        """Analyze data, recommend ML models, and generate executable code."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {path}")

        content = p.read_bytes()
        mt = self.detect_mimetype(p.name, content)
        text = self._prepare_text(content, mt)

        structure_info = self._get_data_structure_info(text, mt, p.name)
        sample = "\n".join(text.splitlines()[:30])

        prompt = self._build_modeling_prompt(p.name, sample, mt, problem_type, target_variable, constraints, structure_info)

        # Increase max_tokens for code generation
        llm_resp = self.llm.generate(prompt, model=model, provider=provider, max_tokens=4000)
        analysis_text = llm_resp.get("text", "")

        # Debug: print first 500 chars of response
        print(f"[DEBUG] LLM Response length: {len(analysis_text)}")
        print(f"[DEBUG] LLM Response preview: {analysis_text[:500]}...")

        recommendations = self._extract_json(analysis_text) or {}

        if not recommendations:
            print(f"[WARNING] Failed to extract JSON from response. Full response:\n{analysis_text}")

        generated_code = recommendations.get("python_code", "")

        code_file_path = None
        if generated_code:
            code_filename = f"{p.stem}_model_code.py"
            code_file_path = Path.cwd() / "generated_code" / code_filename
            code_file_path.parent.mkdir(exist_ok=True)
            code_file_path.write_text(generated_code)
            print(f"[INFO] Code saved to: {code_file_path}")
        else:
            print("[WARNING] No code generated in LLM response")

        return {
            "meta": {
                "filename": p.name,
                "content_type": mt,
                "size": len(content),
                "problem_type": problem_type,
                "target_variable": target_variable,
                "constraints": constraints,
            },
            "structure": structure_info,
            "inferred_problem_type": recommendations.get("inferred_problem_type"),
            "inferred_target": recommendations.get("inferred_target"),
            "recommendations": {
                "recommended_models": recommendations.get("recommended_models", []),
                "model_comparison": recommendations.get("model_comparison", {}),
                "evaluation_metrics": recommendations.get("evaluation_metrics", []),
                "training_strategy": recommendations.get("training_strategy", ""),
                "validation_method": recommendations.get("validation_method", ""),
                "potential_challenges": recommendations.get("potential_challenges", []),
                "optimization_tips": recommendations.get("optimization_tips", []),
                "deployment_considerations": recommendations.get("deployment_considerations", []),
            },
            "generated_code": generated_code,
            "code_file_path": str(code_file_path) if code_file_path else None,
            "raw_response": llm_resp,
        }


class Manager(Agent):
    """Orchestrates the workflow between specialized agents."""

    def __init__(self, llm_manager: LLMManager):
        """Initialize Manager with LLM and OpenML."""
        super().__init__(llm_manager)
        self.openml_agent = OpenMLAgent()

    def orchestrate_pipeline(
        self,
        path: str,
        task: str,
        domain: Optional[str] = None,
        target_variable: Optional[str] = None,
        problem_type: Optional[str] = None,
        constraints: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        use_openml_recommendations: bool = True,
    ) -> dict:
        """Orchestrate the complete data science pipeline with OpenML insights."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {path}")

        content = p.read_bytes()
        mt = self.detect_mimetype(p.name, content)
        text = self._prepare_text(content, mt)

        structure_info = self._get_data_structure_info(text, mt, p.name)
        sample = "\n".join(text.splitlines()[:30])

        result: Dict[str, Any] = {"file_path": path, "task": task, "domain_used": domain, "results": {}, "summary": {}}

        # Check OpenML for similar datasets and recommendations
        openml_insights = {}
        if use_openml_recommendations:
            print("\n🔍 Checking OpenML for similar datasets and best practices...")
            print("   (This may take a few seconds. Timeout set to 15 seconds)")

            try:
                # Set a timeout for the entire OpenML check
                import signal

                def timeout_handler(signum, frame):
                    raise TimeoutError("OpenML check timed out")

                # Set timeout for OpenML operations (optional, if you want to limit total time)
                # signal.signal(signal.SIGALRM, timeout_handler)
                # signal.alarm(20)  # 20 second timeout for entire OpenML check

                try:
                    openml_insights = self.openml_agent.get_best_practices(structure_info, domain)

                    if openml_insights.get("similar_datasets"):
                        print(f"  ✅ Found {len(openml_insights['similar_datasets'])} similar datasets on OpenML")
                        result["openml_insights"] = openml_insights
                    else:
                        print("  ℹ️ No similar datasets found on OpenML (continuing without OpenML insights)")
                finally:
                    pass  # signal.alarm(0)  # Disable alarm

            except TimeoutError:
                print("  ⏱️ OpenML check timed out (continuing without OpenML insights)")
                openml_insights = {"error": "timeout", "openml_available": False}
            except Exception as e:
                print(f"  ⚠️ OpenML check failed: {e}")
                print("  Continuing without OpenML insights...")
                openml_insights = {"error": str(e), "openml_available": False}

        # Run FeatureEngineerAgent with OpenML insights
        feature_agent = FeatureEngineerAgent(self.llm)
        feature_result = feature_agent.engineer_features(path, target_variable, model=model, provider=provider)

        # Enhance feature recommendations with OpenML insights
        if openml_insights.get("common_preprocessing_steps"):
            openml_preprocessing = openml_insights["common_preprocessing_steps"]
            if openml_preprocessing:
                feature_result["openml_preprocessing_recommendations"] = openml_preprocessing

        result["results"]["feature_engineering"] = feature_result
        result["summary"]["feature_engineering_complete"] = True

        # Run modeling agent with OpenML recommendations
        modeling_agent = CompSciModelingExpertAgent(self.llm)

        # Enhance constraints with OpenML insights
        enhanced_constraints = constraints or {}
        if openml_insights.get("recommended_models"):
            enhanced_constraints["openml_models"] = openml_insights["recommended_models"]
        if openml_insights.get("common_preprocessing_steps"):
            enhanced_constraints["openml_preprocessing"] = openml_insights["common_preprocessing_steps"]

        modeling_result = modeling_agent.recommend_models_and_generate_code(
            path, problem_type, target_variable, enhanced_constraints, model=model, provider=provider
        )

        # Add OpenML model recommendations if available
        if openml_insights.get("recommended_models"):
            modeling_result["openml_model_recommendations"] = openml_insights["recommended_models"]

        result["results"]["modeling"] = modeling_result
        result["summary"]["modeling_complete"] = True

        if modeling_result.get("code_file_path"):
            result["summary"]["code_generated"] = True
            result["summary"]["code_location"] = modeling_result["code_file_path"]
        else:
            result["summary"]["code_generated"] = False

        # Run DomainExpertAgent only if domain is provided
        if domain:
            try:
                domain_agent = DomainExpertAgent(self.llm)
                result["results"]["domain_analysis"] = domain_agent.analyze_data(path, domain, model=model, provider=provider)
                result["summary"]["domain_analysis_complete"] = True
            except Exception as e:
                result["summary"]["domain_analysis_error"] = str(e)

        result["summary"]["executed_agents"] = len(result["results"])

        # Add OpenML summary
        if openml_insights and not openml_insights.get("error"):
            result["summary"]["openml_similar_datasets"] = len(openml_insights.get("similar_datasets", []))
            result["summary"]["openml_models_recommended"] = len(openml_insights.get("recommended_models", []))

        return result
