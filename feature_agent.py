"""Feature engineering agent module."""

import json
from typing import Any, Dict, List, Optional

from base_agent import Agent


class FeatureEngineerAgent(Agent):
    """Feature engineering recommendations agent."""

    def analyze(self, path: str, target: Optional[str] = None, user: Optional[str] = None, model: Optional[str] = None) -> dict:
        content, mt, text, structure = self._read_file(path)

        headers = structure.get('headers', [])
        data_types = structure.get('data_types', {})
        rows = structure.get('rows', 0)

        user_context = f"\nUSER CONTEXT: {user}" if user else ""

        prompt = f"""DATA: {path}
            ROWS: {rows}
            COLS: {headers}
            TYPES: {data_types}
            TARGET: {target if target else 'auto'}{user_context}

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

        attempts = 0
        while not self._valid_feature_recommendations(recommendations) and attempts < 2:
            attempts += 1
            print(f"[WARNING] Feature engineer response did not contain required JSON keys. Retry #{attempts}.")
            if isinstance(recommendations, dict) and recommendations.get("error") == "json":
                break

            retry_prompt = f"""DATA: {path}
            ROWS: {rows}
            COLS: {headers}
            TYPES: {data_types}
            TARGET: {target if target else 'auto'}{user_context}

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

            if isinstance(recommendations, dict) and not self._valid_feature_recommendations(recommendations):
                if headers and set(recommendations.keys()).issubset(set(headers)) and all(isinstance(v, str) for v in recommendations.values()):
                    wrapped = {
                        "features": [h for h in headers if h not in recommendations.keys() and h != target][:10],
                        "scale": [h for h in headers if data_types.get(h) in ['numeric', 'float']][:5],
                        "encode": recommendations,
                        "drop": [h for h in headers if 'id' in h.lower() or 'date' in h.lower()],
                        "feature_metadata": {"note": "wrapped encode-only response"}
                    }
                    if self._valid_feature_recommendations(wrapped):
                        recommendations = wrapped
                        break

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

            if headers and set(recommendations.keys()).issubset(set(headers)) and all(isinstance(v, str) for v in recommendations.values()):
                wrapped = {
                    "features": [h for h in headers if h not in recommendations.keys() and h != target][:10],
                    "scale": [h for h in headers if data_types.get(h) in ['numeric', 'float']][:5],
                    "encode": recommendations,
                    "drop": [h for h in headers if 'id' in h.lower() or 'date' in h.lower()],
                    "feature_metadata": {"note": "wrapped encode-only response"}
                }
                if self._valid_feature_recommendations(wrapped):
                    return wrapped

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
            "feature_details": "Fallback default features used because the model response could not be parsed into valid JSON.",
            "feature_reasons": "Fallback generated defaults",
            "feature_notes": "Default feature list because JSON extraction failed.",
            "derived_features": [],
            "feature_metadata": {"note": "default fallback"},
            "raw_feature_response": ""
        }
