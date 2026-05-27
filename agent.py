import csv
import io
import json
import mimetypes
from typing import Any, Dict, List, Optional
from pathlib import Path
from llms import LLMManager


class Agent:
    """Base agent for file and data processing."""

    def __init__(self, llm_manager: LLMManager):
        self.llm = llm_manager

    @staticmethod
    def detect_mimetype(filename: str, content: bytes) -> str:
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
        """Extract JSON from text, handling truncated responses."""
        if text is None:
            return None

        # Try to find complete JSON first
        start = text.find("{")
        if start == -1:
            return None

        # Try to find the end of JSON (matching braces)
        brace_count = 0
        end = -1
        for i in range(start, len(text)):
            if text[i] == "{":
                brace_count += 1
            elif text[i] == "}":
                brace_count -= 1
                if brace_count == 0:
                    end = i
                    break

        if end == -1:
            # JSON is incomplete, try to extract what we can
            # Look for valid JSON up to the last complete field
            json_str = text[start:]
            # Try to complete common truncated patterns
            if json_str.rstrip().endswith('"'):
                json_str += "}"
            elif json_str.rstrip().endswith(","):
                json_str = json_str.rstrip()[:-1] + "}"
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
        try:
            if "." in value:
                return float(value)
            return int(value)
        except Exception:
            return value.lower() if isinstance(value, str) else value

    @staticmethod
    def _sort_csv_text(text: str, sort_keys: List[str], sort_order: str) -> str:
        reader = csv.DictReader(io.StringIO(text))
        rows = list(reader)
        if not rows or sort_keys is None:
            return text
        sort_order = sort_order.lower() if sort_order else "asc"

        def key_func(row: Dict[str, str]) -> Any:
            key = [Agent._coerce_value(row.get(k, "")) for k in sort_keys]
            return key

        rows.sort(key=key_func, reverse=sort_order == "desc")
        output = io.StringIO()
        fieldnames = reader.fieldnames
        if not fieldnames:
            return text
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        return output.getvalue()

    @staticmethod
    def _sort_text_lines(text: str, sort_method: str, sort_order: str) -> str:
        lines = [l for l in text.splitlines() if l.strip()]
        sort_order = sort_order.lower() if sort_order else "asc"
        if sort_method == "numeric":
            lines.sort(
                key=lambda l: Agent._coerce_value(l.strip()),
                reverse=sort_order == "desc",
            )
        else:
            lines.sort(key=lambda l: l.lower(), reverse=sort_order == "desc")
        return "\n".join(lines)


class DataAgent(Agent):
    """Agent that organizes data/files into a better structure using LLM guidance."""

    def _build_organize_prompt(self, filename: str, sample: str, mimetype: str) -> str:
        if mimetype == "text/csv" or filename.lower().endswith(".csv"):
            return (
                "You are a data organization expert. The following CSV file contains tabular records. "
                "Analyze the columns and rows, then recommend the best sort strategy and explain why it is the most useful. "
                "Return JSON with fields: sort_keys (list of column names), sort_order (asc or desc), recommendation, and rationale. "
                f"CSV sample:\n{sample}"
            )
        return (
            "You are a data organization expert. The following text file contains data. "
            "Analyze the structure and recommend the most appropriate organization strategy. "
            "Return JSON with fields: sort_method (numeric or lexicographic), sort_order (asc or desc), recommendation, and rationale. "
            f"Text sample:\n{sample}"
        )

    def organize_file(self, path: str, model: Optional[str] = None, provider: Optional[str] = None) -> dict:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {path}")

        content = p.read_bytes()
        mt = self.detect_mimetype(p.name, content)
        meta = {"filename": p.name, "content_type": mt, "size": len(content)}
        text = self._prepare_text(content, mt)

        if mt == "text/csv" or p.suffix.lower() == ".csv":
            sample = "\n".join(text.splitlines()[:20])
            prompt = self._build_organize_prompt(p.name, sample, mt)
            llm_resp = self.llm.generate(prompt, model=model, provider=provider)
            analysis_text = llm_resp.get("text")
            plan = self._extract_json(analysis_text) or {}
            sorted_preview = self._sort_csv_text(sample, plan.get("sort_keys", []), plan.get("sort_order", "asc"))
            return {
                "meta": meta,
                "analysis": analysis_text,
                "plan": plan,
                "sorted_preview": sorted_preview,
                "raw": llm_resp.get("raw", llm_resp),
            }

        sample = "\n".join(text.splitlines()[:50])
        prompt = self._build_organize_prompt(p.name, sample, mt)
        llm_resp = self.llm.generate(prompt, model=model, provider=provider)
        analysis_text = llm_resp.get("text")
        plan = self._extract_json(analysis_text) or {}
        sorted_preview = self._sort_text_lines(
            sample,
            plan.get("sort_method", "lexicographic"),
            plan.get("sort_order", "asc"),
        )
        return {
            "meta": meta,
            "analysis": analysis_text,
            "plan": plan,
            "sorted_preview": sorted_preview,
            "raw": llm_resp.get("raw", llm_resp),
        }


class VerifierAgent(Agent):
    """Agent that verifies whether data is organized in the best possible way."""

    def _build_verification_prompt(self, filename: str, sample: str, mimetype: str) -> str:
        if mimetype == "text/csv" or filename.lower().endswith(".csv"):
            return (
                "You are a data quality expert. The following CSV file is already organized in some order. "
                "Review the organization and determine whether it is the best way to sort the data. "
                "If it is not, recommend a better sort order and explain why. "
                "Return JSON with fields: valid (true/false), recommendation, why, and suggested_sort_keys. "
                f"CSV sample:\n{sample}"
            )
        return (
            "You are a data quality expert. The following text file contains data organized in a current order. "
            "Review whether the chosen organization is the best for clarity or analysis. "
            "If not, recommend a better organization and explain why. "
            "Return JSON with fields: valid (true/false), recommendation, why, and suggested_sort_method. "
            f"Text sample:\n{sample}"
        )

    def verify_organization(self, path: str, model: Optional[str] = None, provider: Optional[str] = None) -> dict:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {path}")

        content = p.read_bytes()
        mt = self.detect_mimetype(p.name, content)
        meta = {"filename": p.name, "content_type": mt, "size": len(content)}
        text = self._prepare_text(content, mt)

        sample = "\n".join(text.splitlines()[:50])
        prompt = self._build_verification_prompt(p.name, sample, mt)
        llm_resp = self.llm.generate(prompt, model=model, provider=provider)
        analysis_text = llm_resp.get("text")
        verdict = self._extract_json(analysis_text) or {}
        return {
            "meta": meta,
            "analysis": analysis_text,
            "verdict": verdict,
            "raw": llm_resp.get("raw", llm_resp),
        }
