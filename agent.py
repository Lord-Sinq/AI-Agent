"""
Agent Module for LLM-Powered Data Organization and Verification

This module provides a set of agent classes that leverage Large Language Models
(LLMs) to analyze, organize, and verify data files. The agents can handle various
file types including CSV, JSON, and text files, providing intelligent sorting
recommendations and quality verification.

Classes:
    Agent: Base class providing common file processing utilities
    DataAgent: Analyzes files and recommends optimal organization strategies
    VerifierAgent: Validates existing file organization and suggests improvements

Features:
    - Automatic MIME type detection
    - CSV column-based sorting recommendations
    - Text line sorting (numeric or lexicographic)
    - JSON extraction from LLM responses
    - File preview generation for large files
"""

import csv
import io
import json
import mimetypes
from typing import Any, Dict, List, Optional
from pathlib import Path
from llms import LLMManager


class Agent:
    """
    Base agent for file and data processing.

    Provides common utility methods for file handling, MIME type detection,
    text preparation, JSON extraction, and sorting operations. This class is
    meant to be inherited by specialized agents like DataAgent and VerifierAgent.
    """

    def __init__(self, llm_manager: LLMManager):
        """
        Initialize an Agent with an LLM manager.

        Args:
            llm_manager (LLMManager): Configured LLM manager instance for
                making API calls to the language model
        """
        self.llm = llm_manager

    @staticmethod
    def detect_mimetype(filename: str, content: bytes) -> str:
        """
        Detect the MIME type of a file based on its name and content.

        Uses mimetypes.guess_type() for extension-based detection, with fallback
        logic for common binary formats like PDF and JSON.

        Args:
            filename (str): Name of the file (used for extension detection)
            content (bytes): Raw file content for binary signature detection

        Returns:
            str: Detected MIME type (e.g., "application/json", "text/csv",
                "application/pdf", or "application/octet-stream" for unknown types
        """
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
        """
        Convert file content to a text representation for LLM processing.

        For text-based files, attempts UTF-8 decoding with fallback to latin-1.
        For binary files, returns a hex representation of the first 1KB.

        Args:
            content (bytes): Raw file content
            mimetype (str): MIME type of the content

        Returns:
            str: Text representation of the content suitable for LLM consumption
        """
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
        """
        Extract JSON from text, handling truncated or malformed responses.

        Attempts to locate complete JSON objects by matching braces. If the JSON
        is incomplete, tries to repair common truncation patterns (missing closing
        braces or trailing commas).

        Args:
            text (Optional[str]): Text that may contain JSON data

        Returns:
            Any: Parsed JSON data (dict, list, etc.) or None if extraction fails
        """
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
        """
        Convert a string value to its appropriate data type.

        Attempts to convert to integer or float if numeric, otherwise returns
        lowercase version of the string for case-insensitive comparison.

        Args:
            value (str): String value to convert

        Returns:
            Any: Converted value (int, float, or lowercase string)
        """
        try:
            if "." in value:
                return float(value)
            return int(value)
        except Exception:
            return value.lower() if isinstance(value, str) else value

    @staticmethod
    def _sort_csv_text(text: str, sort_keys: List[str], sort_order: str) -> str:
        """
        Sort CSV content by specified columns.

        Reads CSV as DictReader, sorts rows based on the provided column keys,
        and returns a new CSV string with sorted data.

        Args:
            text (str): CSV content as string
            sort_keys (List[str]): Column names to sort by (ordered by priority)
            sort_order (str): "asc" for ascending, "desc" for descending

        Returns:
            str: Sorted CSV content as string. Returns original text if sorting fails.
        """
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
        """
        Sort lines of text using specified method.

        Supports numeric sorting (converts lines to numbers) or lexicographic
        (alphabetical) sorting with case-insensitive comparison.

        Args:
            text (str): Text content with lines separated by newlines
            sort_method (str): "numeric" for numerical sorting, otherwise lexicographic
            sort_order (str): "asc" for ascending, "desc" for descending

        Returns:
            str: Sorted text with lines joined by newlines
        """
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
    """
    Agent that organizes data/files into a better structure using LLM guidance.

    This agent analyzes file content (CSV or text) and uses an LLM to recommend
    optimal organization strategies such as sorting by specific columns or using
    numeric/lexicographic ordering.

    The agent:
        1. Reads and prepares file content
        2. Builds a prompt based on file type
        3. Queries the LLM for organization recommendations
        4. Applies the recommendations to generate a sorted preview
        5. Returns analysis, plan, and preview
    """

    def _build_organize_prompt(self, filename: str, sample: str, mimetype: str) -> str:
        """
        Build a prompt for the LLM to analyze and recommend organization.

        Creates different prompts for CSV vs. text files. CSV prompts ask for
        column-based sorting recommendations, while text prompts ask for line-based
        sorting (numeric or lexicographic).

        Args:
            filename (str): Name of the file being analyzed
            sample (str): Sample content from the file (first few lines/rows)
            mimetype (str): MIME type of the file

        Returns:
            str: Formatted prompt for the LLM
        """
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
        """
        Analyze and organize a file using LLM recommendations.

        Reads the specified file, detects its type, generates a sample, queries
        the LLM for organization strategy, applies the strategy to create a sorted
        preview, and returns comprehensive results.

        Args:
            path (str): Path to the file to organize
            model (Optional[str]): Specific LLM model to use (overrides default)
            provider (Optional[str]): LLM provider to use (defaults to Azure OpenAI)

        Returns:
            dict: A dictionary containing:
                - "meta": File metadata (filename, content_type, size)
                - "analysis": Raw LLM response text
                - "plan": Parsed JSON recommendations from LLM
                - "sorted_preview": Preview of sorted content after applying plan
                - "raw": Complete LLM API response

        Raises:
            FileNotFoundError: If the specified file doesn't exist
        """
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
    """
    Agent that verifies whether data is organized in the best possible way.

    This agent analyzes already-organized files and uses an LLM to validate
    whether the current organization is optimal. It can suggest improvements
    if better organization strategies exist.

    The agent:
        1. Reads and prepares file content
        2. Builds a verification prompt based on file type
        3. Queries the LLM to assess the current organization
        4. Returns a verdict (valid/invalid) with recommendations
    """

    def _build_verification_prompt(self, filename: str, sample: str, mimetype: str) -> str:
        """
        Build a prompt for the LLM to verify file organization.

        Creates different prompts for CSV vs. text files. CSV prompts check if
        column-based sorting is optimal, while text prompts check line-based
        organization.

        Args:
            filename (str): Name of the file being verified
            sample (str): Sample content from the file (first few lines/rows)
            mimetype (str): MIME type of the file

        Returns:
            str: Formatted prompt for the LLM
        """
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
        """
        Verify if a file's organization is optimal using LLM assessment.

        Reads the specified file, detects its type, generates a sample, queries
        the LLM to evaluate the current organization, and returns a verdict
        with recommendations for improvement if needed.

        Args:
            path (str): Path to the file to verify
            model (Optional[str]): Specific LLM model to use (overrides default)
            provider (Optional[str]): LLM provider to use (defaults to Azure OpenAI)

        Returns:
            dict: A dictionary containing:
                - "meta": File metadata (filename, content_type, size)
                - "analysis": Raw LLM response text
                - "verdict": Parsed JSON verdict containing:
                    - "valid": Boolean indicating if organization is optimal
                    - "recommendation": Text recommendation
                    - "why": Explanation of the verdict
                    - "suggested_sort_keys" or "suggested_sort_method": Improvement suggestions
                - "raw": Complete LLM API response

        Raises:
            FileNotFoundError: If the specified file doesn't exist
        """
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
