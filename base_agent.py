"""Base agent utilities and file helpers."""

import csv
import io
import json
import mimetypes
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from llms import LLMManager


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

            for col in df.select_dtypes(include=['object']).columns:
                try:
                    if df[col].iloc[0] and isinstance(df[col].iloc[0], bytes):
                        df[col] = df[col].str.decode('utf-8')
                except (IndexError, AttributeError, UnicodeDecodeError):
                    pass

            info = {"rows": len(df), "columns": len(df.columns), "headers": list(df.columns)}
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
        if filename.endswith(".arff") or mimetype == "text/x-arff":
            return {"rows": 0, "columns": 0, "format": "arff", "needs_full_read": True}

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

        if '<think>' in text:
            think_end = text.find('</think>')
            if think_end != -1:
                text = text[think_end + 8:]

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
