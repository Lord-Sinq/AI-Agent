"""Domain analysis agent module."""

from pathlib import Path
from typing import Optional

from base_agent import Agent


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

        return {"meta": {"file": Path(path).name, "domain": domain}, "structure": structure, "analysis": analysis
        }
