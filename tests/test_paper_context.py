import json

from paper_context import PaperContextAgent, format_paper_context


class DummyLLM:
    def generate(self, *args, **kwargs):
        return {"text": "{}"}


def test_paper_context_finds_dataset_mapping(tmp_path):
    papers_dir = tmp_path / "dataPapers"
    papers_dir.mkdir()
    paper_path = papers_dir / "study.pdf"
    paper_path.write_bytes(b"%PDF")
    (papers_dir / "index.json").write_text(json.dumps({"sample.arff": {"paper": "study.pdf"}}), encoding="utf-8")

    agent = PaperContextAgent(DummyLLM(), papers_dir=papers_dir)

    assert agent.find_paper(str(tmp_path / "sample.arff")) == paper_path


def test_paper_context_is_visible_as_reference_evidence():
    formatted = format_paper_context({"paper": "study.pdf", "important_features": ["age"]})

    assert "PAPER EVIDENCE" in formatted
    assert "important_features" in formatted
    assert "dataset columns remain authoritative" in formatted


def test_unavailable_paper_context_is_not_sent_to_agents():
    formatted = format_paper_context({"paper": "study.pdf", "paper_context_available": False, "status": "OCR required"})

    assert formatted == ""
