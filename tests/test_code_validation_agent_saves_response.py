import json
import tempfile
from pathlib import Path
import pandas as pd

from code_validation_agent import CodeValidationAgent


class DummyLLM:
    def __init__(self, responses_dir: Path):
        self.save_responses = True
        self.responses_dir = responses_dir
        self.saved_files = []

    def _save_response(self, prompt, response, model, max_tokens, worker):
        self.responses_dir.mkdir(parents=True, exist_ok=True)
        filepath = self.responses_dir / f"{worker}_response.json"
        payload = {
            "worker": worker,
            "prompt": prompt,
            "response": response,
            "model": model,
            "max_tokens": max_tokens,
        }
        filepath.write_text(json.dumps(payload), encoding="utf-8")
        self.saved_files.append(str(filepath))


def test_validate_code_saves_response_payload(tmp_path):
    responses_dir = tmp_path / "responses"
    llm = DummyLLM(responses_dir)
    agent = CodeValidationAgent(llm)

    data_path = tmp_path / "sample.csv"
    pd.DataFrame({"feature": [1, 2], "target": [0, 1]}).to_csv(data_path, index=False)

    code_path = tmp_path / "model.py"
    code_path.write_text(
        "import pandas as pd\n" "score = 0.75\n",
        encoding="utf-8",
    )

    result = agent.validate_code(str(code_path), str(data_path))

    assert result["executed"] is True
    assert result["score"] == 0.75
    assert len(llm.saved_files) == 1
    saved_path = Path(llm.saved_files[0])
    assert saved_path.exists()
    payload = json.loads(saved_path.read_text(encoding="utf-8"))
    assert payload["worker"] == "code_validator"
