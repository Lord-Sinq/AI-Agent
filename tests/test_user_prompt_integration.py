import json
from pathlib import Path

import pandas as pd

from domain_agent import DomainExpertAgent
from feature_agent import FeatureEngineerAgent
from modeling_agent import ModelingAgent


class DummyLLM:
    def __init__(self):
        self.prompts = []
        self.save_responses = False

    def generate(self, prompt, model=None, provider=None, max_tokens=4000, worker=None):
        self.prompts.append(
            {
                "prompt": prompt,
                "model": model,
                "provider": provider,
                "max_tokens": max_tokens,
                "worker": worker,
            }
        )
        if worker == "Feature":
            return {"text": json.dumps({"features": ["age"], "scale": [], "encode": {}, "drop": []})}
        if worker == "Modeling":
            return {
                "text": json.dumps(
                    {"problem": "classification", "target": "target", "models": ["RandomForest"], "code": "import pandas as pd"}
                )
            }
        return {
            "text": json.dumps(
                {
                    "domain_insights": "insight",
                    "key_metrics": ["metric"],
                    "data_quality_issues": ["issue"],
                    "business_questions": ["question"],
                    "relevant_context": "context",
                    "data_limitations": ["limit"],
                }
            )
        }

    def _save_response(self, prompt, response, model, max_tokens, worker):
        return None


def test_domain_agent_includes_user_context_in_prompt(tmp_path):
    llm = DummyLLM()
    agent = DomainExpertAgent(llm)
    data_path = tmp_path / "sample.csv"
    pd.DataFrame({"age": [20, 30], "target": [0, 1]}).to_csv(data_path, index=False)

    agent.analyze(path=str(data_path), domain="healthcare", user="Focus on churn risk", model="mock")

    assert llm.prompts, "Expected the agent to call the LLM"
    prompt = llm.prompts[0]["prompt"]
    assert "USER CONTEXT" in prompt
    assert "Focus on churn risk" in prompt


def test_feature_and_modeling_agents_receive_user_context(tmp_path):
    llm = DummyLLM()
    feature_agent = FeatureEngineerAgent(llm)
    modeling_agent = ModelingAgent(llm)

    data_path = tmp_path / "sample.csv"
    pd.DataFrame({"age": [20, 30], "target": [0, 1]}).to_csv(data_path, index=False)

    feature_agent.analyze(path=str(data_path), target="target", user="Use interpretable features", model="mock")
    modeling_agent.generate(
        path=str(data_path),
        feature_info={"features": ["age"], "scale": [], "encode": {}, "drop": []},
        problem_type="classification",
        target="target",
        user="Prioritize explainability",
        model="mock",
    )

    feature_prompt = llm.prompts[0]["prompt"]
    modeling_prompt = llm.prompts[1]["prompt"]

    assert "USER CONTEXT" in feature_prompt
    assert "Use interpretable features" in feature_prompt
    assert "USER CONTEXT" in modeling_prompt
    assert "Prioritize explainability" in modeling_prompt


def test_modeling_agent_includes_hyperparameters_in_prompt(tmp_path):
    llm = DummyLLM()
    modeling_agent = ModelingAgent(llm)
    data_path = tmp_path / "sample.csv"
    pd.DataFrame({"age": [20, 30], "target": [0, 1]}).to_csv(data_path, index=False)

    modeling_agent.generate(
        path=str(data_path),
        feature_info={"features": ["age"], "scale": [], "encode": {}, "drop": []},
        problem_type="classification",
        target="target",
        hyperparameters={"RandomForestClassifier": {"n_estimators": 25, "random_state": 7}},
    )

    prompt = llm.prompts[0]["prompt"]
    assert '"n_estimators": 25' in prompt
    assert '"random_state": 7' in prompt
