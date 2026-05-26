import argparse
import json
from dotenv import load_dotenv

load_dotenv()

from llms import LLMManager
from agent import DataAgent, VerifierAgent


def build_parser():
    p = argparse.ArgumentParser(description="AI-Agent CLI — process and verify data organization with LLM agents")
    p.add_argument("--agent", "-a", choices=["data", "verifier"], required=True, help="Select the agent to run")
    p.add_argument("--file", "-f", required=True, help="Path to the file to process")
    p.add_argument("--task", "-t", choices=["organize", "verify"], default=None, help="Task to perform. 'organize' for data agent, 'verify' for verifier agent")
    p.add_argument("--provider", "-p", choices=["openai", "custom", "echo"], default=None, help="LLM provider to use (default: auto)")
    p.add_argument("--model", "-m", default=None, help="Model name to request from the LLM provider")
    return p


def determine_task(agent_name: str, task: str | None) -> str:
    if task:
        return task
    return "organize" if agent_name == "data" else "verify"


def main():
    parser = build_parser()
    args = parser.parse_args()

    llm = LLMManager()
    task = determine_task(args.agent, args.task)

    try:
        if args.agent == "data":
            if task != "organize":
                raise ValueError("Data agent only supports task 'organize'")
            agent = DataAgent(llm)
            out = agent.organize_file(args.file, model=args.model, provider=args.provider)
        else:
            if task != "verify":
                raise ValueError("Verifier agent only supports task 'verify'")
            agent = VerifierAgent(llm)
            out = agent.verify_organization(args.file, model=args.model, provider=args.provider)
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        return 1

    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
