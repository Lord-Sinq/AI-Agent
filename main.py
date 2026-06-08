"""
AI-Agent CLI - Data Science Pipeline with Code Generation
"""

import argparse
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from llms import LLMManager
from agent import Manager

load_dotenv()


def get_files(data_dir: str = "data") -> list:
    """Get all CSV files from data directory."""
    path = Path(data_dir)
    if not path.exists():
        return []
    return sorted([f.name for f in path.iterdir() if f.is_file() and not f.name.startswith('.')])


def pick_file(data_dir: str = "data") -> str:
    """Let user pick a file interactively."""
    files = get_files(data_dir)
    if not files:
        raise FileNotFoundError(f"No files found in '{data_dir}'")

    print(f"\nAvailable files in '{data_dir}':")
    for i, f in enumerate(files, 1):
        print(f"  {i}. {f}")

    while True:
        try:
            choice = input(f"\nSelect file (1-{len(files)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(files):
                return str(Path(data_dir) / files[idx])
            print(f"Enter 1-{len(files)}")
        except ValueError:
            print("Enter a number")


def ask_save_responses() -> bool:
    """Ask user if they want to save LLM responses."""
    print("\n" + "=" * 60)
    print("Save LLM responses for debugging?")
    print("(Saved to 'responses/' folder)")
    print("=" * 60)

    while True:
        choice = input("\nSave responses? (yes/no): ").strip().lower()
        if choice in ['yes', 'y']:
            return True
        if choice in ['no', 'n']:
            return False
        print("Enter 'yes' or 'no'")


def parse_args():
    """Parse command line arguments."""
    p = argparse.ArgumentParser(
        description="AI-Agent CLI - Data Science Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                    # Interactive mode
  python main.py -f data/sales.csv -d retail       # Specify file and domain
  python main.py --save-responses                  # Enable response saving
  python main.py --list-deployments                # List Azure models
  python main.py --quiet --no-openml               # Silent mode, no OpenML
        """
    )

    p.add_argument("-f", "--file", help="Path to CSV file")
    p.add_argument("-m", "--model", help="LLM model to use")
    p.add_argument("-d", "--domain", help="Domain (healthcare, retail, etc.)")
    p.add_argument("-t", "--task", default="Analyze data and provide ML code", help="Task description")
    p.add_argument("--target", help="Target variable for prediction")
    p.add_argument("--problem", choices=["classification", "regression", "clustering"], help="ML problem type")
    p.add_argument("--list-deployments", action="store_true", help="List Azure deployments")
    p.add_argument("--save-responses", action="store_true", help="Save LLM responses")
    p.add_argument("--no-save-responses", action="store_true", help="Don't save responses")
    p.add_argument("--quiet", "-q", action="store_true", help="Suppress prompts")
    p.add_argument("--no-openml", action="store_true", help="Disable OpenML")

    return p.parse_args()


def setup_save_responses(args):
    """Configure response saving based on args, env, or user prompt."""
    if args.save_responses:
        os.environ["SAVE_RESPONSES"] = "true"
        if not args.quiet:
            print("Response saving enabled")
        return True

    if args.no_save_responses:
        os.environ["SAVE_RESPONSES"] = "false"
        if not args.quiet:
            print("Response saving disabled")
        return False

    env_save = os.getenv("SAVE_RESPONSES", "").lower()
    if env_save in ['true', 'false']:
        save = env_save == 'true'
        if not args.quiet:
            print(f"Response saving from .env: {save}")
        os.environ["SAVE_RESPONSES"] = str(save).lower()
        return save

    if not args.quiet:
        save = ask_save_responses()
        os.environ["SAVE_RESPONSES"] = str(save).lower()
        return save

    os.environ["SAVE_RESPONSES"] = "false"
    return False


def print_summary(result):
    """Print clean summary of results."""
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    summary = result.get("summary", {})

    if summary.get("features_count") is not None:
        print(f"Features selected: {summary.get('features_count', 0)}")

    if summary.get("problem_type"):
        print(f"Problem type: {summary.get('problem_type', 'unknown')}")

    if summary.get("target"):
        print(f"Target variable: {summary.get('target', 'unknown')}")

    models = summary.get("models", [])
    if models:
        print(f"Recommended models: {', '.join(models[:3])}")

    if summary.get("code_generated"):
        # Get code path from pipeline
        if "pipeline" in result and "modeling" in result["pipeline"]:
            code_path = result["pipeline"]["modeling"].get("code_path")
            if code_path:
                print(f"Code saved: {code_path}")

    if result.get("openml_context"):
        print(f"Found {result['openml_context'].get('similar_count', 0)} similar datasets")
    print("\n" + "=" * 60)

def main():
    """Main entry point."""
    args = parse_args()

    if args.list_deployments:
        try:
            llm = LLMManager()
            deployments = llm.list_azure_deployments()
            print("\nAvailable models:")
            for item in deployments.get("items", []):
                print(f"  {item['id']}")
            return 0
        except Exception as e:
            print(f"Error: {e}")
            return 1

    setup_save_responses(args)

    try:
        llm = LLMManager()
    except Exception as e:
        print(f"Failed to initialize LLM: {e}")
        return 1

    try:
        file_path = args.file or pick_file()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    print(f"\nFile: {file_path}")
    if args.domain:
        print(f"Domain: {args.domain}")
    if args.target:
        print(f"Target: {args.target}")
    if args.problem:
        print(f"Problem: {args.problem}")
    print("=" * 60)

    try:
        manager = Manager(llm)
        result = manager.process(
            path=file_path,
            task=args.task,
            domain=args.domain,
            target=args.target,
            problem_type=args.problem,
            use_openml=not args.no_openml,
            model=args.model
        )

        print("\nPipeline complete!")
        print_summary(result)

        if not args.quiet:
            print("\nDetailed results:")
            clean_result = {
                    "file": result.get("file"),
                    "task": result.get("task"),
                    "timestamp": result.get("timestamp"),
                    "features": result["pipeline"].get("features", {}),
                    "modeling": {
                        "problem_type": result["pipeline"]["modeling"].get("problem_type"),
                        "target": result["pipeline"]["modeling"].get("target"),
                        "models": result["pipeline"]["modeling"].get("recommended_models", []),
                        "code_generated": result["pipeline"]["modeling"].get("code_generated"),
                        "code_path": result["pipeline"]["modeling"].get("code_path")
                    }
                }
            print(json.dumps(clean_result, indent=2))

        code_generated = False
        code_preview = ""

        if "pipeline" in result and "modeling" in result["pipeline"]:
            code_generated = result["pipeline"]["modeling"].get("code_generated", False)
            code_preview = result["pipeline"]["modeling"].get("code_preview", "")
        elif "results" in result and "modeling" in result["results"]:
            code_generated = result["results"]["modeling"].get("code_generated", False)
            code_preview = result["results"]["modeling"].get("generated_code", "")

        if code_generated and code_preview and not args.quiet:
            print("\n" + "=" * 60)
            print("CODE PREVIEW (first 800 chars)")
            print("=" * 60)
            print(code_preview[:800])
            if len(code_preview) > 800:
                print("\n... (see full file for complete code)")

        # Show response save location
        if os.getenv("SAVE_RESPONSES") == "true" and hasattr(llm, 'responses_dir'):
            print(f"\nResponses saved to: {llm.responses_dir}")

    except Exception as e:
        print(f"\n Error: {e}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())