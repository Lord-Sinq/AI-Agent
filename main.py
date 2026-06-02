"""
AI-Agent CLI Module - Data Science Pipeline with Code Generation
"""

import argparse
import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from llms import LLMManager
from agent import Manager


def list_available_files(data_dir: str = "data") -> list:
    """List all available files in the data directory."""
    p = Path(data_dir)
    if not p.exists():
        return []
    return sorted([f.name for f in p.iterdir() if f.is_file() and f.name != ".gitkeep" and not f.name.startswith(".")])


def select_file_interactive(data_dir: str = "data") -> str:
    """Prompt user to select a file from the data directory."""
    files = list_available_files(data_dir)
    if not files:
        raise FileNotFoundError(f"No files found in '{data_dir}' directory")

    print(f"\nAvailable files in '{data_dir}':")
    for i, f in enumerate(files, 1):
        print(f"  {i}. {f}")

    while True:
        try:
            choice = input(f"\nSelect a file (1-{len(files)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(files):
                return str(Path(data_dir) / files[idx])
            else:
                print(f"Invalid choice. Please enter a number between 1 and {len(files)}")
        except ValueError:
            print("Invalid input. Please enter a number.")


def prompt_save_responses() -> bool:
    """
    Prompt the user to decide whether to save LLM responses.

    Returns:
        bool: True if user wants to save responses, False otherwise
    """
    print("\n" + "=" * 60)
    print("LLM Response Saving Option")
    print("=" * 60)
    print("Saving LLM responses can help with debugging and analyzing model outputs.")
    print("Each response will be saved as a JSON file in the 'responses' directory.")
    print()

    while True:
        choice = input("Do you want to save LLM responses? (yes/no): ").strip().lower()
        if choice in ["yes", "y"]:
            return True
        elif choice in ["no", "n"]:
            return False
        else:
            print("Please enter 'yes' or 'no'")


def build_parser():
    """Build and configure the command-line argument parser."""
    p = argparse.ArgumentParser(
        description="AI-Agent CLI — Data Science Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with interactive prompts
  python main.py

  # Specify file and domain
  python main.py --file data/employees.csv --domain hr

  # Disable response saving
  python main.py --no-save-responses

  # Enable response saving
  python main.py --save-responses

  # Skip all prompts (for automation)
  python main.py --quiet --no-save-responses

  # List available deployments
  python main.py --list-deployments
        """,
    )
    p.add_argument("--file", "-f", default=None, help="Path to the file to process")
    p.add_argument("--model", "-m", default=None, help="Model name to use")
    p.add_argument("--domain", "-d", default=None, help="Domain context (e.g., retail, healthcare, HR)")
    p.add_argument("--task", "-t", default="Analyze this data and provide ML insights with code", help="Task description")
    p.add_argument("--target-variable", default=None, help="Target variable for prediction")
    p.add_argument("--problem-type", choices=["classification", "regression", "clustering"], default=None, help="Type of ML problem")
    p.add_argument("--list-deployments", action="store_true", help="List available Azure OpenAI deployments")
    p.add_argument("--save-responses", action="store_true", help="Save LLM responses to files")
    p.add_argument("--no-save-responses", action="store_true", help="Disable saving LLM responses")
    p.add_argument("--quiet", "-q", action="store_true", help="Suppress interactive prompts (use defaults)")
    p.add_argument("--no-openml", action="store_true", help="Disable OpenML recommendations")
    p.add_argument("--download-similar", action="store_true", help="Download similar datasets from OpenML")
    return p


def main():
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args()

    # Handle response saving preference
    save_responses = None

    # Check if environment variable is set
    env_save = os.getenv("SAVE_RESPONSES", "").lower()

    if args.save_responses:
        save_responses = True
        if not args.quiet:
            print("✅ Response saving enabled via command line flag")
    elif args.no_save_responses:
        save_responses = False
        if not args.quiet:
            print("❌ Response saving disabled via command line flag")
    elif env_save in ["true", "false"]:
        save_responses = env_save == "true"
        if not args.quiet:
            print(f"📝 Response saving from environment variable: {save_responses}")
    elif args.quiet:
        # In quiet mode, default to not saving responses
        save_responses = False
        if not args.quiet:
            print("❌ Response saving disabled (quiet mode)")
    else:
        # No preference set, prompt the user
        save_responses = prompt_save_responses()
        if save_responses:
            print("✅ Response saving enabled")
        else:
            print("❌ Response saving disabled")

    # Set the environment variable for the LLMManager
    os.environ["SAVE_RESPONSES"] = str(save_responses).lower()

    try:
        llm = LLMManager()
    except Exception as e:
        print(f"❌ Failed to initialize LLM Manager: {e}")
        return 1

    if args.list_deployments:
        try:
            deployments = llm.list_azure_deployments()
            print("\n✅ Available Azure models/deployments:")
            for item in deployments.get("items", []):
                print(f"  • {item['id']}")
            return 0
        except Exception as e:
            print(f"❌ Error listing deployments: {e}")
            return 1

    try:
        file_path = args.file or select_file_interactive()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return 1

    print(f"\n📄 Processing: {file_path}")
    if args.domain:
        print(f"📂 Domain: {args.domain}")
    if args.target_variable:
        print(f"🎯 Target: {args.target_variable}")
    if args.problem_type:
        print(f"📊 Problem: {args.problem_type}")
    print("=" * 60)

    try:
        manager = Manager(llm)

        result = manager.orchestrate_pipeline(
            path=file_path,
            task=args.task,
            domain=args.domain,
            target_variable=args.target_variable,
            problem_type=args.problem_type,
            model=args.model,
            use_openml_recommendations=not args.no_openml,  # Use OpenML unless disabled
        )

        print("\n✅ Pipeline completed!")
        print(f"  • Feature engineering: {'✓' if result['summary'].get('feature_engineering_complete') else '✗'}")
        print(f"  • Modeling & code: {'✓' if result['summary'].get('modeling_complete') else '✗'}")

        if result["summary"].get("code_generated"):
            print(f"  • Code saved to: {result['summary']['code_location']}")

        if result.get("domain_used"):
            print(f"  • Domain analysis: {'✓' if result['summary'].get('domain_analysis_complete') else '✗'}")

        print("\n📊 Results:\n")

        # Clean output for display
        display_result = {
            "file": result["file_path"],
            "task": result["task"],
            "feature_insights": result["results"]["feature_engineering"].get("recommendations", {}),
            "modeling": {
                "inferred_problem_type": result["results"]["modeling"].get("inferred_problem_type"),
                "inferred_target": result["results"]["modeling"].get("inferred_target"),
                "recommended_models": result["results"]["modeling"]["recommendations"]["recommended_models"],
                "code_generated": result["summary"]["code_generated"],
                "code_path": result["summary"].get("code_location"),
            },
        }

        if "domain_analysis" in result["results"]:
            display_result["domain_insights"] = result["results"]["domain_analysis"].get("analysis", {})

        print(json.dumps(display_result, indent=2, default=str))

        # If code was generated, print a preview
        if result["summary"].get("code_generated"):
            code = result["results"]["modeling"].get("generated_code", "")
            if code:
                print("\n" + "=" * 60)
                print("📝 Generated Code Preview (first 1000 chars):")
                print("=" * 60)
                print(code[:1000])
                if len(code) > 1000:
                    print("\n... (truncated, see full file)")

        # Show where responses were saved if applicable
        if save_responses and hasattr(llm, "responses_dir"):
            print("\n" + "=" * 60)
            print(f"💾 LLM responses saved to: {llm.responses_dir}")
            print("=" * 60)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
