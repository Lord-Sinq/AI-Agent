import argparse
import json
import os

from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from llms import LLMManager
from agent import DataAgent, VerifierAgent


def list_available_files(data_dir: str = "data") -> list:
    """List all available files in the data directory."""
    p = Path(data_dir)
    if not p.exists():
        return []

    # Filter out .gitkeep and any hidden files
    return sorted(
        [
            f.name
            for f in p.iterdir()
            if f.is_file()
            and f.name != ".gitkeep"  # Exclude git placeholder
            and not f.name.startswith(".")  # Exclude hidden files (optional)
        ]
    )


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


def build_parser():
    p = argparse.ArgumentParser(description="AI-Agent CLI — process and verify data organization with LLM agents")
    p.add_argument(
        "--file",
        "-f",
        default=None,
        help="Path to the file to process (if not provided, you'll be prompted to select)",
    )
    p.add_argument(
        "--model",
        "-m",
        default=None,
        help="Model name to request from the LLM provider",
    )
    p.add_argument(
        "--list-deployments",
        action="store_true",
        help="List available Azure OpenAI deployments",
    )
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    try:
        llm = LLMManager()
    except Exception as e:
        print(f"❌ Failed to initialize LLM Manager: {e}")
        print("\nPlease check your .env file configuration.")
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

    # Select file interactively if not provided
    try:
        file_path = args.file or select_file_interactive()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return 1

    try:
        print(f"\n📄 Processing file: {file_path}")
        print("=" * 60)

        # Run DataAgent (organizer)
        print("\n[1/2] Running Data Organizer Agent...")
        data_agent = DataAgent(llm)
        organize_result = data_agent.organize_file(file_path, model=args.model)
        print("✅ Data Organizer Agent completed")

        # Run VerifierAgent
        print("\n[2/2] Running Verifier Agent...")
        verifier_agent = VerifierAgent(llm)
        verify_result = verifier_agent.verify_organization(file_path, model=args.model)
        print("✅ Verifier Agent completed")

        # Combine results
        combined_output = {
            "file": file_path,
            "organizer": organize_result,
            "verifier": verify_result,
        }

    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        import traceback

        traceback.print_exc()
        return 1

    print("\n" + "=" * 60)
    print("\n📊 Results:\n")
    print(json.dumps(combined_output, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
