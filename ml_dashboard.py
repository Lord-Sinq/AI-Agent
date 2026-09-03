"""ML Pipeline Dashboard - Textual TUI for AI-Agent"""

import sys
import os
import time
import json
import re
from pathlib import Path
from textual.app import App, ComposeResult
from textual.containers import Container, Vertical, Horizontal, ScrollableContainer
from textual.widgets import Header, Footer, Static, Button, RichLog, RadioSet, RadioButton, Input, Checkbox
from textual import work
from textual.reactive import reactive

# Import your existing modules
from llms import LLMManager
from manager import Manager


def sanitize_id(name: str) -> str:
    """Convert a string to a valid Textual ID."""
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    if sanitized and sanitized[0].isdigit():
        sanitized = "_" + sanitized
    return sanitized


class MLDashboard(App):
    """Textual TUI for the AI-Agent ML Pipeline"""

    CSS_PATH = "style.tcss"

    # Reactive properties (Textual-specific)
    is_running = reactive(False)

    def __init__(self):
        super().__init__()
        self.llm = None
        self.manager = None
        self.current_file = None
        self.results = {}
        self.start_time = None
        self._running_lock = False

        # Configuration flags (matches your CLI args)
        self.config = {
            "task": "Analyze data and provide ML code",
            "domain": None,
            "target": None,
            "problem_type": None,
            "use_openml": True,
            "use_hyperparameters": True,
            "model": None,
            "quiet": False,
            "save_responses": False,
            "list_deployments": False,
        }

        # Available datasets
        self.datasets = []
        self.load_dataset_list()

    def load_dataset_list(self):
        """Scan data directory for available datasets."""
        data_dir = Path("data")
        if data_dir.exists():
            self.datasets = [f for f in data_dir.iterdir() if f.suffix in [".csv", ".arff"]]
        else:
            self.datasets = []

        if self.datasets and not self.current_file:
            self.current_file = str(self.datasets[0])

    def truncate_name(self, name: str, max_len: int = 30) -> str:
        """Truncate a filename to fit in a button."""
        name = Path(name).stem  # Remove extension
        if len(name) > max_len:
            return name[: max_len - 3] + "..."
        return name

    def compose(self) -> ComposeResult:
        """Create the UI layout."""
        yield Header()

        with Container(id="main-container"):
            with Horizontal():
                # ===== SIDEBAR =====
                with Vertical(id="sidebar"):
                    yield Static("ML Pipeline", classes="title")

                    # === DATASET SELECTION ===
                    with Container(classes="dataset-box"):
                        yield Static("Dataset:", classes="section-label")

                        with ScrollableContainer(id="dataset-list"):
                            if self.datasets:
                                for dataset in self.datasets:
                                    is_selected = self.current_file == str(dataset)
                                    variant = "success" if is_selected else "default"
                                    safe_id = sanitize_id(dataset.name)
                                    name = self.truncate_name(dataset.name, 30)
                                    yield Button(name, id=f"ds-{safe_id}", variant=variant)
                            else:
                                yield Static("No datasets found")

                    # === MAIN RUN BUTTON ===
                    yield Button("RUN FULL PIPELINE", id="run-btn", variant="success")

                    # === CONFIGURATION ===
                    with Container(classes="config-box"):
                        yield Static("Configuration:", classes="section-label")

                        yield Static("Problem Type:")
                        yield RadioSet(
                            RadioButton("Auto Detect", id="prob-auto", value=True),
                            RadioButton("Classification", id="prob-class"),
                            RadioButton("Regression", id="prob-reg"),
                            id="problem-type",
                        )

                        yield Input(placeholder="Target column (optional)", id="target-input")
                        yield Input(placeholder="Domain (optional)", id="domain-input")
                        yield Input(placeholder="Specific model (optional)", id="model-input")

                    # === FLAGS ===
                    with Container(classes="flags-box"):
                        yield Static("Runtime Flags:", classes="section-label")
                        yield Checkbox("Use OpenML", id="openml-check", value=True)
                        yield Checkbox("Use Hyperparameters", id="hyperparameters-check", value=True)
                        yield Checkbox("Save LLM Responses", id="save-resp-check", value=False)
                        yield Checkbox("Quiet Mode", id="quiet-check", value=False)
                        yield Checkbox("List Deployments", id="list-deploy-check", value=False)

                    # === DEBUG STEPS ===
                    yield Static("Debug Steps:", classes="section-label")
                    with Horizontal():
                        yield Button("Features", id="feat-btn", variant="default")
                        yield Button("Generate", id="gen-btn", variant="default")
                        yield Button("Validate", id="val-btn", variant="default")
                        yield Button("Results", id="results-btn", variant="default")

                    # === QUIT ===
                    yield Button("Quit", id="quit-btn", variant="error")

                # ===== MAIN CONTENT AREA =====
                with Vertical(id="content-area"):
                    yield Static("Status: Ready - Select a dataset and click RUN", id="status")
                    yield RichLog(id="log-area", wrap=True, highlight=True)
                    yield Static("Results will appear here after running", id="result-area")

        yield Footer()

    # ============= EVENT HANDLERS =============

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id

        if button_id.startswith("ds-"):
            safe_id = button_id[3:]
            for dataset in self.datasets:
                if sanitize_id(dataset.name) == safe_id:
                    self.current_file = str(dataset)
                    for btn in self.query(Button):
                        if btn.id and btn.id.startswith("ds-"):
                            btn.variant = "success" if btn.id == button_id else "default"
                    log = self.query_one("#log-area")
                    log.write(f"[green]Selected dataset: {dataset.name}[/]")
                    return

        if button_id == "quit-btn":
            self.exit()
        elif button_id == "run-btn":
            self.run_pipeline()
        elif button_id == "feat-btn":
            self.run_step("feature")
        elif button_id == "gen-btn":
            self.run_step("generate")
        elif button_id == "val-btn":
            self.run_step("validate")
        elif button_id == "results-btn":
            self.show_results()

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        """Handle radio button changes."""
        if event.radio_set.id == "problem-type":
            selected = event.radio_set.pressed_button.id
            if selected == "prob-auto":
                self.config["problem_type"] = None
            elif selected == "prob-class":
                self.config["problem_type"] = "classification"
            elif selected == "prob-reg":
                self.config["problem_type"] = "regression"

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        """Handle checkbox changes."""
        checkbox_id = event.checkbox.id

        if checkbox_id == "openml-check":
            self.config["use_openml"] = event.value
        elif checkbox_id == "hyperparameters-check":
            self.config["use_hyperparameters"] = event.value
        elif checkbox_id == "save-resp-check":
            self.config["save_responses"] = event.value
            os.environ["SAVE_RESPONSES"] = "true" if event.value else "false"
        elif checkbox_id == "quiet-check":
            self.config["quiet"] = event.value
        elif checkbox_id == "list-deploy-check":
            self.config["list_deployments"] = event.value

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submissions."""
        if event.input.id == "target-input":
            self.config["target"] = event.value if event.value else None
        elif event.input.id == "domain-input":
            self.config["domain"] = event.value if event.value else None
        elif event.input.id == "model-input":
            self.config["model"] = event.value if event.value else None

    # ============= PIPELINE EXECUTION =============

    def run_pipeline(self):
        """Start the full pipeline."""
        log = self.query_one("#log-area")
        status = self.query_one("#status")

        if self._running_lock:
            log.write("[yellow]Pipeline is already running![/]")
            return

        if not self.current_file:
            log.write("[red]Please select a dataset first![/]")
            status.update("No dataset selected")
            return

        try:
            log.write("[dim]Initializing LLM Manager...[/]")
            self.llm = LLMManager()
            self.manager = Manager(self.llm)
            log.write("[green]LLM Manager initialized[/]")
        except Exception as e:
            log.write(f"[red]Failed to initialize LLM: {e}[/]")
            status.update("LLM initialization failed")
            return

        self.results = {}
        log.clear()
        self.start_time = time.time()

        status.update("Running Pipeline...")
        log.write("[bold]========================================[/]")
        log.write("[bold]          PIPELINE START                 [/]")
        log.write("[bold]========================================[/]")
        log.write(f"[dim]Dataset: {Path(self.current_file).name}[/]")
        log.write(f"[dim]Config: {self.config}[/]")

        self._running_lock = True
        self._run_pipeline_async()

    @work
    async def _run_pipeline_async(self):
        """Background task for the full pipeline."""
        log = self.query_one("#log-area")
        status = self.query_one("#status")

        try:
            log.write("\n[bold blue]Running full pipeline...[/]")

            if self.config.get("list_deployments", False):
                log.write("[bold]Listing available deployments...[/]")
                try:
                    deployments = self.llm.list_azure_deployments()
                    for item in deployments.get("items", []):
                        log.write(f"  {item['id']}")
                except Exception as e:
                    log.write(f"[red]Failed to list deployments: {e}[/]")
                status.update("Deployments listed")
                self._running_lock = False
                return

            result = self.manager.process(
                path=self.current_file,
                task=self.config["task"],
                domain=self.config["domain"],
                target=self.config["target"],
                problem_type=self.config["problem_type"],
                use_openml=self.config["use_openml"],
                use_hyperparameters=self.config["use_hyperparameters"],
                model=self.config["model"],
            )

            self.results = result

            log.write("\n[bold green]Pipeline Complete![/]")

            if "summary" in result:
                summary = result["summary"]
                log.write(f"\n[bold]Summary:[/]")
                log.write(f"  Features: {summary.get('features_count', 'N/A')}")
                log.write(f"  Problem Type: {summary.get('problem_type', 'N/A')}")
                log.write(f"  Target: {summary.get('target', 'N/A')}")
                log.write(f"  Code Generated: {summary.get('code_generated', False)}")
                models = summary.get("models", [])
                if models:
                    log.write(f"  Models: {', '.join(models)}")
                if summary.get("validation_passed"):
                    log.write(f"[green]  Validation Passed[/]")
                    if summary.get("improvement") is not None:
                        log.write(f"  Improvement: {summary['improvement']:+.4f}")
                else:
                    log.write(f"[yellow]  Validation Failed[/]")

            if "pipeline" in result and "modeling" in result["pipeline"]:
                code_path = result["pipeline"]["modeling"].get("code_path")
                if code_path:
                    log.write(f"\n[bold]Code saved to: {code_path}[/]")

            elapsed = time.time() - self.start_time
            log.write(f"\n[dim]Runtime: {elapsed:.2f} seconds[/]")

            log.write("\n[bold green]========================================[/]")
            log.write("[bold green]          PIPELINE COMPLETE!             [/]")
            log.write("[bold green]========================================[/]")

            self.show_results_summary()
            status.update(f"Complete! {elapsed:.1f}s")

        except Exception as e:
            log.write(f"\n[red]Pipeline failed: {e}[/]")
            import traceback

            log.write(f"[dim]{traceback.format_exc()}[/]")
            status.update("Pipeline Failed")

        finally:
            self._running_lock = False

    def run_step(self, step: str):
        """Run an individual step."""
        log = self.query_one("#log-area")
        status = self.query_one("#status")

        if self._running_lock:
            log.write("[yellow]Pipeline is already running![/]")
            return

        if not self.current_file:
            log.write("[red]Please select a dataset first![/]")
            status.update("No dataset selected")
            return

        if not self.manager:
            try:
                self.llm = LLMManager()
                self.manager = Manager(self.llm)
            except Exception as e:
                log.write(f"[red]Failed to initialize: {e}[/]")
                return

        log.write(f"\n[bold blue]Running step: {step}[/]")
        status.update(f"Running {step}...")
        self._running_lock = True
        self._run_step_async(step)

    @work
    async def _run_step_async(self, step: str):
        """Run individual step in background."""
        log = self.query_one("#log-area")
        status = self.query_one("#status")

        try:
            result = self.manager.process(
                path=self.current_file,
                task=self.config["task"],
                domain=self.config["domain"],
                target=self.config["target"],
                problem_type=self.config["problem_type"],
                use_openml=self.config["use_openml"],
                use_hyperparameters=self.config["use_hyperparameters"],
                model=self.config["model"],
            )
            self.results = result
            log.write(f"[green]{step} complete![/]")
            status.update(f"{step} complete")
        except Exception as e:
            log.write(f"[red]{step} failed: {e}[/]")
            status.update(f"{step} failed")
        finally:
            self._running_lock = False

    def show_results(self):
        """Show results in the result area."""
        result_area = self.query_one("#result-area")
        log = self.query_one("#log-area")

        if not self.results:
            result_area.update("[yellow]No results yet. Run the pipeline first![/]")
            return

        lines = []
        lines.append("PIPELINE RESULTS")
        lines.append("-" * 30)

        pipeline = self.results.get("pipeline", {})

        features = pipeline.get("feature_specs", {})
        if features:
            lines.append(f"Features: {len(features.get('features', []))}")

        modeling = pipeline.get("modeling", {})
        if modeling:
            lines.append(f"Problem: {modeling.get('problem_type', 'N/A')}")
            lines.append(f"Target: {modeling.get('target', 'N/A')}")
            models = modeling.get("recommended_models", [])
            if models:
                lines.append(f"Models: {', '.join(models)}")
            lines.append(f"Code Generated: {modeling.get('code_generated', False)}")

        validation = pipeline.get("code_validation", {})
        if validation:
            score = validation.get("score")
            lines.append(f"Score: {score:.4f}" if score is not None else "Score: N/A")
            lines.append(f"Issues: {len(validation.get('issues', []))}")

        perf = pipeline.get("validation", {})
        if perf and perf.get("validated"):
            lines.append(f"Improvement: {perf.get('improvement', 0):+.4f}")

        summary = self.results.get("summary", {})
        if summary:
            lines.append(f"Validation: {'PASSED' if summary.get('validation_passed') else 'FAILED'}")

        result_area.update("\n".join(lines))
        log.write("[green]Results displayed in result area[/]")

    def show_results_summary(self):
        """Show a quick summary."""
        self.show_results()

    def on_mount(self) -> None:
        """Called when app is mounted."""
        log = self.query_one("#log-area")
        log.write("[green]ML Pipeline Dashboard ready![/]")
        log.write("[dim]Click a dataset, then RUN[/]")

        if self.datasets:
            self.current_file = str(self.datasets[0])
            safe_id = sanitize_id(self.datasets[0].name)
            for btn in self.query(Button):
                if btn.id == f"ds-{safe_id}":
                    btn.variant = "success"


def main():
    """Entry point for the TUI."""
    try:
        app = MLDashboard()
        app.run()
    except KeyboardInterrupt:
        print("\nExited by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
