"""Pipeline manager orchestrator module."""

from pathlib import Path
from typing import Optional, Dict, Any, List
import datetime
import json

import pandas as pd
from base_agent import Agent
from caafeValidator import CAAFEFeatureValidator
from code_validation_agent import CodeValidationAgent
from domain_agent import DomainExpertAgent
from feature_agent import FeatureEngineerAgent
from openMLAgentLocal import OpenMLAgent
from modeling_agent import ModelingAgent


class Manager(Agent):
    """Main orchestrator with automatic code fixing and improvement."""

    def __init__(self, llm_manager):
        super().__init__(llm_manager)
        self.openml = OpenMLAgent()
        self.domain_agent = DomainExpertAgent(llm_manager)
        self.feature_agent = FeatureEngineerAgent(llm_manager)
        self.modeling_agent = ModelingAgent(llm_manager)
        self.code_validation_agent = CodeValidationAgent(llm_manager)
        self.max_retry_attempts = 3
        self.improvement_history = []

    def process(
        self,
        path: str,
        task: str,
        user: str,
        domain: Optional[str] = None,
        target: Optional[str] = None,
        problem_type: Optional[str] = None,
        use_openml: bool = True,
        model: Optional[str] = None,
        max_retry_attempts: int = 5,
        auto_fix: bool = True,
    ) -> dict:
        """
        Process with automatic code fixing and improvement.

        Args:
            path: Path to data file
            task: Task description
            domain: Domain name for context
            target: Target column name
            problem_type: 'classification' or 'regression'
            use_openml: Whether to use OpenML for context
            model: Specific model to use
            max_retry_attempts: Maximum retry attempts for code fixing
            auto_fix: Whether to automatically fix code errors

        Returns:
            Dictionary with pipeline results including fix attempts
        """
        print(f"\nProcessing: {path}")
        _, _, _, structure = self._read_file(path)

        result = {
            "file": path,
            "task": task,
            "user": user,
            "user_prompt": user,
            "timestamp": datetime.datetime.now().isoformat(),
            "pipeline": {
                "domain_analysis": {},
                "feature_specs": {},
                "modeling": {},
                "code_validation": {},
                "validation": {},
                "code_fixing": {
                    "attempts": 0,
                    "success": False,
                    "history": []
                }
            },
            "summary": {}
        }

        # --- Domain Analysis ---
        domain_analysis = {}
        if domain:
            print(f"\n[Domain Expert] Analyzing domain: {domain}")
            domain_result = self.domain_agent.analyze(
                path=path,
                user=user,
                domain=domain,
                model=model
            )
            domain_analysis = domain_result.get("analysis", {})
            result["pipeline"]["domain_analysis"] = {
                "domain": domain,
                "insights": domain_analysis.get("domain_insights", ""),
                "key_metrics": domain_analysis.get("key_metrics", []),
                "business_questions": domain_analysis.get("business_questions", []),
                "data_quality_issues": domain_analysis.get("data_quality_issues", []),
                "data_limitations": domain_analysis.get("data_limitations", []),
                "relevant_context": domain_analysis.get("relevant_context", "")
            }
            print(f"  Domain insights extracted")
            print(f"  Key metrics identified: {len(domain_analysis.get('key_metrics', []))}")

        # --- OpenML Context ---
        if use_openml and self.openml.dataset_index:
            print("\n[OpenML] Checking similar datasets...")
            similar = self.openml.find_similar_datasets(structure, limit=3)
            if similar:
                result["pipeline"]["domain_analysis"]["openml_context"] = {
                    "similar_count": len(similar),
                    "top_match": similar[0]["dataset"]["name"] if similar else None
                }
                print(f"  Found {len(similar)} similar datasets")

        # --- Feature Engineering ---
        print("\n[Feature Engineer] Designing features...")
        feature_specs = self.feature_agent.analyze(
            path=path,
            target=target,
            user=user,
            model=model
        )

        result["pipeline"]["feature_specs"] = {
            "features": feature_specs.get("features", []),
            "scale": feature_specs.get("scale", []),
            "encode": feature_specs.get("encode", {}),
            "drop": feature_specs.get("drop", []),
            "feature_details": feature_specs.get("feature_details", ""),
            "feature_reasons": feature_specs.get("feature_reasons", ""),
            "feature_notes": feature_specs.get("feature_notes", ""),
            "derived_features": feature_specs.get("derived_features", []),
            "feature_metadata": feature_specs.get("feature_metadata", {})
        }

        print(f"  Designed {len(feature_specs.get('features', []))} features")
        print(f"  Will scale: {len(feature_specs.get('scale', []))} columns")
        print(f"  Will encode: {len(feature_specs.get('encode', {}))} columns")
        print(f"  Will drop: {len(feature_specs.get('drop', []))} columns")

        # --- Code Generation with Auto-Fixing ---
        print("\n[Model] Generating model code with auto-fix capability...")

        code_path = None
        code_generated = False
        validation_passed = False
        attempt = 0
        last_error = None
        fix_history = []
        all_validation_results = []
        code_preview = ""

        # Set max attempts
        max_attempts = max_retry_attempts if auto_fix else 1

        while attempt < max_attempts and not validation_passed:
            attempt += 1
            print(f"\n{'='*60}")
            print(f"Code Generation Attempt {attempt}/{max_attempts}")
            if attempt > 1:
                print(f"  Previous error: {last_error.get('error_summary', 'Unknown error')}")
            print(f"{'='*60}")

            # Generate or regenerate code
            if attempt == 1:
                modeling_result = self.modeling_agent.generate(
                    path=path,
                    feature_info=feature_specs,
                    problem_type=problem_type,
                    target=target,
                    user=user,
                    model=model
                )
            else:
                # Regenerate with error feedback
                modeling_result = self.modeling_agent.regenerate_with_feedback(
                    path=path,
                    feature_info=feature_specs,
                    error=last_error,
                    previous_code_path=code_path,
                    problem_type=problem_type,
                    target=target,
                    user=user,
                    model=model
                )

            code_path = modeling_result.get('code_path')
            code_generated = modeling_result.get('code_generated', False)
            code_preview = modeling_result.get('code_preview', "")

            if not code_generated or not code_path:
                print(f"  ✗ Failed to generate code on attempt {attempt}")
                fix_history.append({
                    "attempt": attempt,
                    "success": False,
                    "error": "Code generation failed",
                    "type": "generation_failure"
                })
                continue

            # Validate the generated code
            print(f"\n[Code Validation] Validating generated code (attempt {attempt})...")
            code_validation_result = self.code_validation_agent.validate_code(
                code_path=str(code_path),
                data_path=path,
                target=target,
                problem_type=problem_type
            )
            all_validation_results.append(code_validation_result)

            # Check validation results
            syntax_ok = code_validation_result.get('syntax_ok', False)
            executed = code_validation_result.get('executed', False)
            issues = code_validation_result.get('issues', [])
            warnings = code_validation_result.get('warnings', [])
            score = code_validation_result.get('score')

            print(f"  Syntax OK: {syntax_ok}")
            print(f"  Executed: {executed}")
            if score is not None:
                print(f"  Score: {score:.4f}")
            if issues:
                print(f"  Issues: {len(issues)}")
                for issue in issues[:3]:
                    print(f"    - {issue}")
                if len(issues) > 3:
                    print(f"    ... and {len(issues) - 3} more")

            # Determine if validation passed
            if syntax_ok and (executed or len(issues) == 0):
                validation_passed = True
                print(f"\n  Validation PASSED on attempt {attempt}!")
                if score is not None:
                    print(f"  Score: {score:.4f}")
            else:
                # Store error for next attempt
                error_summary = issues[0] if issues else "Unknown validation error"
                error_type = "syntax_error" if not syntax_ok else "execution_error" if not executed else "validation_error"

                last_error = {
                    'error_summary': error_summary,
                    'error_type': error_type,
                    'syntax_ok': syntax_ok,
                    'executed': executed,
                    'issues': issues,
                    'warnings': warnings,
                    'attempt': attempt,
                    'code_preview': code_preview[:500] if code_preview else ""
                }

                fix_history.append({
                    "attempt": attempt,
                    "success": False,
                    "error": error_summary,
                    "error_type": error_type,
                    "issues_count": len(issues)
                })

                if auto_fix and attempt < max_attempts:
                    print(f"\n  Validation failed. Will retry with error feedback...")
                    print(f"  Error: {error_summary[:100]}...")
                else:
                    print(f"\n  Validation failed. Maximum attempts reached or auto-fix disabled.")

        # --- Save modeling results ---
        result["pipeline"]["modeling"] = {
            "problem_type": modeling_result.get("problem_type"),
            "target": modeling_result.get("target"),
            "recommended_models": modeling_result.get("recommended_models", []),
            "code_path": code_path,
            "code_generated": code_generated,
            "code_preview": code_preview,
            "final_attempt": attempt,
            "validation_passed": validation_passed
        }

        print(f"\n  Problem type: {modeling_result.get('problem_type', 'unknown')}")
        print(f"  Target: {modeling_result.get('target', 'unknown')}")
        print(f"  Recommended models: {', '.join(modeling_result.get('recommended_models', []))}")
        if code_generated:
            print(f"  Code saved to: {code_path}")

        # --- Store code validation results ---
        if all_validation_results:
            # Use the final validation result for the main result
            final_validation = all_validation_results[-1] if all_validation_results else {}
            result["pipeline"]["code_validation"] = final_validation

            # Also store history
            result["pipeline"]["code_validation"]["history"] = [
                {
                    "attempt": i + 1,
                    "syntax_ok": v.get('syntax_ok', False),
                    "executed": v.get('executed', False),
                    "score": v.get('score'),
                    "issues_count": len(v.get('issues', [])),
                    "warnings_count": len(v.get('warnings', []))
                }
                for i, v in enumerate(all_validation_results)
            ]
        else:
            result["pipeline"]["code_validation"] = {
                "executed": False,
                "issues": ["No validation performed"],
                "notes": ["Code validation skipped"]
            }

        # --- Code fixing history ---
        result["pipeline"]["code_fixing"] = {
            "attempts": attempt,
            "success": validation_passed,
            "auto_fix_enabled": auto_fix,
            "max_attempts": max_attempts,
            "history": fix_history
        }

        # --- CAAFE Performance Validation ---
        print("\n[CAAFE Validator] Validating generated model code performance...")
        validation_result = {
            "validated": False,
            "stage": "post_modeling",
            "metrics": {}
        }

        # Detect target if 'auto'
        detected_target = target
        if target == 'auto' or target is None:
            print("  Auto-detecting target column...")
            # Load data to detect target
            if path.endswith('.arff'):
                from scipy.io import arff
                data, meta = arff.loadarff(path)
                df_temp = pd.DataFrame(data)
                for col in df_temp.select_dtypes(include=['object']).columns:
                    try:
                        if df_temp[col].iloc[0] and isinstance(df_temp[col].iloc[0], bytes):
                            df_temp[col] = df_temp[col].str.decode('utf-8')
                    except (IndexError, AttributeError, UnicodeDecodeError):
                        pass
            else:
                df_temp = pd.read_csv(path)

            # Try common target names
            possible_targets = ['class', 'Class', 'target', 'Target', 'label', 'Label', 'y', 'Y', 'Utility', 'Contraceptive_method_used', 'Response']
            found_targets = [col for col in df_temp.columns if col in possible_targets]
            if found_targets:
                detected_target = found_targets[0]
                print(f"  Detected target: {detected_target}")
            else:
                # Use last column as target
                detected_target = df_temp.columns[-1]
                print(f"  Using last column as target: {detected_target}")

        # Now proceed with validation
        if modeling_result.get('code_generated') and detected_target:
            code_path = modeling_result.get('code_path')
            if code_path and Path(code_path).exists():
                try:
                    print("  Loading data for validation...")
                    if path.endswith('.arff'):
                        try:
                            from scipy.io import arff
                            data, meta = arff.loadarff(path)
                            df = pd.DataFrame(data)
                            for col in df.select_dtypes(include=['object']).columns:
                                try:
                                    if df[col].iloc[0] and isinstance(df[col].iloc[0], bytes):
                                        df[col] = df[col].str.decode('utf-8')
                                except (IndexError, AttributeError, UnicodeDecodeError):
                                    pass
                        except ImportError:
                            print(f"  [ERROR] scipy required for ARFF support. Install with: pip install scipy")
                            raise
                    else:
                        df = pd.read_csv(path)

                    validator = CAAFEFeatureValidator(target=detected_target)
                    print("  Evaluating baseline performance...")
                    validator.evaluate_baseline(df)
                    baseline_score = validator.baseline_score
                    print(f"  Baseline score: {baseline_score:.4f}")

                    print("  Validating model code...")
                    is_valid, metrics = validator.validate_model_code(
                        str(code_path),
                        df
                    )

                    validation_result = {
                        "validated": is_valid,
                        "code_path": str(code_path),
                        "baseline_score": baseline_score,
                        "metrics": metrics,
                        "model_score": metrics.get("score", 0),
                        "improvement": metrics.get("score", 0) - baseline_score if is_valid else 0.0,
                        "issues": metrics.get("issues", []),
                        "warnings": metrics.get("warnings", [])
                    }

                    if is_valid:
                        print(f"  Validation PASSED!")
                        print(f"  Model score: {metrics.get('score', 0):.4f}")
                        print(f"  Improvement over baseline: {validation_result['improvement']:.4f}")
                    else:
                        print(f"  Validation FAILED")
                        if validation_result.get("issues"):
                            print(f"  Issues: {len(validation_result['issues'])}")
                            for issue in validation_result['issues'][:3]:
                                print(f"    - {issue}")

                except Exception as e:
                    validation_result = {
                        "validated": False,
                        "error": str(e),
                        "stage": "validation_error"
                    }
                    print(f"  Validation error: {e}")
            else:
                validation_result = {
                    "validated": False,
                    "note": "Code file not found for validation",
                    "stage": "file_not_found"
                }
                print("  Code file not found for validation")
        else:
            if not modeling_result.get('code_generated'):
                validation_result["note"] = "No code generated for validation"
                print("  No code generated - skipping validation")
            elif not detected_target:
                validation_result["note"] = "No target column specified for validation"
                print("  No target column - skipping validation")

        result["pipeline"]["validation"] = validation_result

        # --- Summary ---
        result["summary"] = {
            "features_count": len(feature_specs.get('features', [])),
            "problem_type": modeling_result.get('problem_type'),
            "target": modeling_result.get('target'),
            "code_generated": code_generated,
            "code_validation_passed": validation_passed,
            "attempts_used": attempt,
            "models": modeling_result.get('recommended_models', []),
            "validation_passed": validation_result.get('validated', False),
            "improvement": validation_result.get('improvement') if validation_result.get('validated') else None,
            "code_fixed": validation_passed and attempt > 1,
            "fix_attempts": attempt - 1 if validation_passed else attempt
        }

        # --- Final Report ---
        print(f"\n{'='*60}")
        print(f" Pipeline Complete!")
        print(f"{'='*60}")
        print(f"  Features designed: {len(feature_specs.get('features', []))}")
        print(f"  Problem type: {modeling_result.get('problem_type', 'unknown')}")
        print(f"  Models: {', '.join(modeling_result.get('recommended_models', []))}")

        if code_generated:
            print(f"  Code generated: {code_path}")
            print(f"  Code validation: {' PASSED' if validation_passed else ' FAILED'}")
            if validation_passed:
                print(f"  Attempts used: {attempt}")
                if attempt > 1:
                    print(f"  Code fixed after {attempt - 1} attempt(s)")

        if validation_result.get('validated'):
            print(f"  Performance validation: PASSED")
            print(f"  Improvement: {validation_result['improvement']:.4f}")
        else:
            print(f"  Performance validation: SKIPPED/FAILED")

        if code_path and code_generated:
            print(f"\n   Run: python {code_path}")
        print(f"{'='*60}")

        return result

    def quick_process(self, path: str, target: Optional[str] = None) -> dict:
        """
        Quick processing without domain analysis or auto-fixing.

        Args:
            path: Path to data file
            target: Target column name

        Returns:
            Dictionary with quick processing results
        """
        _, _, _, structure = self._read_file(path)
        feature_info = self.feature_agent.analyze(path, target)
        modeling_result = self.modeling_agent.generate(path, feature_info, target=target)

        return {
            "file": path,
            "target": modeling_result.get('target'),
            "problem": modeling_result.get('problem_type'),
            "models": modeling_result.get('recommended_models', [])[:2],
            "code": modeling_result.get('code_path')
        }

    def get_improvement_summary(self) -> Dict[str, Any]:
        """
        Get summary of all code improvements made.

        Returns:
            Dictionary with improvement statistics
        """
        return {
            "total_attempts": sum(1 for h in self.improvement_history),
            "successful_fixes": sum(1 for h in self.improvement_history if h.get('success', False)),
            "average_attempts": sum(h.get('attempts', 0) for h in self.improvement_history) / len(self.improvement_history) if self.improvement_history else 0,
            "history": self.improvement_history[-10:]  # Last 10 improvements
        }

    def _read_file(self, path: str) -> tuple:
        """
        Read data file and return metadata.

        Args:
            path: Path to data file

        Returns:
            Tuple of (df, rows, cols, structure)
        """
        if path.endswith('.arff'):
            try:
                from scipy.io import arff
                data, meta = arff.loadarff(path)
                df = pd.DataFrame(data)
                for col in df.select_dtypes(include=['object']).columns:
                    try:
                        if df[col].iloc[0] and isinstance(df[col].iloc[0], bytes):
                            df[col] = df[col].str.decode('utf-8')
                    except (IndexError, AttributeError, UnicodeDecodeError):
                        pass
            except ImportError:
                raise ImportError("scipy required for ARFF support. Install with: pip install scipy")
        else:
            df = pd.read_csv(path)

        rows, cols = df.shape
        structure = {
            'rows': rows,
            'columns': cols,
            'column_names': df.columns.tolist(),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'sample': df.head(3).to_dict('records')
        }

        return df, rows, cols, structure