"""
OpenML Agent Module with Improved Error Handling
"""

import json
import requests
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta


class OpenMLAgent:
    """
    Agent that queries OpenML for dataset insights and best practices.
    Includes retry logic and better timeout handling.
    """

    def __init__(self, cache_dir: str = "openml_cache", datasets_dir: str = "openMLdatasets"):
        """
        Initialize the OpenML Agent.

        Args:
            cache_dir (str): Directory to cache OpenML API responses
            datasets_dir (str): Directory to store downloaded datasets
        """
        self.base_url = "https://www.openml.org/api/v1/json"
        self.cache_dir = Path(cache_dir)
        self.datasets_dir = Path(datasets_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.datasets_dir.mkdir(exist_ok=True)

        # Configuration
        self.timeout = 15  # Reduced timeout to avoid long hangs
        self.max_retries = 2
        self.retry_delay = 2  # seconds

        # Load download history
        self.download_history_file = self.cache_dir / "download_history.json"
        self.download_history = self._load_download_history()

    def _load_download_history(self) -> Dict:
        """Load download history to track what's been downloaded."""
        if self.download_history_file.exists():
            try:
                with open(self.download_history_file, "r") as f:
                    return json.load(f)
            except:
                return {"downloaded_datasets": {}}
        return {"downloaded_datasets": {}}

    def _save_download_history(self):
        """Save download history."""
        with open(self.download_history_file, "w") as f:
            json.dump(self.download_history, f, indent=2)

    def _make_request(self, endpoint: str, params: Optional[Dict] = None, max_retries: int = None) -> dict:
        """
        Make a request to the OpenML API with caching and retry logic.

        Args:
            endpoint (str): API endpoint
            params (dict): Query parameters
            max_retries (int): Override default retry count

        Returns:
            dict: API response or empty dict on failure
        """
        max_retries = max_retries or self.max_retries

        # Create cache key
        cache_key = f"{endpoint}_{json.dumps(params or {}, sort_keys=True)}"
        cache_key = cache_key.replace("/", "_").replace("?", "_")
        cache_file = self.cache_dir / f"{cache_key}.json"

        # Check cache (7 days expiry)
        if cache_file.exists():
            try:
                cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
                if cache_age.days < 7:
                    with open(cache_file, "r") as f:
                        return json.load(f)
            except:
                pass  # Cache corrupted, will fetch fresh

        # Make request with retries
        url = f"{self.base_url}/{endpoint}"

        for attempt in range(max_retries):
            try:
                print(f"  📡 Attempt {attempt + 1}/{max_retries} to query OpenML...")
                response = requests.get(url, params=params, timeout=self.timeout)
                response.raise_for_status()
                data = response.json()

                # Cache response
                with open(cache_file, "w") as f:
                    json.dump(data, f, indent=2)

                return data

            except requests.exceptions.Timeout:
                print(f"  ⏱️ OpenML API timeout (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                else:
                    print(f"  ⚠️ OpenML API unavailable after {max_retries} attempts")
                    return {}

            except requests.exceptions.RequestException as e:
                print(f"  ⚠️ OpenML API error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                else:
                    return {}

        return {}

    def find_similar_datasets(self, structure_info: Dict[str, Any], limit: int = 5, skip_downloaded: bool = True) -> List[Dict]:
        """
        Find similar datasets on OpenML based on structure.
        Returns empty list if API is unavailable.
        """
        similar_datasets = []

        # Extract key features from structure_info
        num_columns = structure_info.get("columns", 0)

        # If no columns info, return empty (can't find similar)
        if num_columns == 0:
            print("  ℹ️ Insufficient structure info to find similar datasets")
            return []

        try:
            # Get list of datasets (limit to 50 to be efficient)
            response = self._make_request("data/list", {"limit": 50})

            if response and "data" in response and "data" in response["data"]:
                datasets = response["data"]["data"]

                for dataset in datasets:
                    # Extract basic info
                    dataset_info = {
                        "id": dataset.get("did"),
                        "name": dataset.get("name"),
                        "version": dataset.get("version"),
                        "status": dataset.get("status"),
                        "format": dataset.get("format"),
                        "tags": dataset.get("tag", "").split(",") if dataset.get("tag") else [],
                    }

                    # Check if dataset has similar number of columns
                    dataset_cols = dataset.get("NumberOfFeatures", 0)
                    if dataset_cols and num_columns:
                        # Accept datasets with similar column count (±50%)
                        col_ratio = abs(dataset_cols - num_columns) / max(num_columns, 1)
                        if col_ratio <= 0.5:
                            # Get more details
                            detailed = self.get_dataset_details(dataset_info["id"])
                            if detailed:
                                dataset_info["detailed"] = detailed
                                similar_datasets.append(dataset_info)

                    if len(similar_datasets) >= limit:
                        break

            else:
                print("  ℹ️ No datasets found on OpenML or API unavailable")

        except Exception as e:
            print(f"  ⚠️ Error finding similar datasets: {e}")

        # Filter out already downloaded datasets if requested
        if skip_downloaded and similar_datasets:
            original_count = len(similar_datasets)
            similar_datasets = [ds for ds in similar_datasets if not self.is_dataset_downloaded(ds["id"])]
            if len(similar_datasets) < original_count:
                print(f"  📦 Skipped {original_count - len(similar_datasets)} already downloaded datasets")

        return similar_datasets

    def get_dataset_details(self, dataset_id: int) -> Dict:
        """Get detailed information about a specific OpenML dataset."""
        if not dataset_id:
            return {}

        response = self._make_request(f"data/{dataset_id}")
        if response and "data_set" in response:
            return response["data_set"]
        return {}

    def is_dataset_downloaded(self, dataset_id: int) -> bool:
        """Check if a specific dataset is already downloaded."""
        if str(dataset_id) in self.download_history["downloaded_datasets"]:
            dataset_info = self.download_history["downloaded_datasets"][str(dataset_id)]
            file_path = Path(dataset_info["path"])
            if file_path.exists():
                return True
            else:
                # File missing, remove from history
                del self.download_history["downloaded_datasets"][str(dataset_id)]
                self._save_download_history()
        return False

    def download_dataset(self, dataset_id: int, force: bool = False) -> Optional[Path]:
        """
        Download a dataset from OpenML for reference.

        Args:
            dataset_id (int): OpenML dataset ID
            force (bool): Force re-download even if already exists

        Returns:
            Path: Path to downloaded file, or None if failed
        """
        # Check if already downloaded
        if not force and self.is_dataset_downloaded(dataset_id):
            dataset_info = self.download_history["downloaded_datasets"][str(dataset_id)]
            print(f"  ✅ Dataset {dataset_id} already downloaded")
            return Path(dataset_info["path"])

        dataset_details = self.get_dataset_details(dataset_id)
        if not dataset_details or "file_id" not in dataset_details:
            return None

        file_id = dataset_details["file_id"]
        dataset_name = dataset_details.get("name", "dataset")

        # Create sanitized filename
        safe_name = "".join(c for c in dataset_name if c.isalnum() or c in (" ", "-", "_")).rstrip()
        filename = f"{dataset_id}_{safe_name}.csv"
        target_path = self.datasets_dir / filename

        # Check again with specific file path
        if not force and target_path.exists():
            print(f"  ✅ Dataset file already exists: {filename}")
            return target_path

        # Download the dataset
        download_url = f"https://www.openml.org/data/v1/download/{file_id}"

        try:
            print(f"  📥 Downloading dataset {dataset_id} ({dataset_name})...")
            response = requests.get(download_url, timeout=30)
            response.raise_for_status()

            # Save the file
            with open(target_path, "wb") as f:
                f.write(response.content)

            # Update download history
            self.download_history["downloaded_datasets"][str(dataset_id)] = {
                "id": dataset_id,
                "name": dataset_name,
                "path": str(target_path),
                "downloaded_at": datetime.now().isoformat(),
                "size_bytes": len(response.content),
            }
            self._save_download_history()

            print(f"  ✅ Downloaded to {target_path}")
            return target_path

        except Exception as e:
            print(f"  ❌ Failed to download: {e}")
            return None

    def get_best_practices(self, structure_info: Dict[str, Any], domain: Optional[str] = None) -> Dict:
        """
        Get best practices for similar datasets from OpenML.
        Returns empty dict if API is unavailable.
        """
        similar_datasets = self.find_similar_datasets(structure_info, limit=3)

        best_practices = {
            "similar_datasets": similar_datasets,
            "common_preprocessing_steps": [],
            "recommended_models": [],
            "evaluation_metrics": [],
            "common_issues": [],
            "openml_available": len(similar_datasets) > 0,
        }

        # If no similar datasets found, return early
        if not similar_datasets:
            best_practices["openml_available"] = False
            return best_practices

        # Analyze similar datasets to find common practices
        for dataset in similar_datasets:
            detailed = dataset.get("detailed", {})

            # Extract preprocessing hints from dataset description
            description = detailed.get("description", "")
            if description:
                preprocessing_keywords = ["normalize", "scale", "encode", "impute", "pca", "standardize"]
                for keyword in preprocessing_keywords:
                    if keyword in description.lower():
                        best_practices["common_preprocessing_steps"].append({"technique": keyword, "source_dataset": dataset["name"]})

            # Get flows used on this dataset (with timeout)
            try:
                flows = self.get_flows_for_dataset(dataset["id"], limit=5)
                for flow in flows:
                    flow_name = flow.get("name", "")
                    model_keywords = ["random forest", "xgboost", "neural network", "svm", "logistic", "gradient boosting"]
                    for model in model_keywords:
                        if model in flow_name.lower():
                            best_practices["recommended_models"].append({"model": model, "source_dataset": dataset["name"]})
            except:
                pass  # Skip flow analysis if it fails

        # Deduplicate
        best_practices["common_preprocessing_steps"] = list(
            {v["technique"]: v for v in best_practices["common_preprocessing_steps"]}.values()
        )
        best_practices["recommended_models"] = list({v["model"]: v for v in best_practices["recommended_models"]}.values())

        return best_practices

    def get_flows_for_dataset(self, dataset_id: int, limit: int = 10) -> List[Dict]:
        """Get machine learning flows (pipelines) used on a dataset."""
        response = self._make_request(f"flow/list", {"data_id": dataset_id, "limit": limit})
        flows = []

        if response and "flows" in response and "flow" in response["flows"]:
            for flow in response["flows"]["flow"][:limit]:
                flows.append(
                    {
                        "id": flow.get("flow_id"),
                        "name": flow.get("name"),
                        "version": flow.get("version"),
                    }
                )

        return flows

    def suggest_preprocessing(self, structure_info: Dict[str, Any]) -> List[Dict]:
        """Suggest preprocessing steps based on similar OpenML datasets."""
        best_practices = self.get_best_practices(structure_info)

        # If OpenML is unavailable, return empty suggestions (agent will use defaults)
        if not best_practices.get("openml_available"):
            return []

        suggestions = []
        data_types = structure_info.get("data_types", {})

        # Add suggestions from similar datasets
        for step in best_practices.get("common_preprocessing_steps", [])[:3]:
            suggestions.append(
                {
                    "step": step["technique"].capitalize(),
                    "reason": f"Used on similar dataset '{step['source_dataset']}'",
                    "priority": "medium",
                    "source": "OpenML",
                }
            )

        return suggestions

    def suggest_models_from_openml(self, structure_info: Dict[str, Any], problem_type: Optional[str] = None) -> List[Dict]:
        """Suggest models based on what worked for similar OpenML datasets."""
        best_practices = self.get_best_practices(structure_info)

        # If OpenML is unavailable, return empty list (agent will use defaults)
        if not best_practices.get("openml_available"):
            return []

        models = []
        for model_info in best_practices.get("recommended_models", [])[:3]:
            models.append(
                {"name": model_info["model"], "source": f"OpenML dataset '{model_info['source_dataset']}'", "confidence": "medium"}
            )

        return models
