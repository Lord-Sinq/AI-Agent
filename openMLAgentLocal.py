"""
OpenML Agent Module - Local Dataset Analysis
Supports CSV and ARFF files via scipy.io.arff
"""

import os
import json
import hashlib
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


class OpenMLAgent:
    def __init__(self, datasets_dir: Optional[str] = None, index_file: str = "dataset_index.json"):
        """
        Initialize the OpenML Agent with local datasets.

        Args:
            datasets_dir (str): Directory containing downloaded OpenML datasets.
                                If None, reads from OPENML_DATASETS_DIR env var or defaults to "openMLdatasets"
            index_file (str): File to store the dataset index
        """
        # Check for scipy availability
        self.has_scipy = self._check_scipy()

        # Determine datasets directory
        if datasets_dir is None:
            datasets_dir = os.getenv("OPENML_DATASETS_DIR", "openMLdatasets")

        self.datasets_dir = Path(datasets_dir)
        self.index_file = self.datasets_dir / index_file
        self.datasets_dir.mkdir(exist_ok=True)

        print(f"[INFO] OpenML Agent using datasets directory: {self.datasets_dir.absolute()}")
        print(f"[INFO] Index file will be saved to: {self.index_file}")
        print(f"[INFO] SciPy available: {self.has_scipy}")

        if not self.has_scipy:
            print("[WARNING] SciPy not installed. ARFF files will be ignored.")
            print("         Install with: pip install scipy")

        # Load or build dataset index
        self.dataset_index = self._load_or_build_index()

    def _check_scipy(self) -> bool:
        """Check if scipy is available for ARFF parsing."""
        try:
            from scipy.io import arff
            return True
        except ImportError:
            return False

    def _load_or_build_index(self) -> Dict:
        """Load existing dataset index or build it from scratch."""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    index = json.load(f)
                print(f"[INFO] Loaded existing dataset index with {len(index)} datasets")
                return index
            except Exception as e:
                print(f"[WARNING] Could not load index file: {e}")

        print("[INFO] Building dataset index from local files...")
        return self._build_index()

    def _read_arff(self, file_path: Path) -> Optional[pd.DataFrame]:
        """
        Read ARFF file using scipy.io.arff.

        Args:
            file_path (Path): Path to ARFF file

        Returns:
            DataFrame or None if failed
        """
        if not self.has_scipy:
            print(f"  [WARNING] Cannot read ARFF (scipy missing): {file_path.name}")
            return None

        try:
            from scipy.io import arff

            # Load ARFF file
            data, meta = arff.loadarff(file_path)
            df = pd.DataFrame(data)

            # Decode bytes to strings for string columns
            for col in df.select_dtypes(include=['object']).columns:
                try:
                    # Check if column contains bytes
                    if df[col].iloc[0] and isinstance(df[col].iloc[0], bytes):
                        df[col] = df[col].str.decode('utf-8')
                except (IndexError, AttributeError, UnicodeDecodeError):
                    pass

            return df

        except Exception as e:
            print(f"  [ERROR] Failed to read ARFF file {file_path.name}: {e}")
            return None

    def _read_csv(self, file_path: Path) -> Optional[pd.DataFrame]:
        """
        Read CSV file using pandas.

        Args:
            file_path (Path): Path to CSV file

        Returns:
            DataFrame or None if failed
        """
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            print(f"  [ERROR] Failed to read CSV file {file_path.name}: {e}")
            return None

    def _analyze_dataset(self, file_path: Path) -> Optional[Dict]:
        """
        Analyze a single dataset to extract its characteristics.

        Args:
            file_path (Path): Path to dataset file

        Returns:
            Dictionary with dataset metadata or None if failed
        """
        try:
            # Read based on file extension
            if file_path.suffix.lower() == '.arff':
                df = self._read_arff(file_path)
            else:
                df = self._read_csv(file_path)

            if df is None or df.empty:
                return None

            rows, cols = df.shape

            # Extract dataset ID and name from filename
            # Format: {dataset_id}_{dataset_name}.{csv|arff}
            stem = file_path.stem
            parts = stem.split('_', 1)
            dataset_id = parts[0] if parts else "unknown"
            dataset_name = parts[1] if len(parts) > 1 else stem

            # Detect data types
            data_types = {}
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    if df[col].dtype == 'int64':
                        data_types[col] = "integer"
                    else:
                        data_types[col] = "float"
                elif pd.api.types.is_datetime64_dtype(df[col]):
                    data_types[col] = "datetime"
                else:
                    data_types[col] = "categorical"

            # Calculate missing values
            missing_percentage = {
                col: (df[col].isnull().sum() / rows) * 100
                for col in df.columns
            }

            # Create file hash for change detection
            file_hash = hashlib.md5(file_path.read_bytes()).hexdigest()

            return {
                "id": dataset_id,
                "name": dataset_name,
                "file_path": str(file_path),
                "file_format": file_path.suffix.lower(),
                "file_hash": file_hash,
                "rows": int(rows),
                "columns": int(cols),
                "headers": list(df.columns),
                "data_types": data_types,
                "missing_percentage": missing_percentage,
                "last_analyzed": datetime.now().isoformat()
            }

        except Exception as e:
            print(f"  [ERROR] Failed to analyze {file_path.name}: {e}")
            return None

    def _build_index(self) -> Dict:
        """
        Scan the datasets directory and build an index of all datasets.

        Returns:
            Dictionary mapping dataset IDs to their metadata
        """
        index = {}

        # Find all CSV and ARFF files
        csv_files = list(self.datasets_dir.glob("*.csv"))
        arff_files = list(self.datasets_dir.glob("*.arff"))
        all_files = csv_files + arff_files

        if not all_files:
            print(f"[INFO] No CSV or ARFF files found in {self.datasets_dir}")
            return index

        print(f"[INFO] Found {len(all_files)} files to index")
        if csv_files:
            print(f"       - {len(csv_files)} CSV files")
        if arff_files:
            print(f"       - {len(arff_files)} ARFF files")

        for file_path in all_files:
            dataset_info = self._analyze_dataset(file_path)

            if dataset_info:
                index[dataset_info["id"]] = dataset_info
                print(f"  [OK] {dataset_info['name']} - {dataset_info['rows']} rows, "
                      f"{dataset_info['columns']} cols [{dataset_info['file_format']}]")

        # Save the index
        try:
            with open(self.index_file, 'w') as f:
                json.dump(index, f, indent=2, default=str)
            print(f"[INFO] Index saved to {self.index_file}")
        except Exception as e:
            print(f"[WARNING] Could not save index file: {e}")

        return index

    def find_similar_datasets(self, structure_info: Dict[str, Any], limit: int = 5) -> List[Dict]:
        """
        Find similar datasets based on structural characteristics.

        Args:
            structure_info (dict): Structure information from the user's dataset
            limit (int): Maximum number of similar datasets to return

        Returns:
            List of similar datasets with similarity scores
        """
        if not self.dataset_index:
            return []

        user_rows = structure_info.get("rows", 0)
        user_cols = structure_info.get("columns", 0)
        user_data_types = structure_info.get("data_types", {})

        similarities = []

        for dataset_id, dataset_info in self.dataset_index.items():
            # Row similarity
            row_similarity = 1.0
            if user_rows > 0 and dataset_info["rows"] > 0:
                row_ratio = min(user_rows, dataset_info["rows"]) / max(user_rows, dataset_info["rows"])
                row_similarity = row_ratio

            # Column similarity
            col_similarity = 1.0
            if user_cols > 0 and dataset_info["columns"] > 0:
                col_ratio = min(user_cols, dataset_info["columns"]) / max(user_cols, dataset_info["columns"])
                col_similarity = col_ratio

            # Data type similarity
            type_similarity = 0.5
            if user_data_types and dataset_info.get("data_types"):
                user_types = set(user_data_types.values())
                dataset_types = set(dataset_info["data_types"].values())
                if user_types and dataset_types:
                    intersection = len(user_types.intersection(dataset_types))
                    union = len(user_types.union(dataset_types))
                    type_similarity = intersection / union if union > 0 else 0

            # Weighted average
            total_similarity = (row_similarity * 0.3 + col_similarity * 0.4 + type_similarity * 0.3)

            similarities.append({
                "dataset": dataset_info,
                "similarity_score": round(total_similarity, 3),
                "row_similarity": round(row_similarity, 3),
                "col_similarity": round(col_similarity, 3),
                "type_similarity": round(type_similarity, 3)
            })

        # Sort by similarity score descending
        similarities.sort(key=lambda x: x["similarity_score"], reverse=True)

        return similarities[:limit]

    def get_best_practices(self, structure_info: Dict[str, Any], domain: Optional[str] = None) -> Dict:
        """
        Get best practices from similar local datasets.

        Args:
            structure_info (dict): Structure information from the user's dataset
            domain (str, optional): Domain context for filtering

        Returns:
            Dictionary with best practices recommendations
        """
        similar_datasets = self.find_similar_datasets(structure_info, limit=10)

        best_practices = {
            "similar_datasets": [
                {
                    "id": s["dataset"]["id"],
                    "name": s["dataset"]["name"],
                    "rows": s["dataset"]["rows"],
                    "columns": s["dataset"]["columns"],
                    "format": s["dataset"].get("file_format", "unknown"),
                    "similarity": s["similarity_score"]
                }
                for s in similar_datasets
            ],
            "common_preprocessing_steps": [],
            "recommended_models": [],
            "dataset_statistics": {
                "avg_rows": 0,
                "avg_columns": 0,
                "data_type_distribution": {}
            },
            "local_datasets_available": len(self.dataset_index) > 0
        }

        if not similar_datasets:
            return best_practices

        # Aggregate statistics from similar datasets
        total_rows = 0
        total_cols = 0
        data_type_counts = {}

        for sim in similar_datasets:
            dataset = sim["dataset"]
            total_rows += dataset["rows"]
            total_cols += dataset["columns"]

            for col, dtype in dataset.get("data_types", {}).items():
                data_type_counts[dtype] = data_type_counts.get(dtype, 0) + 1

        num_datasets = len(similar_datasets)
        best_practices["dataset_statistics"]["avg_rows"] = total_rows // num_datasets
        best_practices["dataset_statistics"]["avg_columns"] = total_cols // num_datasets
        best_practices["dataset_statistics"]["data_type_distribution"] = data_type_counts

        # Find common column patterns
        common_columns = {}
        for sim in similar_datasets:
            for header in sim["dataset"].get("headers", []):
                common_columns[header.lower()] = common_columns.get(header.lower(), 0) + 1

        threshold = num_datasets * 0.6

        # Check for ID columns
        if "id" in common_columns or "identifier" in common_columns:
            best_practices["common_preprocessing_steps"].append({
                "technique": "Remove identifier columns",
                "reason": "Similar datasets often drop ID columns before modeling",
                "priority": "high"
            })

        # Check for date columns
        if any(col in common_columns for col in ["date", "timestamp", "created_at", "datetime"]):
            best_practices["common_preprocessing_steps"].append({
                "technique": "Extract date features",
                "reason": "Date columns are common and often need feature extraction",
                "priority": "medium"
            })

        # Check for missing values
        has_missing = any(
            any(v > 0 for v in sim["dataset"].get("missing_percentage", {}).values())
            for sim in similar_datasets
        )
        if has_missing:
            best_practices["common_preprocessing_steps"].append({
                "technique": "Handle missing values",
                "reason": "Similar datasets have missing values that need treatment",
                "priority": "high"
            })

        # Model recommendations based on dataset size
        avg_rows = best_practices["dataset_statistics"]["avg_rows"]

        if avg_rows < 1000:
            best_practices["recommended_models"] = [
                {"model": "Random Forest", "source": "Small datasets (<1000 rows)", "confidence": "high"},
                {"model": "XGBoost", "source": "Good for small to medium datasets", "confidence": "medium"},
                {"model": "Logistic Regression", "source": "Good baseline for small data", "confidence": "medium"}
            ]
        elif avg_rows < 10000:
            best_practices["recommended_models"] = [
                {"model": "Gradient Boosting", "source": "Medium datasets (1k-10k rows)", "confidence": "high"},
                {"model": "Random Forest", "source": "Ensemble methods work well", "confidence": "high"},
                {"model": "Neural Networks", "source": "Can capture complex patterns", "confidence": "medium"}
            ]
        else:
            best_practices["recommended_models"] = [
                {"model": "Deep Learning", "source": "Large datasets (>10k rows)", "confidence": "high"},
                {"model": "XGBoost", "source": "Scales well to large data", "confidence": "high"},
                {"model": "Gradient Boosting", "source": "Effective for large datasets", "confidence": "medium"}
            ]

        return best_practices

    def refresh_index(self):
        """Refresh the dataset index by rescanning all files."""
        print("[INFO] Refreshing dataset index...")
        self.dataset_index = self._build_index()
        print(f"[INFO] Index refreshed with {len(self.dataset_index)} datasets")

    def list_all_datasets(self) -> List[Dict]:
        """List all available local datasets."""
        return [
            {
                "id": info["id"],
                "name": info["name"],
                "rows": info["rows"],
                "columns": info["columns"],
                "format": info.get("file_format", "unknown"),
                "file_path": info["file_path"]
            }
            for info in self.dataset_index.values()
        ]