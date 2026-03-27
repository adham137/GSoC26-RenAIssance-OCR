"""
src/model_engine/adapter_downloader.py
=======================================
Adapter Download Module with Hugging Face Hub Integration.

This module provides functionality to download LoRA adapters from Hugging Face Hub
with progress tracking and authentication support.

Usage
-----
    downloader = AdapterDownloader()
    downloader.download(
        repo_id="username/adapter-repo",
        local_dir="models/my_adapter",
        token="hf_xxx",
        progress_callback=my_callback
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class DownloadProgress:
    """Track download progress for UI feedback."""

    repo_id: str
    total_files: int = 0
    downloaded_files: int = 0
    total_bytes: int = 0
    downloaded_bytes: int = 0
    current_file: str = ""
    status: str = "pending"  # pending, downloading, complete, error
    error_message: str = ""

    @property
    def percentage(self) -> float:
        """Return download percentage (0-100)."""
        if self.total_bytes == 0:
            return 0.0
        return (self.downloaded_bytes / self.total_bytes) * 100

    @property
    def is_complete(self) -> bool:
        """Return True if download is complete."""
        return self.status == "complete"

    @property
    def has_error(self) -> bool:
        """Return True if download encountered an error."""
        return self.status == "error"


class AdapterDownloader:
    """
    Download LoRA adapters from Hugging Face Hub with progress tracking.

    Parameters
    ----------
    models_dir : str | Path
        Base directory where adapters will be stored.
        Default: "models" in project root.
    """

    def __init__(self, models_dir: str | Path | None = None) -> None:
        if models_dir is None:
            # Default to project root / models
            self.models_dir = Path(__file__).parent.parent.parent / "models"
        else:
            self.models_dir = Path(models_dir)

        self.models_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Adapter models directory: {self.models_dir}")

    def download(
        self,
        repo_id: str,
        local_dir_name: str | None = None,
        token: str | None = None,
        progress_callback: Optional[Callable[[DownloadProgress], None]] = None,
        revision: str = "main",
    ) -> Path:
        """
        Download a LoRA adapter from Hugging Face Hub.

        Parameters
        ----------
        repo_id : str
            Hugging Face repository ID (e.g., "username/my-adapter").
        local_dir_name : str, optional
            Name for the local directory. If None, uses repo name.
        token : str, optional
            Hugging Face authentication token. Required for private repos.
        progress_callback : callable, optional
            Function to call with DownloadProgress updates.
        revision : str
            Branch/tag/commit to download from. Default: "main".

        Returns
        -------
        Path
            Path to the downloaded adapter directory.

        Raises
        ------
        RuntimeError
            If download fails or adapter is invalid.
        """
        from huggingface_hub import HfApi, hf_hub_download, list_repo_files

        progress = DownloadProgress(repo_id=repo_id)

        try:
            # Initialize progress
            progress.status = "downloading"
            if progress_callback:
                progress_callback(progress)

            # Determine local directory name
            if local_dir_name is None:
                local_dir_name = repo_id.split("/")[-1]

            local_path = self.models_dir / local_dir_name
            local_path.mkdir(parents=True, exist_ok=True)

            logger.info(f"Downloading adapter from '{repo_id}' to '{local_path}'")

            # Get list of files in the repo
            if token:
                api = HfApi(token=token)
            else:
                api = HfApi()

            # List files to get total size
            try:
                files_info = api.list_repo_files(
                    repo_id=repo_id, revision=revision, token=token
                )
                # Filter for adapter-related files
                adapter_files = [
                    f
                    for f in files_info
                    if f.endswith((".bin", ".safetensors", ".json", ".txt"))
                ]
                progress.total_files = len(adapter_files)
                logger.info(f"Found {progress.total_files} adapter files to download")
            except Exception as e:
                logger.warning(f"Could not list repo files: {e}")
                # Continue with download anyway
                adapter_files = None

            # Download files with progress tracking
            if adapter_files:
                for file_path in adapter_files:
                    try:
                        # Download file to local directory
                        downloaded_file = hf_hub_download(
                            repo_id=repo_id,
                            filename=file_path,
                            revision=revision,
                            token=token,
                            local_dir=str(local_path),
                            local_dir_use_symlinks=False,
                        )
                        # Track downloaded bytes
                        downloaded_size = (
                            Path(downloaded_file).stat().st_size
                            if Path(downloaded_file).exists()
                            else 0
                        )
                        progress.downloaded_bytes += downloaded_size
                        progress.downloaded_files += 1
                        progress.current_file = file_path

                        if progress_callback:
                            progress_callback(progress)

                    except Exception as e:
                        logger.warning(f"Failed to download {file_path}: {e}")
                        continue
            else:
                # Fallback: use snapshot_download for simple download
                from huggingface_hub import snapshot_download

                # Create a wrapper to track progress
                downloaded = [0]

                def _track_progress(filename: str, _):
                    progress.current_file = filename
                    downloaded[0] += 1
                    progress.downloaded_files = downloaded[0]
                    if progress_callback:
                        progress_callback(progress)

                snapshot_download(
                    repo_id=repo_id,
                    local_dir=str(local_path),
                    revision=revision,
                    token=token,
                    ignore_patterns=["*.md", "README*"],
                )

            # Validate the downloaded adapter
            progress.status = "validating"
            if progress_callback:
                progress_callback(progress)

            self._validate_adapter(local_path)

            # Mark as complete
            progress.status = "complete"
            progress.current_file = ""
            if progress_callback:
                progress_callback(progress)

            logger.info(f"Adapter download complete: {local_path}")
            return local_path

        except Exception as e:
            progress.status = "error"
            progress.error_message = str(e)
            if progress_callback:
                progress_callback(progress)

            logger.error(f"Download failed: {e}")
            raise RuntimeError(f"Failed to download adapter: {e}") from e

    def _validate_adapter(self, adapter_path: Path) -> None:
        """
        Validate that the downloaded directory contains a valid adapter.

        Parameters
        ----------
        adapter_path : Path
            Path to the adapter directory.

        Raises
        ------
        RuntimeError
            If the adapter is invalid or missing required files.
        """
        required_files = ["adapter_config.json"]
        optional_files = ["adapter_model.safetensors", "adapter_model.bin"]

        missing_required = []
        for file in required_files:
            if not (adapter_path / file).exists():
                missing_required.append(file)

        if missing_required:
            raise RuntimeError(
                f"Invalid adapter: missing required files: {', '.join(missing_required)}"
            )

        # Check for at least one model weights file
        has_weights = any((adapter_path / f).exists() for f in optional_files)

        if not has_weights:
            logger.warning(
                "No adapter weights found (adapter_model.safetensors or adapter_model.bin). "
                "This may be a config-only adapter."
            )

        logger.info(f"Adapter validation passed: {adapter_path}")

    def list_available_adapters(self) -> list[Path]:
        """
        List all adapters available in the models directory.

        Returns
        -------
        list[Path]
            List of paths to valid adapter directories.
        """
        adapters = []
        if not self.models_dir.exists():
            return adapters

        for item in self.models_dir.iterdir():
            if item.is_dir() and (item / "adapter_config.json").exists():
                adapters.append(item)

        return sorted(adapters)
