"""
src/prompt_manager/prompt_registry.py
======================================
Prompt Management Centre (System Design §2.2).

Design Principles
-----------------
* **Zero prompt text in Python files.**  All prompt strings live in
  ``prompts/*.txt`` files.  This module only *loads*, *versions*, and
  *injects variables* into those templates.
* Templates use ``{variable_name}`` placeholders compatible with Python's
  ``str.format_map()``.
* Version tags are embedded as a comment on the first line of each ``.txt``
  file: ``# version: 1.3.0``

Prompt File Naming Convention
------------------------------
``prompts/<prompt_key>.txt``

Expected keys (see ``prompts/`` directory):
    * ``initial_extraction``     — zero-shot first-pass OCR prompt
    * ``capability_reflection``  — asks model to diagnose errors & plan
    * ``memory_reflection``      — feeds history + asks for new reflection
    * ``guided_refinement``      — drives the final answer generation
    * ``capability_filter``      — instructs model to remove infeasible actions
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Sentinel for missing required template variables
_MISSING = object()


class PromptRegistry:
    """
    Centralised store for all prompt templates used across the pipeline.

    Loads templates lazily from disk on first access and caches them in
    memory for subsequent calls.

    Parameters
    ----------
    prompts_dir : str | Path
        Directory containing ``*.txt`` prompt template files.
        Defaults to ``prompts/``.

    Examples
    --------
    >>> registry = PromptRegistry("prompts/")
    >>> prompt = registry.render(
    ...     "initial_extraction",
    ...     image_description="faded ink on vellum",
    ... )
    """

    def __init__(self, prompts_dir: str | Path = "prompts/") -> None:
        self._prompts_dir : Path = Path(prompts_dir)
        self._cache       : dict[str, str] = {}
        self._versions    : dict[str, str] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(self, prompt_key: str, **variables: str) -> str:
        """
        Load (or retrieve from cache), inject variables, and return a
        fully-rendered prompt string.

        Parameters
        ----------
        prompt_key : str
            Basename of the ``.txt`` file without extension.
        **variables : str
            Key-value pairs matching ``{placeholder}`` tokens in the template.

        Returns
        -------
        str
            The rendered prompt ready to be passed to the VLM.

        Raises
        ------
        FileNotFoundError
            If ``<prompts_dir>/<prompt_key>.txt`` does not exist.
        KeyError
            If a required ``{placeholder}`` is missing from ``variables``.

        Notes
        -----
        Variable injection uses :meth:`str.format_map` with a
        ``_SafeMap`` fallback so that *extra* variables are silently
        ignored rather than raising ``KeyError``.
        """
        template = self._load_template(prompt_key)
        rendered = self._inject_variables(template, variables)
        logger.debug("Rendered prompt '%s' (%d chars)", prompt_key, len(rendered))
        return rendered

    def get_version(self, prompt_key: str) -> str:
        """
        Return the version string parsed from the first comment line of
        the template file, e.g. ``"1.3.0"``.

        Parameters
        ----------
        prompt_key : str
            Template identifier.

        Returns
        -------
        str
            Version string, or ``"unknown"`` if no version comment is found.
        """
        if prompt_key not in self._versions:
            self._load_template(prompt_key)  # populates self._versions as a side-effect
        return self._versions.get(prompt_key, "unknown")

    def list_available(self) -> list[str]:
        """
        Return the keys of all ``.txt`` files present in ``prompts_dir``.

        Returns
        -------
        list[str]
            Sorted list of prompt keys (filenames without extension).
        """
        return sorted(p.stem for p in self._prompts_dir.glob("*.txt"))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_template(self, prompt_key: str) -> str:
        """
        Read a ``.txt`` template from disk, parse its version comment,
        strip comment lines, and cache the result.

        Parameters
        ----------
        prompt_key : str
            Template identifier.

        Returns
        -------
        str
            Raw template string with comment lines removed.

        Raises
        ------
        FileNotFoundError
            If the template file does not exist.
        """
        if prompt_key in self._cache:
            return self._cache[prompt_key]

        template_path = self._prompts_dir / f"{prompt_key}.txt"
        if not template_path.exists():
            raise FileNotFoundError(
                f"Prompt template not found: '{template_path}'. "
                f"Available keys: {self.list_available()}"
            )

        raw = template_path.read_text(encoding="utf-8")

        # Parse optional version comment: "# version: X.Y.Z"
        version_match = re.match(r"#\s*version:\s*(\S+)", raw.splitlines()[0])
        self._versions[prompt_key] = version_match.group(1) if version_match else "unknown"

        # Strip comment lines
        lines    = [ln for ln in raw.splitlines() if not ln.startswith("#")]
        template = "\n".join(lines).strip()

        self._cache[prompt_key] = template
        logger.info("Loaded prompt template '%s' (version %s)", prompt_key, self._versions[prompt_key])
        return template

    @staticmethod
    def _inject_variables(template: str, variables: dict[str, str]) -> str:
        """
        Substitute ``{placeholders}`` in *template* with values from
        *variables*, ignoring any extra keys.

        Parameters
        ----------
        template : str
            The raw template string containing ``{placeholder}`` tokens.
        variables : dict[str, str]
            Mapping of placeholder names to their runtime values.

        Returns
        -------
        str
            Fully rendered prompt string.
        """
        # SafeMap: unknown keys are left as-is rather than raising KeyError
        class _SafeMap(dict):
            def __missing__(self, key: str) -> str:
                return f"{{{key}}}"

        return template.format_map(_SafeMap(variables))
