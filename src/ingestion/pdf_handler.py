"""
src/ingestion/pdf_handler.py
============================
PDF Handling & Ingestion Module (System Design §2.1).

Responsibilities
----------------
* Accept a raw PDF path.
* Split it into individual pages.
* Rasterise each page to a high-resolution PNG.
* Attach document-level metadata to each :class:`~src.models.DocumentPage`.

Implementation Notes
--------------------
* Uses ``PyMuPDF`` (``fitz``) for rasterisation — it preserves fine ink
  detail critical for 17th-century manuscript analysis.
* Default DPI is 300; override via ``dpi`` argument for higher fidelity.
* Output images are written to ``data/output_images/<doc_stem>/`` so they
  persist between runs and can be inspected manually.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.models import DocumentPage

logger = logging.getLogger(__name__)


class PDFHandler:
    """
    Entry-point module for ingesting raw PDF documents.

    Parameters
    ----------
    output_dir : str | Path
        Root directory where rasterised page images are stored.
        Defaults to ``data/output_images``.
    dpi : int
        Resolution used when converting PDF pages to images.
        300 is the recommended minimum for historical manuscripts.

    Examples
    --------
    >>> handler = PDFHandler(output_dir="data/output_images", dpi=300)
    >>> pages = handler.load_pdf("data/raw_pdfs/manuscript_1640.pdf")
    >>> print(pages[0].page_id)
    manuscript_1640_page_001
    """

    def __init__(
        self,
        output_dir: str | Path = "data/output_images",
        dpi: int = 300,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.dpi = dpi
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_pdf(self, pdf_path: str | Path) -> list[DocumentPage]:
        """
        Parse a PDF and return one :class:`~src.models.DocumentPage` per page.

        Parameters
        ----------
        pdf_path : str | Path
            Absolute or relative path to the source PDF.

        Returns
        -------
        list[DocumentPage]
            Ordered list of page objects ready for downstream inference.

        Raises
        ------
        FileNotFoundError
            If ``pdf_path`` does not exist.
        RuntimeError
            If ``PyMuPDF`` fails to open or render the document.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logger.info("Loading PDF: %s", pdf_path)

        doc_metadata = self._extract_document_metadata(pdf_path)
        pages = self._split_and_rasterise(pdf_path, doc_metadata)

        logger.info("Loaded %d pages from '%s'", len(pages), pdf_path.name)
        return pages

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_document_metadata(self, pdf_path: Path) -> dict:
        """
        Extract top-level PDF metadata using PyMuPDF's document info dict.

        Returns a dict with keys such as ``title``, ``author``,
        ``creationDate``, plus ``source_filename`` and ``page_count``
        added by this method.  All values are strings; empty string means
        the field was absent in the PDF.

        Parameters
        ----------
        pdf_path : Path
            Path to the PDF file.

        Returns
        -------
        dict
            Metadata dictionary; safe to pass to DocumentPage.metadata.
        """
        try:
            import fitz  # PyMuPDF
        except ImportError as exc:
            raise ImportError(
                "PyMuPDF is required for PDF ingestion. "
                "Install it with: pip install PyMuPDF"
            ) from exc

        doc = fitz.open(str(pdf_path))
        # keys: title, author, subject, keywords, creator, producer,
        #       creationDate, modDate  — absent fields are empty strings
        metadata = dict(doc.metadata)
        metadata["source_filename"] = pdf_path.name
        metadata["page_count"] = str(doc.page_count)
        doc.close()

        populated = {k: v for k, v in metadata.items() if v}
        logger.debug("PDF metadata populated fields: %s", list(populated.keys()))
        return metadata

    def _split_and_rasterise(
        self,
        pdf_path: Path,
        doc_metadata: dict,
    ) -> list[DocumentPage]:
        """
        Iterate over every PDF page, render it to PNG at ``self.dpi``,
        and construct a :class:`~src.models.DocumentPage` for each.

        Pages whose PNG already exists on disk are not re-rendered
        (simple existence check — delete the output dir to force a
        full re-render).

        Parameters
        ----------
        pdf_path : Path
            Path to the PDF file.
        doc_metadata : dict
            Pre-extracted document-level metadata to attach to each page.

        Returns
        -------
        list[DocumentPage]
            One entry per page; ``image_path`` points to the saved PNG.
        """
        try:
            import fitz  # PyMuPDF
        except ImportError as exc:
            raise ImportError(
                "PyMuPDF is required. Install with: pip install PyMuPDF"
            ) from exc

        stem = pdf_path.stem
        page_dir = self.output_dir / stem
        page_dir.mkdir(parents=True, exist_ok=True)

        # PyMuPDF internal units are 72 DPI — scale up to target DPI
        scale = self.dpi / 72.0
        mat = fitz.Matrix(scale, scale)

        doc = fitz.open(str(pdf_path))
        pages: list[DocumentPage] = []

        try:
            for i, page in enumerate(doc):
                page_num = i + 1
                page_id = f"{stem}_page_{page_num:03d}"
                img_path = page_dir / f"{page_id}.png"

                if img_path.exists():
                    logger.debug("Skipping cached page: %s", img_path.name)
                else:
                    pix = page.get_pixmap(matrix=mat, alpha=False)
                    pix.save(str(img_path))
                    logger.debug(
                        "Rasterised page %d → %s  (%d×%d px)",
                        page_num,
                        img_path.name,
                        pix.width,
                        pix.height,
                    )

                if not self._validate_image(img_path):
                    raise RuntimeError(f"Rasterised image invalid or empty: {img_path}")

                pages.append(
                    DocumentPage(
                        page_id=page_id,
                        image_path=str(img_path.resolve()),
                        metadata={
                            **doc_metadata,
                            "page_number": page_num,
                            "page_width": str(round(page.rect.width, 1)),
                            "page_height": str(round(page.rect.height, 1)),
                            "dpi": str(self.dpi),
                        },
                    )
                )
        finally:
            doc.close()

        return pages

    def _validate_image(self, image_path: Path) -> bool:
        """
        Sanity-check that a rasterised image is readable and non-empty.

        Parameters
        ----------
        image_path : Path
            Path to the PNG/JPEG to validate.

        Returns
        -------
        bool
            ``True`` if the image can be opened and has non-zero dimensions.
        """
        try:
            from PIL import Image

            # Increase decompression bomb limit for high-res manuscript images
            # Default is 178956970 pixels; 300 DPI scans can exceed this
            Image.MAX_IMAGE_PIXELS = None  # Disable limit for validation

            with Image.open(image_path) as img:
                w, h = img.size
                return w > 0 and h > 0
        except Exception as exc:
            logger.warning("Image validation failed for '%s': %s", image_path, exc)
            return False
