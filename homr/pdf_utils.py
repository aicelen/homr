import os

import cv2
import numpy as np
import pypdfium2 as pdfium

from homr.autocrop import autocrop
from homr.transformer.configs import root_dir


def render_pdf_to_image(pdf_path: str, dpi: int = 300) -> list[str]:
    "Renders a pdf to a folder and returns its path"
    scale = dpi / 72.0
    pdf = pdfium.PdfDocument(pdf_path)
    assert pdf, f"invalid PDF {pdf_path}"  # noqa: S101
    images = []
    try:
        for i, page in enumerate(pdf):
            ouptut_path = os.path.join(f"{os.path.splitext(pdf_path)[0]}_{i}.png")
            bitmap = page.render(scale=scale)
            rgb = np.array(bitmap.to_pil().convert("RGB"))
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(ouptut_path, autocrop(bgr))
            images.append(ouptut_path)

    finally:
        pdf.close()
    return images
