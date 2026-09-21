# make_pdfs.py
# Generates sample policy PDFs (one per entry in mock_docs.py) for the
# RAG demo to ingest.

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.enums import TA_LEFT
from mock_docs import NGO_DOCUMENTS

OUT_DIR = "documents"
os.makedirs(OUT_DIR, exist_ok=True)

styles = getSampleStyleSheet()

letterhead_style = ParagraphStyle(
    "Letterhead", parent=styles["Normal"],
    fontName="Helvetica-Bold", fontSize=9, textColor="#2B4C82",
    spaceAfter=2
)
title_style = ParagraphStyle(
    "DocTitle", parent=styles["Title"],
    fontName="Helvetica-Bold", fontSize=17, alignment=TA_LEFT,
    spaceAfter=4, spaceBefore=6
)
meta_style = ParagraphStyle(
    "Meta", parent=styles["Normal"],
    fontName="Helvetica-Oblique", fontSize=9, textColor="#555555",
    spaceAfter=14
)
body_style = ParagraphStyle(
    "Body", parent=styles["Normal"],
    fontName="Helvetica", fontSize=10.5, leading=16,
    spaceAfter=10
)
footer_style = ParagraphStyle(
    "Footer", parent=styles["Normal"],
    fontName="Helvetica", fontSize=8, textColor="#888888"
)

for i, doc in enumerate(NGO_DOCUMENTS, start=1):
    filename = os.path.join(OUT_DIR, f"{doc['id']}.pdf")
    pdf = SimpleDocTemplate(
        filename, pagesize=letter,
        topMargin=0.9 * inch, bottomMargin=0.8 * inch,
        leftMargin=0.9 * inch, rightMargin=0.9 * inch
    )

    policy_no = f"CHA-POL-{i:03d}"

    elements = [
        Paragraph("CEDAR HEALTH ALLIANCE", letterhead_style),
        Paragraph("Internal Policy Document · Not for External Distribution", meta_style),
        Paragraph(doc["title"], title_style),
        Paragraph(
            f"Category: {doc['category']} &nbsp;|&nbsp; Policy No: {policy_no} &nbsp;|&nbsp; Effective: 01 Jan 2026 &nbsp;|&nbsp; Version: 1.2",
            meta_style
        ),
        HRFlowable(width="100%", thickness=0.75, color="#223049", spaceAfter=14),
        Paragraph(doc["content"], body_style),
        Spacer(1, 0.4 * inch),
        Paragraph(
            f"Cedar Health Alliance — Internal Policy Register &nbsp;·&nbsp; {policy_no} &nbsp;·&nbsp; Page 1 of 1",
            footer_style
        ),
    ]

    pdf.build(elements)
    print(f"Generated {filename}")

print(f"\nDone — {len(NGO_DOCUMENTS)} PDF documents created in '{OUT_DIR}/'")
