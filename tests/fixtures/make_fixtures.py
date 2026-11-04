"""Generate the small synthetic PDFs used by the ingestion tests.

    uv run python tests/fixtures/make_fixtures.py

The PDFs are committed, so this only needs re-running when their content changes. They imitate
real CBDT/CBIC formatting (running headers and footers, "Circular No. 3/2025", dated lines,
hyphenation across a line break, a bilingual Hindi block, a blank page) but are SYNTHETIC: the
text is made up and each page says so.
"""

from pathlib import Path

import pymupdf

HERE = Path(__file__).resolve().parent
DISCLAIMER = "SYNTHETIC TEST FIXTURE - not an official document."
HEADER = "GOVERNMENT OF INDIA / MINISTRY OF FINANCE / CENTRAL BOARD OF DIRECT TAXES"
GST_HEADER = "[TO BE PUBLISHED IN THE GAZETTE OF INDIA, EXTRAORDINARY, PART II, SECTION 3]"
# Devanagari needs a font with the glyphs. Windows ships Nirmala UI; elsewhere use Noto.
HINDI_FONTS = [
    Path("C:/Windows/Fonts/Nirmala.ttc"),
    Path("C:/Windows/Fonts/Nirmala.ttf"),
    Path("/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf"),
]

CIRCULAR_PAGES = [
    """Circular No. 3/2025
F. No. 275/12/2025-IT(B)
New Delhi, dated 15th March, 2025

Subject: Deduction of tax at source on fees for professional or technical services under
section 194J of the Income-tax Act, 1961 - clarification regarding.

1. Section 194J of the Income-tax Act, 1961 (the Act) provides for deduction of tax at source
on fees for professional services, fees for technical services, royalty and non-compete fees.
Representations have been received seeking clarity on the threshold for such deduc-
tion where payments are made in instalments during the financial year.

2. The matter has been examined. It is clarified that the threshold of Rs. 50,000 is to be
applied to the aggregate amount credited or paid to the payee during the financial year,
and not to each instalment separately.""",
    """3. Where the payee has furnished a certificate under section 197 of the Act, tax shall
be deducted at the rate specified in that certificate. The deductor is not required to deduct
tax at the rate specified in section 194J for the period covered by the certificate.

4. These clarifications apply to payments made on or after 1st April, 2025. The deduction
under section 80C or section 80D claimed by the payee has no bearing on the
deduction required under section 194J.

(Ravi Kumar)
Under Secretary to the Government of India""",
    "",  # a blank page: a scanned page with no text layer looks like this
]

NOTIFICATION_HINDI = (
    "अधिसूचना संख्या 12/2025 - केंद्रीय कर (दर)\n"
    "केंद्रीय माल और सेवा कर अधिनियम, 2017 की धारा 9 की उप-धारा (1) द्वारा प्रदत्त शक्तियों का प्रयोग करते हुए"
)
NOTIFICATION_PAGES = [
    """Notification No. 12/2025 – Central Tax (Rate)
New Delhi, the 28th June, 2025

G.S.R. 401(E).- In exercise of the powers conferred by sub-section (1) of section 9 of the
Central Goods and Services Tax Act, 2017 (12 of 2017), the Central Government, on the
recommendations of the Council, hereby notifies that the central tax on the intra-State
supply of goods specified in the Schedule below shall be levied at the rate specified.""",
    """Schedule
S. No. | Chapter / Heading | Description of goods | Rate
1. | 0401 | Fresh milk and pasteurised milk | Nil
2. | 1905 | Pizza bread | 2.5%

This notification shall come into force on the 1st day of July, 2025.
[F. No. 354/117/2025-TRU]
(Anita Sharma)
Under Secretary to the Government of India""",
]


def _hindi_font() -> Path | None:
    return next((p for p in HINDI_FONTS if p.exists()), None)


def _page(
    doc: pymupdf.Document, header: str, body: str, number: int, total: int, font: Path | None = None
) -> None:
    page = doc.new_page(width=595, height=842)  # A4
    fontname = "helv"
    if font is not None:  # a TrueType font: needed for Devanagari and for the en-dash
        fontname = "tt"
        page.insert_font(fontname=fontname, fontfile=str(font))
    page.insert_text((50, 40), header, fontsize=8, fontname=fontname)
    if body:
        box = pymupdf.Rect(50, 70, 545, 760)
        page.insert_textbox(box, body, fontsize=10, fontname=fontname)
        page.insert_text((50, 790), DISCLAIMER, fontsize=7, fontname=fontname)
        page.insert_text((270, 810), f"Page {number} of {total}", fontsize=8, fontname=fontname)


def make_circular(path: Path) -> None:
    doc = pymupdf.open()
    total = len(CIRCULAR_PAGES)
    for i, body in enumerate(CIRCULAR_PAGES, start=1):
        if body:
            _page(doc, HEADER, body, i, total)
        else:
            doc.new_page(width=595, height=842)
    doc.set_metadata({"title": "Synthetic sample circular", "creationDate": "", "modDate": ""})
    doc.save(path, garbage=4, deflate=True, no_new_id=True)


def make_notification(path: Path) -> None:
    font = _hindi_font()
    if font is None:
        raise SystemExit("no Devanagari font found; install Noto Sans Devanagari")
    doc = pymupdf.open()
    total = len(NOTIFICATION_PAGES)
    for i, body in enumerate(NOTIFICATION_PAGES, start=1):
        if i == 1:  # bilingual gazette notices put the Hindi version first
            body = f"{NOTIFICATION_HINDI}\n\n{body}"
        _page(doc, GST_HEADER, body, i, total, font=font)
    doc.subset_fonts()
    doc.set_metadata({"title": "Synthetic sample notification", "creationDate": "", "modDate": ""})
    doc.save(path, garbage=4, deflate=True, no_new_id=True)


# Gazette Act layout: a main column (x 117-477) and section headings as narrow margin notes,
# left on even pages and right on odd ones. (y, body, note) per section start.
ACT_COLUMN = (117, 477)
ACT_PAGES: list[list[tuple[float, str, str | None]]] = [
    [
        (100, "THE SAMPLE TAXATION ACT, 2025\nCHAPTER I\nPRELIMINARY", None),
        (
            160,
            "1. (1) This Act may be called the Sample Taxation Act, 2025, and it is a synthetic\n"
            "document made up for the TaxLens test-suite.\n"
            "(2) It extends to the whole of the imaginary territory to which it applies.",
            "Short title,\nextent and\ncommencement.",
        ),
        (
            250,
            "2. In this Act, unless the context otherwise requires, the following table applies\n"
            "for computing the distance referred to in section 80C and section 194J:-",
            "Definitions.",
        ),
        # A numbered table row inside section 2: it must not become a section.
        (330, "1. More than 10000 and upto 100000. Two kilometres.", None),
        (350, "2. More than 100000 and upto 1000000. Six kilometres.", None),
    ],
    [
        # The section number alone on its line, as in parts of the Income-tax Act, 2025.
        (
            100,
            "80C.\nAn individual or a Hindu undivided family shall be allowed a deduction of the\n"
            "whole of the amount paid in the previous year as life insurance premia, being\n"
            "the aggregate of the sums specified in sub-section (2), not exceeding Rs. 1,50,000.",
            "Deduction in\nrespect of life\ninsurance premia,\netc.",
        ),
        (
            220,
            "194J. (1) Any person paying fees for professional services or fees for technical\n"
            "services shall deduct tax at source at the rate of ten per cent of such sum, as\n"
            "provided under the Sample Taxation Act, 2025 and the Income-tax Act, 1961.",
            "Fees for\nprofessional or\ntechnical services.",
        ),
    ],
]


def make_act(path: Path) -> None:
    doc = pymupdf.open()
    left, right = ACT_COLUMN
    for number, sections in enumerate(ACT_PAGES, start=1):
        page = doc.new_page(width=595, height=842)
        page.insert_text((200, 40), "THE GAZETTE OF INDIA EXTRAORDINARY", fontsize=9)
        note_x = 500 if number % 2 else 57  # right margin on odd pages, left on even
        for y, body, note in sections:
            page.insert_textbox(pymupdf.Rect(left, y, right, y + 120), body, fontsize=10)
            if note:
                box = pymupdf.Rect(note_x, y, note_x + 55, y + 80)
                page.insert_textbox(box, note, fontsize=7)
        if number == 2:
            page.insert_text((57, 285), "43 of 1961.", fontsize=7)  # cites the 1961 Act
        page.insert_text((50, 790), DISCLAIMER, fontsize=7)
    doc.set_metadata({"title": "Synthetic sample act", "creationDate": "", "modDate": ""})
    doc.save(path, garbage=4, deflate=True, no_new_id=True)


if __name__ == "__main__":
    make_circular(HERE / "sample_circular.pdf")
    make_notification(HERE / "sample_notification.pdf")
    make_act(HERE / "sample_act.pdf")
    for name in ("sample_circular.pdf", "sample_notification.pdf", "sample_act.pdf"):
        print(f"wrote {name} ({(HERE / name).stat().st_size:,} bytes)")
