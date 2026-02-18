"""
Create a playful, image-rich Fractions PPT for Class 5 (Telangana SSC).
Exclude: 7, 8, 9, 10, 12, 13, 14, 15. Include all 11 remaining: 1, 2, 3, 4, 5, 6, 11, 16, 17, 18, 19.
"""
from __future__ import annotations

import io
import ssl
import urllib.request
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt

# --- Image URLs per slide (as provided). Some may fail to download; we skip and keep layout. ---
SLIDES = [
    {
        "title": "Fractions — Where Sharing Gets Mathematical!",
        "subtitle": "Class: V  |  Subject: Mathematics\nBoard: SSC, State of Telangana",
        "images": [
            "https://images.openai.com/static-rsc-3/AP1sgrfysnusvfxsnWDXBpUXp4PL-jDLhnSjpg_OCul06lnWNGfKxcWXdXnUxzWA7l1JLCVQjQ1d3uCLi0Kv8Jl8qFSM9uvqLSMlKNtXS90?purpose=fullsize&v=1",
            "https://images.openai.com/static-rsc-3/6zg26lyf55N6UPtYByAz6XXuoJZjfn9SQ2G8JTtWZgOPtrUE0pJQ0Wu0yHzrkUqfuYui-Q-DvFRvQGaMtbJvGqW--_6TIgKIEsUifmOctX8?purpose=fullsize&v=1",
            "https://m.media-amazon.com/images/I/71FQe223o3L._AC_UF1000%2C1000_QL80_.jpg",
            "https://m.media-amazon.com/images/I/71kxGRWk9aL._AC_UF894%2C1000_QL80_.jpg",
        ],
        "body": None,
    },
    # Slide 2 — What we'll learn today
    {
        "title": "What we'll learn today!",
        "subtitle": "A quick map of our fraction journey",
        "body": "• What is a fraction? (parts of a whole)\n• Numerator & denominator\n• Like and unlike fractions\n• And why sharing pizza is real math!",
        "images": [
            "https://m.media-amazon.com/images/I/71FQe223o3L._AC_UF1000%2C1000_QL80_.jpg",
            "https://images.twinkl.co.uk/tw1n/image/private/t_630_eco/image_repo/c4/7a/t-m-1647521024-chocolate-bar-fractions-activity_ver_1.jpeg",
            "https://images.openai.com/static-rsc-3/CoCk-DBA6MhhG5u0DHY2ZYI9qeIbdA-PgTTzZUbQ0cCubfHinhBLwTIsEyDkvZt2ZcF2Oi02wH7Om1v771D4Pdgw9GnIqI9dlk1d4psZDbU?purpose=fullsize&v=1",
            "https://www.math-only-math.com/images/like-fractions.png",
        ],
    },
    {
        "title": "Have you ever shared a pizza or chocolate with your friend?",
        "subtitle": "Yum + Math = Win!  —  Real-Life Hook",
        "body": "Rahim has 1 chocolate bar.\nHe breaks it into 4 equal pieces and eats 1 piece.\n\nHe ate 1 out of 4 equal parts → 1/4\n\nThat is a Fraction! (And totally fair sharing!)",
        "images": [
            "https://images.twinkl.co.uk/tw1n/image/private/t_630_eco/image_repo/c4/7a/t-m-1647521024-chocolate-bar-fractions-activity_ver_1.jpeg",
            "https://images.openai.com/static-rsc-3/4ZzyJwEhtSCYySqPjQgKLbb64MgZvogg0gL7d_GYD1Up5yrINZ5r39gDiGPFi66renqbsJDPJMd2oveOXAyCowP7pf_kUT98BQ2zQf0nOUg?purpose=fullsize&v=1",
            "https://www.researchgate.net/publication/349822215/figure/fig1/AS:998004022579202@1614954109539/Example-of-representation-of-the-fraction-1-4-with-A-an-area-model-B-a-number-line.jpg",
            "https://cdn.education.com/worksheet-image/236640/fractions-shapes-14-first-grade-2023-01-25.gif",
        ],
    },
    {
        "title": "What is a Fraction? (Hint: Equal parts only!)",
        "subtitle": "Part of a whole — and the whole must be divided into equal parts!",
        "body": '"If something is not divided into equal parts — it is NOT a fraction!"\n\n(Only the equally divided shape represents a fraction. No cheating!)',
        "images": [
            "https://images.openai.com/static-rsc-3/noUgyvaAEx9napXf7_Dwomcwvlg-fF0cwHJh7YMcMWILKHlyCxgKFTOWHam95afdPfWrnJbumd1nHioTGsWRvmQMLYYeKzDCeOfuLEw0ixA?purpose=fullsize&v=1",
            "https://ecdn.teacherspayteachers.com/cdn-cgi/image/format%3Davif%2Cquality%3D70%2Cwidth%3D525%2Cheight%3D525%2Conerror%3Dredirect/thumbitem/Fraction-Activity-and-Anchor-Chart-Equal-vs-Unequal-Parts-FREE-Fractions-3619802-1657527503/750f-3619802-1.jpg",
            "https://ecdn.teacherspayteachers.com/thumbitem/Equal-and-Unequal-Parts-Anchor-chart-8671585-1666002019/original-8671585-1.jpg",
        ],
    },
    {
        "title": "Meet the Fraction Family: Numerator & Denominator",
        "subtitle": "3/4  —  Numerator → 3 (parts taken)  |  Denominator → 4 (total equal parts)  |  The line = Fraction bar",
        "body": None,
        "images": [
            "https://images.twinkl.co.uk/tr/raw/upload/u/ux/numerator_ver_1.jpg",
            "https://hands-oneducation.com/img/key-stage-two/fractions-four/fractions_four_1.webp",
            "https://images.openai.com/static-rsc-3/CoCk-DBA6MhhG5u0DHY2ZYI9qeIbdA-PgTTzZUbQ0cCubfHinhBLwTIsEyDkvZt2ZcF2Oi02wH7Om1v771D4Pdgw9GnIqI9dlk1d4psZDbU?purpose=fullsize&v=1",
            "https://images.openai.com/static-rsc-3/KL8DsTIa9mf8HilNCxz9GTD56A9nVWdin5GUNXGgLW073hfVrTu87Hhap2W99KV7o9X0Y2VYv3ClaTcLpLMaxo-pcXF5Ag7ZIVJN-W-cpkg?purpose=fullsize&v=1",
        ],
    },
    {
        "title": "Try These! (You've got this!)",
        "subtitle": "How much is shaded? Write the fraction.",
        "body": "Answers:  1/3  |  5/8  |  3/4  |  1/2  —  Well done!",
        "images": [
            "https://images.openai.com/static-rsc-3/xv_RuQsUxyv15zf_JFv6CSY55TUnIwjZmtrRExFgUdNSMAxNFKaApmtVOeNDDPnIrlLIUgM_05ofKhsT-7o9jiKsT9WofZZWEM72UOGrV58?purpose=fullsize&v=1",
            "https://vt-vtwa-assets.varsitytutors.com/vt-vtwa/uploads/problem_question_image/image/17184/5_8_rectangle.png",
            "https://jsx-images.mathspace.co/ac-3-2021-1038-chapter-7/7.06-1.svg?versionId=PW1jEk1bK5S1BbejDpCi2h38YUnv5BsF",
            "https://us-static.z-dn.net/files/ddb/972dd48ffb38cb40fcc052cb8443e846.png",
        ],
    },
    {
        "title": "Like vs Unlike — Fraction Friends!",
        "subtitle": "Like = same denominator (e.g. 1/7, 3/7, 5/7)  |  Unlike = different denominators (e.g. 1/2, 2/3, 3/5)",
        "body": None,
        "images": [
            "https://www.math-only-math.com/images/like-fractions.png",
            "https://static1.squarespace.com/static/54905286e4b050812345644c/560b1c27e4b0a43ccfccc9d9/65e7aa724d9dc22172cdd411/1709852095221/Add-Fractions-Banner.jpg?format=1500w",
            "https://images.openai.com/static-rsc-3/0TAjCD2A3E4OTcTlUjlx8D5jjf8ORCRPM_2d6J_A3ZADZ45hV8q5Phv4WDUe74hlZ8hfxeqBfS4lZCEDccY-e2sTRCZRtwQsnq1Gb_Dwhg0?purpose=fullsize&v=1",
            "https://images.openai.com/static-rsc-3/74FCQ_b58aGkhJm3uYVLv9y9cvBwPv6svIB5TgoY5K8IsPABZ-c0DNEUcjU4J44bglIGLokWBrK1_VlwEddWyK7tg-nT5o7JYJQtGpy5J20?purpose=fullsize&v=1",
        ],
    },
    # Slide 16 — Quick revision
    {
        "title": "Quick revision — What did we learn?",
        "subtitle": "Fractions in a nutshell!",
        "body": "• Fraction = part of a whole (equal parts only!)\n• Numerator = parts taken, Denominator = total parts\n• Like fractions = same denominator; Unlike = different",
        "images": [
            "https://images.openai.com/static-rsc-3/noUgyvaAEx9napXf7_Dwomcwvlg-fF0cwHJh7YMcMWILKHlyCxgKFTOWHam95afdPfWrnJbumd1nHioTGsWRvmQMLYYeKzDCeOfuLEw0ixA?purpose=fullsize&v=1",
            "https://images.twinkl.co.uk/tr/raw/upload/u/ux/numerator_ver_1.jpg",
            "https://static1.squarespace.com/static/54905286e4b050812345644c/560b1c27e4b0a43ccfccc9d9/65e7aa724d9dc22172cdd411/1709852095221/Add-Fractions-Banner.jpg?format=1500w",
        ],
    },
    # Slide 17 — Practice / activity
    {
        "title": "Practice time — Find fractions around you!",
        "subtitle": "At home: share a chapati, a chocolate, or a fruit.",
        "body": "How many equal parts? How many did you take? That's your fraction!",
        "images": [
            "https://cdn.education.com/worksheet-image/236640/fractions-shapes-14-first-grade-2023-01-25.gif",
            "https://images.openai.com/static-rsc-3/4ZzyJwEhtSCYySqPjQgKLbb64MgZvogg0gL7d_GYD1Up5yrINZ5r39gDiGPFi66renqbsJDPJMd2oveOXAyCowP7pf_kUT98BQ2zQf0nOUg?purpose=fullsize&v=1",
            "https://ecdn.teacherspayteachers.com/thumbitem/Equal-and-Unequal-Parts-Anchor-chart-8671585-1666002019/original-8671585-1.jpg",
            "https://hands-oneducation.com/img/key-stage-two/fractions-four/fractions_four_1.webp",
        ],
    },
    # Slide 18 — Recap / questions
    {
        "title": "Any questions?",
        "subtitle": "Recap: Equal parts → Fraction. Numerator on top, denominator below!",
        "body": "Keep practising with real things — pizza, chocolate, paper folding!",
        "images": [
            "https://www.explorelearning.com/user_area/content_media/raw/celebrate-math-growth-in-classroom-header.webp?format=webp&quality=80&w=815",
            "https://images.openai.com/static-rsc-3/nbxL1SBrdOtunk8qmGtx1zq4qsAp1J3PHIU3TeEZAtFkiaxUV1F0c8Yut2sAW0JWHxnbamOtWfddccUUEDEjrp1L1ZwoWfD-7gSbzkqXU0E?purpose=fullsize&v=1",
        ],
    },
    {
        "title": "Thank You!",
        "subtitle": "Keep slicing, sharing, and solving!  —  Fractions are everywhere!",
        "body": None,
        "images": [
            "https://images.openai.com/static-rsc-3/QwvdUDqMZrQ538jwlxVr-rs4TDphWhnMiR9ImhmBv0hmU6-IYxcemnBSJjKXd3tEsbMK70Et_B89kWsjMWVHDgTli92L4Jo9LqXgKZmRg0c?purpose=fullsize&v=1",
            "https://images.openai.com/static-rsc-3/nbxL1SBrdOtunk8qmGtx1zq4qsAp1J3PHIU3TeEZAtFkiaxUV1F0c8Yut2sAW0JWHxnbamOtWfddccUUEDEjrp1L1ZwoWfD-7gSbzkqXU0E?purpose=fullsize&v=1",
            "https://www.explorelearning.com/user_area/content_media/raw/celebrate-math-growth-in-classroom-header.webp?format=webp&quality=80&w=815",
            "https://www.explorelearning.com/user_area/content_media/raw/math-holidays.webp?format=webp&quality=80&w=509",
        ],
    },
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
}


def download_image(url: str, timeout: int = 15) -> bytes | None:
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        ctx = ssl.create_default_context()
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
            return r.read()
    except Exception:
        return None


def add_slide(prs: Presentation, data: dict, slide_index: int) -> None:
    layout = prs.slide_layouts[6]  # Blank
    slide = prs.slides.add_slide(layout)

    # Title at top
    left, top, width, height = Inches(0.5), Inches(0.3), Inches(9), Inches(1.0)
    tx = slide.shapes.add_textbox(left, top, width, height)
    tf = tx.text_frame
    p = tf.paragraphs[0]
    p.text = data["title"]
    p.font.size = Pt(28 if slide_index == 0 else 22)
    p.font.bold = True
    p.font.color.rgb = RGBColor(0x44, 0x22, 0x88)

    # Subtitle
    if data.get("subtitle"):
        tx2 = slide.shapes.add_textbox(Inches(0.5), Inches(1.2), Inches(9), Inches(0.8))
        tf2 = tx2.text_frame
        tf2.word_wrap = True
        p2 = tf2.paragraphs[0]
        p2.text = data["subtitle"].replace("  |  ", "\n")
        p2.font.size = Pt(16)
        p2.font.color.rgb = RGBColor(0x33, 0x33, 0x33)

    # Body text if present
    body_top = Inches(2.0)
    if data.get("body"):
        tx3 = slide.shapes.add_textbox(Inches(0.5), body_top, Inches(5), Inches(2.2))
        tf3 = tx3.text_frame
        tf3.word_wrap = True
        for line in data["body"].split("\n"):
            p3 = tf3.add_paragraph()
            p3.text = line
            p3.font.size = Pt(14)
        body_top = Inches(2.0)

    # Images in a grid (2x2 or 2x1/1x2 for 3)
    images = data.get("images") or []
    img_streams: list[bytes] = []
    for url in images:
        content = download_image(url)
        if content:
            img_streams.append(content)

    if img_streams:
        n = len(img_streams)
        if n == 1:
            try:
                pic = slide.shapes.add_picture(
                    io.BytesIO(img_streams[0]), Inches(5.5), Inches(2), width=Inches(4), height=Inches(3.5)
                )
            except Exception:
                pass
        elif n == 2:
            for i, stream in enumerate(img_streams):
                try:
                    x = 5.5 + (i % 2) * 2.2
                    slide.shapes.add_picture(io.BytesIO(stream), Inches(x), Inches(2), width=Inches(2), height=Inches(1.8))
                except Exception:
                    pass
        else:
            # 2x2 or 2x2 + 1
            img_w, img_h = 2.0, 1.6
            start_x, start_y = 5.5, 2.0
            for i, stream in enumerate(img_streams[:4]):
                row, col = i // 2, i % 2
                try:
                    slide.shapes.add_picture(
                        io.BytesIO(stream),
                        Inches(start_x + col * (img_w + 0.15)),
                        Inches(start_y + row * (img_h + 0.15)),
                        width=Inches(img_w),
                        height=Inches(img_h),
                    )
                except Exception:
                    pass

    # Playful footer on content slides (not title or thank you)
    if slide_index not in (0, len(SLIDES) - 1):
        footer = slide.shapes.add_textbox(Inches(0.5), Inches(7.0), Inches(9), Inches(0.4))
        footer.text_frame.paragraphs[0].text = "Math is fun when we share!  —  Class 5  •  Fractions"
        footer.text_frame.paragraphs[0].font.size = Pt(10)
        footer.text_frame.paragraphs[0].font.italic = True
        footer.text_frame.paragraphs[0].font.color.rgb = RGBColor(0x66, 0x66, 0x66)


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent
    out_path = out_dir / "Fractions_Class5_Part1.pptx"

    prs = Presentation()
    prs.slide_width = Inches(10)
    prs.slide_height = Inches(7.5)

    for i, data in enumerate(SLIDES):
        add_slide(prs, data, i)

    try:
        prs.save(str(out_path))
    except PermissionError:
        out_path = out_dir / "Fractions_Class5_Part1_new.pptx"
        prs.save(str(out_path))
        print("(Original file was in use; saved as Fractions_Class5_Part1_new.pptx)")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
