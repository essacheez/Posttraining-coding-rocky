"""Render REPORT.md to HTML with embedded figures, then print to PDF with headless Chrome."""
import base64
import os
import re
import subprocess

import markdown

WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

CSS = """
@page { size: Letter; margin: 0.7in 0.75in; }
body { font-family: -apple-system, "Helvetica Neue", Arial, sans-serif; font-size: 10.2pt; line-height: 1.42; color: #1a1a1a; }
h1 { font-size: 17pt; margin: 0 0 4px; line-height: 1.2; }
h1 + p { color: #52514e; margin-top: 2px; }
h2 { font-size: 12.5pt; margin: 18px 0 6px; padding-bottom: 3px; border-bottom: 1px solid #e1e0d9; break-after: avoid; }
p { margin: 6px 0; }
ul, ol { margin: 4px 0 6px; padding-left: 20px; }
li { margin: 2px 0; }
table { border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 9pt; break-inside: avoid; }
th, td { border-bottom: 1px solid #e1e0d9; padding: 4px 6px; text-align: left; vertical-align: top; }
th { background: #f4f3ef; font-weight: 600; }
img { width: 100%; margin: 8px 0 2px; break-inside: avoid; }
p:has(> img) { break-inside: avoid; margin-bottom: 0; }
code { font-size: 8.8pt; background: #f4f3ef; padding: 0 3px; border-radius: 3px; }
.todo { background: #fff4d6; border-left: 3px solid #eda100; padding: 5px 9px; margin: 8px 0; font-size: 9.2pt; }
.appendix { margin-top: 16px; padding-top: 8px; border-top: 1px solid #e1e0d9; font-size: 9pt; color: #52514e; }
.flag { background: #fff4d6; padding: 0 2px; }
.checklist { margin-top: 14px; padding: 8px 14px; border: 1px solid #c3c2b7; border-radius: 6px; background: #f7f7f4; break-inside: avoid; }
.checklist ol { margin: 4px 0; }
em { color: inherit; }
"""


def embed_images(html):
    def replace(match):
        path = os.path.join(WORK, match.group(1))
        encoded = base64.b64encode(open(path, "rb").read()).decode()
        return f'src="data:image/png;base64,{encoded}"'
    return re.sub(r'src="(figures/[^"]+\.png)"', replace, html)


def main():
    source = open(os.path.join(WORK, "REPORT.md"), encoding="utf-8").read()
    body = markdown.markdown(source, extensions=["tables", "md_in_html"])
    body = re.sub(r"\[(TODO[^\]]*|OPEN|CONFIRM)\]", r'<span class="flag">[\1]</span>', body)
    html = f"<!doctype html><html><head><meta charset='utf-8'><title>Rocky post-training report</title><style>{CSS}</style></head><body>{embed_images(body)}</body></html>"
    html_path = os.path.join(WORK, "REPORT.html")
    open(html_path, "w", encoding="utf-8").write(html)
    pdf_path = os.path.join(WORK, "REPORT.pdf")
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf_path}", "file://" + html_path], check=True, capture_output=True)
    print("wrote", pdf_path)


if __name__ == "__main__":
    main()
