import markdown2
from pathlib import Path


def convert_markdown_to_html(markdown_file, output_file):
    # Read markdown content
    md_content = markdown_file.read_text(encoding="utf-8")

    # Convert to HTML
    html = markdown2.markdown(md_content, extras=["fenced-code-blocks", "latex"])

    # Create HTML document with MathJax
    full_html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{markdown_file.stem}</title>
    <script>
        MathJax = {{
            tex: {{
                inlineMath: [['$', '$']],
                displayMath: [['$$', '$$']]
            }}
        }};
    </script>
    <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <style>
        body {{
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            line-height: 1.6;
        }}
        pre {{
            background-color: #f6f8fa;
            padding: 16px;
            border-radius: 6px;
            overflow: auto;
        }}
        code {{
            font-family: ui-monospace, SFMono-Regular, SF Mono, Menlo, Consolas, Liberation Mono, monospace;
        }}
    </style>
</head>
<body>
    {html}
</body>
</html>"""

    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Write to file
    output_file.write_text(full_html, encoding="utf-8")
    print(f"Converted {markdown_file} to {output_file}")


def main():
    # Get the explain directory
    explain_dir = Path("explain")
    html_dir = explain_dir / "html"

    # Create html directory if it doesn't exist
    html_dir.mkdir(exist_ok=True)

    # Convert all markdown files
    for md_file in explain_dir.glob("*.md"):
        html_file = html_dir / f"{md_file.stem}.html"
        convert_markdown_to_html(md_file, html_file)


if __name__ == "__main__":
    main()
