import json
import re
import time

import requests


def safe_unicode_decode(content):
    # Regular expression to find all Unicode escape sequences of the form \uXXXX
    unicode_escape_pattern = re.compile(r"\\u([0-9a-fA-F]{4})")

    # Function to replace the Unicode escape with the actual character
    def replace_unicode_escape(match):
        # Convert the matched hexadecimal value into the actual Unicode character
        return chr(int(match.group(1), 16))

    # Perform the substitution
    decoded_content = unicode_escape_pattern.sub(
        replace_unicode_escape, content.decode("utf-8")
    )

    return decoded_content


def clean_json_text(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:].strip()
    elif text.startswith("```"):
        text = text[3:].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text


def html2md(html_content: str, max_retry=4) -> str:
    """
    Convert HTML content to Markdown format using a remote service.

    Args:
        html_content (str): Raw HTML.

    Returns:
        str: Markdown content extracted from the HTML.
    """

    # Patterns
    SCRIPT_PATTERN = r"<[ ]*script.*?\/[ ]*script[ ]*>"
    STYLE_PATTERN = r"<[ ]*style.*?\/[ ]*style[ ]*>"
    META_PATTERN = r"<[ ]*meta.*?>"
    COMMENT_PATTERN = r"<[ ]*!--.*?--[ ]*>"
    LINK_PATTERN = r"<[ ]*link.*?>"
    BASE64_IMG_PATTERN = r'<img[^>]+src="data:image/[^;]+;base64,[^"]+"[^>]*>'
    SVG_PATTERN = r"(<svg[^>]*>)(.*?)(<\/svg>)"

    def replace_svg(html: str, new_content: str = "this is a placeholder") -> str:
        return re.sub(
            SVG_PATTERN,
            lambda match: f"{match.group(1)}{new_content}{match.group(3)}",
            html,
            flags=re.DOTALL,
        )

    def replace_base64_images(html: str, new_image_src: str = "#") -> str:
        return re.sub(BASE64_IMG_PATTERN, f'<img src="{new_image_src}"/>', html)

    def clean_html(html: str, clean_svg: bool = False, clean_base64: bool = False):
        html = re.sub(
            SCRIPT_PATTERN, "", html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL
        )
        html = re.sub(
            STYLE_PATTERN, "", html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL
        )
        html = re.sub(
            META_PATTERN, "", html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL
        )
        html = re.sub(
            COMMENT_PATTERN, "", html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL
        )
        html = re.sub(
            LINK_PATTERN, "", html, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL
        )

        if clean_svg:
            html = replace_svg(html)
        if clean_base64:
            html = replace_base64_images(html)
        return html

    headers = {
        # "Accept": "application/json",
        "Content-Type": "application/json",
        "X-Engine": "browser",
        "X-Md-Heading-Style": "setext",
        "X-Retain-Images": "none",
        "X-Return-Format": "markdown",
        # "X-With-Images-Summary": "true",
        # "X-With-Links-Summary": "true",
    }
    cl_html = clean_html(html_content, clean_svg=True, clean_base64=True)
    data = {"url": "https://example.com", "html": cl_html}

    retry = 0
    sleep_time = 6

    while retry < max_retry:
        response = requests.post(
            url="https://r.jina.ai/",
            headers=headers,
            data=json.dumps(data),
        )

        retry += 1
        if response.status_code != 200:
            print(
                f"Failed to convert HTML to Markdown: {response.status_code}. Retry[{retry}/{max_retry}]"
            )
        else:
            break

        time.sleep(sleep_time)  # Sleep to avoid rate limiting
        sleep_time *= 2

    return response.text


def fix_table(text):
    lines = text.split("\n")
    fixed_lines = []
    for line in lines:
        if (
            "|" in line
            and not line.strip().startswith("|")
            and line.strip().endswith("|")
            and not line.strip().startswith("#")
        ):
            # print(line)
            # split at first "|"
            prefix, rest = line.split("|", 1)
            fixed_lines.append(prefix.strip())
            # print(fixed_lines[-1])
            fixed_lines.append(
                "| " + rest.strip()
            )  # reconstruct proper table header line
            # print(fixed_lines[-1])
            # print('\n')
        else:
            fixed_lines.append(line)
    lines = fixed_lines
    fixed_text = "\n".join(lines)
    return fixed_text


def md_links_to_text(md_content: str) -> str:
    # pattern = r'\[(.+?)\]\([^)]+(?:\s+"[^"]*")?\)'
    pattern = r'\[(.*?)\]\([^)]+(?:\s+"[^"]*")?\)'
    return re.sub(pattern, r"\1", md_content)


def process_html(html_content):
    md_content = html2md(html_content)
    md_content = fix_table(md_content)
    clean_md_content = md_links_to_text(md_content)
    return clean_md_content


if __name__ == "__main__":
    with open(
        "/home/hongyi/CPIPC/datasets/crag-retrieval-summarization/first_20_data/html/data0/page1.html",
        "r",
        encoding="utf-8",
    ) as file:
        content = file.read()
    content = process_html(content)
    print(content)
