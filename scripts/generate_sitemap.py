#!/usr/bin/env python3
"""
Sitemap Generator for Strata Documentation Site.

Generates a compliant sitemap.xml adhering to the Sitemaps XML protocol (0.9),
enabling search engines to discover and index all documentation pages,
tutorials, how-to guides, architectural explanations, and API reference symbols.
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET
from xml.dom import minidom

ROOT_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT_DIR / "docs"
DEFAULT_BASE_URL = "https://ethqnol.github.io/strata-mojo"
DEFAULT_OUTPUT_FILE = DOCS_DIR / "sitemap.xml"

# Route classifications for SEO priority and change frequency
ROUTE_CONFIG = {
    "root": {
        "priority": "1.0",
        "changefreq": "daily"
    },
    "tutorials": {
        "priority": "0.9",
        "changefreq": "weekly"
    },
    "how_to": {
        "priority": "0.8",
        "changefreq": "weekly"
    },
    "explanation": {
        "priority": "0.8",
        "changefreq": "weekly"
    },
    "reference_index": {
        "priority": "0.7",
        "changefreq": "weekly"
    },
    "reference_symbol": {
        "priority": "0.6",
        "changefreq": "monthly"
    },
    "default": {
        "priority": "0.5",
        "changefreq": "monthly"
    }
}


def classify_route(doc_id: str) -> dict:
    """Classifies a document ID to determine its SEO priority and change frequency."""
    if not doc_id or doc_id == "root":
        return ROUTE_CONFIG["root"]
    if doc_id.startswith("tutorials/"):
        return ROUTE_CONFIG["tutorials"]
    if doc_id.startswith("how_to/"):
        return ROUTE_CONFIG["how_to"]
    if doc_id.startswith("explanation/"):
        return ROUTE_CONFIG["explanation"]
    if doc_id.startswith("reference/"):
        if doc_id.endswith("/index"):
            return ROUTE_CONFIG["reference_index"]
        return ROUTE_CONFIG["reference_symbol"]
    return ROUTE_CONFIG["default"]


def get_file_lastmod(file_path: Path) -> str:
    """Extracts YYYY-MM-DD last modified timestamp from a file."""
    if file_path.exists():
        mtime = file_path.stat().st_mtime
        dt = datetime.fromtimestamp(mtime, tz=timezone.utc)
        return dt.strftime("%Y-%m-%d")
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def discover_doc_pages(docs_dir: Path) -> list[dict]:
    """
    Discovers all documentation markdown pages across tutorials, how_to, explanation,
    and reference directories.
    """
    pages = []

    # 1. Root landing page
    index_html = docs_dir / "index.html"
    pages.append({
        "doc_id": "root",
        "rel_path": "",
        "file_path": index_html,
        "lastmod": get_file_lastmod(index_html)
    })

    # 2. Quadrants: tutorials, how_to, explanation
    quadrants = ["tutorials", "how_to", "explanation"]
    for q in quadrants:
        q_dir = docs_dir / q
        if q_dir.exists():
            for md_file in sorted(q_dir.glob("*.md")):
                doc_id = f"{q}/{md_file.stem}"
                pages.append({
                    "doc_id": doc_id,
                    "rel_path": f"#{doc_id}",
                    "file_path": md_file,
                    "lastmod": get_file_lastmod(md_file)
                })

    # 3. Reference module indices and individual symbol pages
    ref_dir = docs_dir / "reference"
    if ref_dir.exists():
        for mod_dir in sorted([d for d in ref_dir.iterdir() if d.is_dir()]):
            mod_key = mod_dir.name

            # Module index
            mod_index = mod_dir / "index.md"
            if mod_index.exists():
                doc_id = f"reference/{mod_key}/index"
                pages.append({
                    "doc_id": doc_id,
                    "rel_path": f"#{doc_id}",
                    "file_path": mod_index,
                    "lastmod": get_file_lastmod(mod_index)
                })

            # Individual symbols
            for sym_file in sorted(mod_dir.glob("*.md")):
                if sym_file.name == "index.md":
                    continue
                doc_id = f"reference/{mod_key}/{sym_file.stem}"
                pages.append({
                    "doc_id": doc_id,
                    "rel_path": f"#{doc_id}",
                    "file_path": sym_file,
                    "lastmod": get_file_lastmod(sym_file)
                })

    return pages


def build_sitemap_xml(base_url: str, pages: list[dict]) -> str:
    """Builds a formatted sitemap XML string."""
    clean_base = base_url.rstrip("/")

    urlset = ET.Element("urlset")
    urlset.set("xmlns", "http://www.sitemaps.org/schemas/sitemap/0.9")

    for p in pages:
        doc_id = p["doc_id"]
        rel_path = p["rel_path"]
        lastmod = p["lastmod"]

        cfg = classify_route(doc_id)

        url_el = ET.SubElement(urlset, "url")

        loc_el = ET.SubElement(url_el, "loc")
        if not rel_path:
            loc_el.text = f"{clean_base}/"
        else:
            loc_el.text = f"{clean_base}/{rel_path}"

        lastmod_el = ET.SubElement(url_el, "lastmod")
        lastmod_el.text = lastmod

        changefreq_el = ET.SubElement(url_el, "changefreq")
        changefreq_el.text = cfg["changefreq"]

        priority_el = ET.SubElement(url_el, "priority")
        priority_el.text = cfg["priority"]

    # Format cleanly with indentation
    rough_string = ET.tostring(urlset, encoding="utf-8")
    reparsed = minidom.parseString(rough_string)
    formatted = reparsed.toprettyxml(indent="  ", encoding="utf-8").decode("utf-8")

    # Remove extra blank lines that minidom sometimes inserts
    cleaned_lines = [line for line in formatted.splitlines() if line.strip()]
    return "\n".join(cleaned_lines) + "\n"


def generate_sitemap(
    base_url: str = DEFAULT_BASE_URL,
    docs_dir: Path | None = None,
    output_file: Path | None = None
) -> Path:
    """
    Main entrypoint to generate sitemap.xml.

    Returns the path to the written sitemap.xml file.
    """
    if docs_dir is None:
        docs_dir = DOCS_DIR
    if output_file is None:
        output_file = DEFAULT_OUTPUT_FILE

    pages = discover_doc_pages(docs_dir)
    xml_content = build_sitemap_xml(base_url, pages)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(xml_content, encoding="utf-8")

    print(f"  ✓ Generated sitemap with {len(pages)} URLs at {output_file.relative_to(ROOT_DIR) if output_file.is_relative_to(ROOT_DIR) else output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Generate sitemap.xml for Strata documentation.")
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"Base URL for site (default: {DEFAULT_BASE_URL})"
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=DOCS_DIR,
        help="Path to docs directory"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help="Path to write output sitemap.xml"
    )

    args = parser.parse_args()
    print("Generating Strata Documentation Sitemap...")
    generate_sitemap(base_url=args.base_url, docs_dir=args.docs_dir, output_file=args.output)
    print("✨ Sitemap generation complete!")


if __name__ == "__main__":
    main()
