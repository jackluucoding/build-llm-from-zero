"""
concat_book.py -- Concatenates all book chapters into combined.md for PDF generation.

Run from the project root:
    python scripts/concat_book.py
"""

import os
import re

CHAPTERS = [
    "book/preface.md",
    "book/section-1-foundations/ch01-environment-setup.md",
    "book/section-1-foundations/ch02-what-is-an-llm.md",
    "book/section-1-foundations/ch03-tensors-and-pytorch.md",
    "book/section-1-foundations/ch04-tokenization.md",
    "book/section-1-foundations/ch05-embeddings.md",
    "book/section-2-attention/ch06-self-attention.md",
    "book/section-2-attention/ch07-multi-head-attention.md",
    "book/section-2-attention/ch08-feedforward-and-norms.md",
    "book/section-3-the-transformer/ch09-transformer-block.md",
    "book/section-3-the-transformer/ch10-full-gpt-architecture.md",
    "book/section-3-the-transformer/ch11-causal-language-modeling.md",
    "book/section-4-training/ch12-dataset-and-dataloader.md",
    "book/section-4-training/ch13-training-loop.md",
    "book/section-4-training/ch14-checkpointing.md",
    "book/section-5-generation/ch15-greedy-and-sampling.md",
    "book/section-5-generation/ch16-temperature-and-topk.md",
    "book/section-5-generation/ch17-putting-it-all-together.md",
]

# Part headers for the PDF (raw LaTeX \part{})
# Note: the LaTeX template already prepends "Part N" via \titleformat{\part},
# so only pass the section name here to avoid "Part 1 Part 1: Foundations".
PDF_PART_HEADERS = {
    "book/section-1-foundations/ch01-environment-setup.md":
        "\n\n\\part{Setup}\n\n",
    "book/section-1-foundations/ch02-what-is-an-llm.md":
        "\n\n\\part{Foundations}\n\n",
    "book/section-2-attention/ch06-self-attention.md":
        "\n\n\\part{The Attention Mechanism}\n\n",
    "book/section-3-the-transformer/ch09-transformer-block.md":
        "\n\n\\part{The Transformer}\n\n",
    "book/section-4-training/ch12-dataset-and-dataloader.md":
        "\n\n\\part{Training}\n\n",
    "book/section-5-generation/ch15-greedy-and-sampling.md":
        "\n\n\\part{Generation}\n\n",
}

PDF_OUTPUT = "book/combined.md"


def strip_chapter_prefix(content, is_preface=False):
    """
    Clean chapter content for PDF:
    - Remove 'Chapter N: ' from h1 headings (LaTeX auto-numbers them)
    - Remove 'Next up:' navigation links (dead links in the PDF)
    - Mark preface heading as unnumbered so ch01 becomes Chapter 1
    """
    content = re.sub(r'^# Chapter \d+:\s*', '# ', content, flags=re.MULTILINE)
    content = re.sub(r'^\*Next up:.*\*\s*$', '', content, flags=re.MULTILINE)
    if is_preface:
        content = re.sub(r'^(# .+?)$', r'\1 {.unnumbered}', content, count=1, flags=re.MULTILINE)
    return content


def build_pdf():
    parts = []
    for path in CHAPTERS:
        if not os.path.exists(path):
            print(f"WARNING: {path} not found, skipping.")
            continue
        if path in PDF_PART_HEADERS:
            parts.append(PDF_PART_HEADERS[path])
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        parts.append(strip_chapter_prefix(content.strip(), is_preface=(path == "book/preface.md")))
        parts.append("\n\n\\newpage\n\n")

    with open(PDF_OUTPUT, "w", encoding="utf-8") as f:
        f.write("".join(parts))
    print(f"PDF source  -> {PDF_OUTPUT}")


if __name__ == "__main__":
    total_chars = sum(
        len(open(p, encoding="utf-8").read())
        for p in CHAPTERS if os.path.exists(p)
    )
    print(f"Combining {len(CHAPTERS)} chapters ({total_chars:,} characters)...")
    build_pdf()
    print("Done.")
