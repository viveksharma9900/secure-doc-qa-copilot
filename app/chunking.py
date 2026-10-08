def chunk_text(text: str, max_chars: int = 700, overlap_words: int = 20) -> list[str]:
    """Paragraph-aware chunking with word overlap between chunks."""
    paras = [p.strip() for p in text.replace("\r", "").split("\n\n") if p.strip()]
    chunks, current = [], ""
    for p in paras:
        if current and len(current) + len(p) + 1 > max_chars:
            chunks.append(current)
            tail = " ".join(current.split()[-overlap_words:])
            current = f"{tail} {p}".strip()
        else:
            current = f"{current}\n{p}".strip() if current else p
        while len(current) > max_chars * 2:  # very long paragraph
            chunks.append(current[:max_chars])
            current = current[max_chars - overlap_words * 6:]
    if current:
        chunks.append(current)
    return chunks
