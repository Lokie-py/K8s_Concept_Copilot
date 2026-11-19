import re
from collections import Counter
from langchain_core.documents import Document

def collapse_whitespace(text: str) -> str:
    text= re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text= text.strip()

    return text

def strip_bad_characters(text: str) -> str:
    keep_chars= []
    for char in text:
        if char.isprintable() or char in ["\n", "\t", " "]:
            keep_chars.append(char)

    return "".join(keep_chars)

def clean_text(text: str) -> str:
    text= collapse_whitespace(text)
    result= strip_bad_characters(text)

    return result

def find_repeated_lines(docs: list[Document], threshold: float = 0.6) -> set[str]:
    if not docs:
        return set()

    line_counter = Counter()

    for doc in docs:
        unique_lines = {line.strip() for line in doc.page_content.split("\n") if line.strip()}
        line_counter.update(unique_lines)

    repeated_lines = {line for line, count in line_counter.items() if count / len(docs) > threshold}

    return repeated_lines

def clean_documents(docs: list[Document]) -> list[Document]:
    repeated_lines = find_repeated_lines(docs)

    for doc in docs:
        cleaned_lines = [line for line in doc.page_content.split("\n") if line.strip() not in repeated_lines]

        cleaned_text = "\n".join(cleaned_lines)
        doc.page_content = clean_text(cleaned_text)

    return docs