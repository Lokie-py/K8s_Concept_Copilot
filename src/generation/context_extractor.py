from langchain_core.documents import Document

def extract_context(
    documents: list[Document],
    max_chunks: int = 5
) -> str:
 
    if not documents:
        return ""

    selected_documents = documents[:max_chunks]

    context_parts = []

    for i, document in enumerate(
        selected_documents,
        start=1
    ):
        source = document.metadata.get(
            "source",
            "Unknown source"
        )

        context_parts.append(
            f"""
[Context {i}]
Source: {source}

{document.page_content}
""".strip()
        )

    return "\n\n".join(context_parts)

if __name__ == "__main__":

    documents = [
        Document(
            page_content=(
                "Pods are the smallest deployable units "
                "of computing that you can create and "
                "manage in Kubernetes."
            ),
            metadata={
                "source": "Workloads.pdf"
            }
        ),
        Document(
            page_content=(
                "A Pod is a group of one or more containers "
                "with shared storage and network resources."
            ),
            metadata={
                "source": "Workloads.pdf"
            }
        )
    ]

    context = extract_context(
        documents,
        max_chunks=5
    )

    print("=" * 70)
    print("EXTRACTED CONTEXT")
    print("=" * 70)
    print(context)