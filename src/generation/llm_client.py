import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from src.config import GEMINI_API_KEY

logger = logging.getLogger(__name__)

MODEL_NAME = "gemini-3.5-flash"

_llm = None


def get_llm() -> ChatGoogleGenerativeAI:
  
    global _llm

    if _llm is None:

        if not GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY is not configured."
            )

        logger.info(
            "Initializing Gemini model: %s",
            MODEL_NAME
        )

        _llm = ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            google_api_key=GEMINI_API_KEY,
            temperature=0,
        )

    return _llm


def generate(prompt: str) -> str:

    if not prompt.strip():
        raise ValueError("Prompt cannot be empty.")

    llm = get_llm()

    response = llm.invoke(prompt)

    content = response.content

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):

        text_parts = []

        for block in content:

            if (
                isinstance(block, dict)
                and block.get("type") == "text"
            ):
                text_parts.append(
                    block.get("text", "")
                )

        return "".join(text_parts).strip()

    return str(content).strip()


if __name__ == "__main__":

    test_prompt = """
    Answer the following question in one concise paragraph.

    Question:
    What is a Kubernetes Pod?
    """

    answer = generate(test_prompt)

    print("=" * 70)
    print("GEMINI RESPONSE")
    print("=" * 70)
    print(answer)