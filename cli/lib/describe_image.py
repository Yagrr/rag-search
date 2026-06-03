import os
import time
import mimetypes

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.types import GenerateContentResponse

from utils_search import DEFAULT_LLM


load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment is not set")

client = genai.Client(api_key=api_key)
model = DEFAULT_LLM

def describe_image(prompt: str, parts) -> GenerateContentResponse:
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                safety_settings=[
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                        threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
                    ),
                ]
            ),
        )
    except Exception as e:
        print(f"{e} - Retrying LLM call...")
        time.sleep(5)
        response = describe_image(prompt, parts)

    if response is None or response.text is None:
        response = describe_image(prompt, parts)

    return response


def command_describe_image(path_image: str, query: str) -> None:
    if not os.path.exists(path_image):
        raise ValueError("Error: input path to image does not exist")

    with open(path_image, "rb") as f:
        img: bytes = f.read()

    mime, _ = mimetypes.guess_type(path_image)
    mime = mime or "image/jpeg"

    system_prompt: str = """Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
    - Synthesize visual and textual information
    - Focus on movie-specific details (actors, scenes, style, etc.)
    - Return only the rewritten query, without any additional commentary
    """

    parts = [
        system_prompt,
        types.Part.from_bytes(data=img, mime_type=mime),
        query.strip(),
    ]

    response = describe_image(query, parts)

    if response.text is not None:
        print(f"Rewritten query: {response.text.strip()}")
    if response.usage_metadata is not None:
        print(f"Total tokens:    {response.usage_metadata.total_token_count}")

