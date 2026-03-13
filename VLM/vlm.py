
import os
import base64
from dotenv import load_dotenv
from groq import Groq

load_dotenv()


def get_api_key():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Get a FREE API key at: https://console.groq.com/keys\n")
        api_key = input("Paste your Groq API key: ").strip()
        if not api_key:
            exit("API key is required.")
    return api_key


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_image_media_type(image_path: str) -> str:
    ext = os.path.splitext(image_path)[1].lower()
    types = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
             ".gif": "image/gif", ".webp": "image/webp"}
    return types.get(ext, "image/jpeg")


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff")


def clean_path(path: str) -> str:
    """Strip surrounding quotes and extra whitespace from a path."""
    return path.strip().strip('"').strip("'").strip()


def looks_like_image(text: str) -> bool:
    """Check if the text looks like an image path or URL."""
    cleaned = clean_path(text)
    if cleaned.startswith(("http://", "https://")):
        return any(cleaned.lower().endswith(ext) for ext in IMAGE_EXTENSIONS)
    return os.path.splitext(cleaned)[1].lower() in IMAGE_EXTENSIONS


def build_user_content(user_text: str, image_source: str | None = None):
    if not image_source:
        return user_text

    content = []

    image_source = clean_path(image_source)

    if image_source.startswith(("http://", "https://")):
        content.append({
            "type": "image_url",
            "image_url": {"url": image_source},
        })
    elif os.path.isfile(image_source):
        media_type = get_image_media_type(image_source)
        b64 = encode_image(image_source)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{b64}"},
        })
    else:
        print(f"  ⚠ Could not find image: {image_source}\n")
        return user_text

    content.append({"type": "text", "text": user_text})
    return content


def get_vlm_response(client, image_source, question):
    MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
    
    system_msg = {
        "role": "system",
        "content": "You are a helpful Vision Language Model assistant. "
                   "You can analyze images and answer questions about them. "
                   "Be concise and informative."
    }

    user_content = build_user_content(question, image_source)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            system_msg,
            {"role": "user", "content": user_content},
        ],
        temperature=0.7,
        max_tokens=512,
    )
    return response.choices[0].message.content


def main():
    api_key = get_api_key()

    client = Groq(api_key=api_key)

    while True:
        try:
            user_input = input("You ► ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input or user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        # Parse  image_path_or_url | question   OR   plain question
        if "|" in user_input:
            image_part, text_part = user_input.split("|", 1)
            image_source = image_part.strip()
            question = text_part.strip() or "Describe this image in detail."
        elif looks_like_image(user_input):
            # User pasted just a path/URL with no question
            image_source = user_input
            question = "Describe this image in detail."
        else:
            image_source = None
            question = user_input

        user_content = build_user_content(question, image_source)

        try:
            result = get_vlm_response(client, image_source, question)
            print(f"AI  ► {result}\n")
        except Exception as e:
            print(f"  ⚠ Error: {e}\n")


if __name__ == "__main__":
    main()
