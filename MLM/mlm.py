import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


def get_api_key():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Get a FREE API key at: https://console.groq.com/keys\n")
        api_key = input("Paste your Groq API key: ").strip()
        if not api_key:
            exit("API key is required.")
    return api_key


def get_chain(api_key):
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=api_key,
        temperature=0.7,
        max_tokens=1024,
    )

    chain = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant powered by a Medium Language Model. "
                   "You provide balanced, high-quality, and nuanced responses. "
                   "Be informative and comprehensive."),
        ("human", "{input}")
    ]) | llm | StrOutputParser()
    
    return chain


def main():
    api_key = get_api_key()

    chain = get_chain(api_key)

    print('Type your question. Type "quit" to exit.\n')

    while True:
        try:
            user_input = input("You ► ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input or user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        result = chain.invoke({"input": user_input})
        print(f"AI  ► {result}\n")


if __name__ == "__main__":
    main()
