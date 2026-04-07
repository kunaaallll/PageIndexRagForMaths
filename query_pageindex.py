import json
import asyncio
import textwrap
import os
import re
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pageindex import utils
import re

# =========================
# Load environment variables
# =========================
load_dotenv()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =========================
# Choose model
# =========================
MODEL_NAME = "gpt-4o-mini"   # You can change to gpt-4o if you want


# =========================
# Custom helper: create node mapping
# =========================
def create_node_mapping(tree):
    node_map = {}

    def traverse(node):
        if isinstance(node, dict):
            if "node_id" in node:
                node_map[node["node_id"]] = node

            for key, value in node.items():
                if isinstance(value, (dict, list)):
                    traverse(value)

        elif isinstance(node, list):
            for item in node:
                traverse(item)

    traverse(tree)
    return node_map


# =========================
# Custom helper: pretty print wrapped text
# =========================
def print_wrapped(text, width=100):
    for line in textwrap.wrap(str(text), width=width):
        print(line)


# =========================
# Helper: clean JSON output
# =========================
def extract_json(text):
    """
    Extract JSON safely from model output.
    Handles ```json ... ``` wrapping.
    """
    text = text.strip()
    text = re.sub(r"^```json", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^```", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    return text


# =========================
# Real OpenAI LLM call
# =========================
async def call_llm(prompt):
    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful document retrieval and QA assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    return response.choices[0].message.content

async def clean_math_text(raw_text, node_title=""):
    cleanup_prompt = f"""
You are cleaning OCR-extracted mathematical textbook text.

Your task:
- Repair broken equations and mathematical notation ONLY when strongly supported by the text
- Preserve the original meaning as faithfully as possible
- DO NOT invent or guess formulas if uncertain
- If a symbol/expression is unclear, keep it conservative rather than making up a mathematically precise-looking expression
- Keep formulas readable and student-friendly
- Do NOT explain or summarize
- Return only cleaned text

Context:
This text comes from a math textbook section titled: "{node_title}"

OCR Text:
{raw_text}
"""

    cleaned = await call_llm(cleanup_prompt)
    return cleaned.strip()


def make_answer_human_readable(text: str) -> str:
    """
    Convert LaTeX-ish math output into terminal / plain-text readable format.
    """

    # Remove block math wrappers
    text = text.replace("\\[", "").replace("\\]", "")
    text = text.replace("\\(", "").replace("\\)", "")

    # Common LaTeX replacements
    text = text.replace("\\frac{dy}{dx}", "dy/dx")
    text = text.replace("\\frac{dY}{dX}", "dY/dX")
    text = text.replace("\\equiv", "≡")
    text = text.replace("\\quad", " ")
    text = text.replace("\\,", " ")

    # Convert generic \frac{a}{b} -> a/b
    text = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", text)

    # Convert \left( and \right)
    text = text.replace("\\left(", "(").replace("\\right)", ")")
    text = text.replace("\\left[", "[").replace("\\right]", "]")
    text = text.replace("\\left{", "{").replace("\\right}", "}")

    # Remove \text{...}
    text = re.sub(r"\\text\{([^{}]+)\}", r"\1", text)

    # Collapse extra spaces
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)

    return text.strip()


def detect_question_type(query: str):
    q = query.lower().strip()

    if "formula" in q or "equation" in q:
        return "formula"
    elif q.startswith("what is") or "define" in q or "definition" in q:
        return "definition"
    elif "derive" in q or "derivation" in q or "prove" in q:
        return "derivation"
    elif "solve" in q or "find" in q:
        return "solve"
    elif "difference between" in q or "compare" in q:
        return "comparison"
    else:
        return "general"

async def main():
    # =========================
    # Load tree JSON
    # =========================
    with open(r"D:\AIAgents\class10ScienceBook\chp1\PageIndex\results\jeeMaths1_structure.json", "r", encoding="utf-8") as f:
        tree = json.load(f)

    # =========================
    # Build node map
    # =========================
    node_map = create_node_mapping(tree)

    print("\nAvailable Node IDs (first 20):")
    for i, key in enumerate(node_map.keys()):
        if i >= 20:
            break
        print(key)

    # Optional debug
    print("\nSample node structure:")
    first_node_id = next(iter(node_map.keys()))
    print(json.dumps(node_map[first_node_id], indent=2, ensure_ascii=False)[:1500])

    # =========================
    # User Query
    # =========================
    query = "what is NUMERICALLY GREATEST TERM OF BINOMIAL EXPANSION?explain with derivation."
    question_type = detect_question_type(query)
    print(f"\nDetected Question Type: {question_type}")

    # Remove full text before sending to LLM
    tree_without_text = utils.remove_fields(tree.copy(), fields=["text"])

    # =========================
    # Step 2: Tree Search
    # =========================
    search_prompt = f"""
You are given a question and a tree structure of a document.
Each node contains a node id, node title, and a corresponding summary.
Your task is to find ALL nodes that are likely to contain the answer to the question.

Question: {query}

Document tree structure:
{json.dumps(tree_without_text, indent=2, ensure_ascii=False)}

Please reply ONLY in the following JSON format:
{{
    "thinking": "<Your reasoning about which nodes are relevant>",
    "node_list": ["node_id_1", "node_id_2", ..., "node_id_n"]
}}

Rules:
- Return only valid JSON
- Do not include markdown
- Do not include explanations outside JSON
- node_list must contain only node IDs present in the tree
"""

    tree_search_result = await call_llm(search_prompt)
    tree_search_result = extract_json(tree_search_result)

    try:
        tree_search_result_json = json.loads(tree_search_result)
    except Exception as e:
        print("\n❌ Failed to parse model JSON response:")
        print(tree_search_result)
        raise e

    print("\nReasoning Process:")
    print_wrapped(tree_search_result_json.get("thinking", "No reasoning returned."))

    print("\nRetrieved Nodes:")
    node_list = tree_search_result_json.get("node_list", [])

    for node_id in node_list:
        if node_id in node_map:
            node = node_map[node_id]
            print(
                f"Node ID: {node.get('node_id', 'N/A')}\t "
                f"Page: {node.get('page_index', 'N/A')}\t "
                f"Title: {node.get('title', 'N/A')}"
            )
        else:
            print(f"⚠ Node ID {node_id} not found in tree.")

    # =========================
    # Step 3.1: Extract Relevant Content
    # =========================
    # relevant_chunks = []
    # for node_id in node_list:
    #     if node_id in node_map and "text" in node_map[node_id]:
    #         relevant_chunks.append(node_map[node_id]["text"])

    # relevant_content = "\n\n".join(relevant_chunks)

    # print("\nRetrieved Context:\n")
    # if relevant_content.strip():
    #     print_wrapped(relevant_content[:2000] + "...")
    # else:
    #     print("No text content found for the retrieved nodes.")

    # =========================
# Step 3.1: Extract and Clean Relevant Content
# =========================
    relevant_chunks = []

    for node_id in node_list:
        if node_id in node_map:
            node = node_map[node_id]

            title = node.get("title", "")
            summary = node.get("summary", "")
            raw_text = node.get("text", "")

            cleaned_text = ""
            if raw_text.strip():
                print(f"\nCleaning OCR math text for node {node_id}...")
                cleaned_text = await clean_math_text(raw_text, node_title=title)

            chunk_parts = []
            if title:
                chunk_parts.append(f"Title: {title}")
            if summary:
                chunk_parts.append(f"Summary: {summary}")
            if cleaned_text:
                chunk_parts.append(f"Cleaned Text: {cleaned_text}")

            if chunk_parts:
                relevant_chunks.append("\n\n".join(chunk_parts))

    relevant_content = "\n\n" + ("\n\n" + "="*80 + "\n\n").join(relevant_chunks)

    print("\nRetrieved + Cleaned Context:\n")
    if relevant_content.strip():
        print_wrapped(relevant_content[:3000] + "...")
    else:
        print("No content found for the retrieved nodes.")

    # =========================
    # Step 3.2: Generate Final Answer
    # =========================
    if question_type == "formula":
        answer_prompt = f"""
    Answer the question based ONLY on the context below.

    Question:
    {query}

    Context:
    {relevant_content}

    Instructions:
    - The user is asking ONLY for the formula / form
    - Give a SHORT, direct, student-friendly answer
    - Do NOT explain the full topic
    - Do NOT include unnecessary background or extra theory
    - Start with a short heading
    - Then write:
        1. One-line explanation
        2. The formula clearly
        3. If relevant, the substitution / method
    - Keep the answer compact and easy to revise
    - Use simple readable math notation like dy/dx, y/x, x/y
    - Avoid excessive LaTeX formatting
    """

    elif question_type == "definition":
        answer_prompt = f"""
    Answer the question based ONLY on the context below.

    Question:
    {query}

    Context:
    {relevant_content}

    Instructions:
    - Give a clear definition first
    - Then a short student-friendly explanation
    - Keep it concise and readable
    - Use line breaks
    - Avoid unnecessary extra theory
    """

    elif question_type == "derivation":
        answer_prompt = f"""
    Answer the question based ONLY on the context below.

    Question:
    {query}

    Context:
    {relevant_content}

    Instructions:
    - Explain step-by-step in a student-friendly way
    - Preserve formulas carefully
    - Use headings and line breaks
    - Make the derivation easy to follow
    """

    else:
        answer_prompt = f"""
    Answer the question based ONLY on the context below.

    Question:
    {query}

    Context:
    {relevant_content}

    Instructions:
    - Write the answer in a clean, human-readable, textbook-style format
    - Use short paragraphs and line breaks
    - Keep it concise but clear
    - Preserve formulas and notation correctly
    """

    print("\nGenerated Answer:\n")
    answer = await call_llm(answer_prompt)
    answer = make_answer_human_readable(answer)
    print_wrapped(answer)


if __name__ == "__main__":
    asyncio.run(main())






# import json
# import asyncio

# import pageindex.utils as utils
# import textwrap

# # =========================
# # Example async LLM function
# # (replace this with your real one)
# # =========================
# async def call_llm(prompt, tree=None, mode="search"):
#     if mode == "search" and tree is not None:
#         node_map = create_node_mapping(tree)
#         first_node_id = next(iter(node_map.keys()))  # safely picks first valid node id
#         return json.dumps({
#             "thinking": f"Temporary test reasoning. Returning first available node: {first_node_id}",
#             "node_list": [first_node_id]
#         })

#     elif mode == "answer":
#         return "This is a temporary fake answer based on the retrieved context."

#     return json.dumps({
#         "thinking": "Temporary fallback reasoning",
#         "node_list": []
#     })


# def print_wrapped(text, width=100):
#     for line in textwrap.wrap(str(text), width=width):
#         print(line)

# def create_node_mapping(tree):
#     node_map = {}

#     def traverse(node):
#         if isinstance(node, dict):
#             if "node_id" in node:
#                 node_map[node["node_id"]] = node

#             # Traverse possible child containers
#             for key, value in node.items():
#                 if isinstance(value, (dict, list)):
#                     traverse(value)

#         elif isinstance(node, list):
#             for item in node:
#                 traverse(item)

#     traverse(tree)
#     return node_map


# async def main():
#     # =========================
#     # Load tree JSON from file
#     # =========================
#     with open(r"D:\AIAgents\class10ScienceBook\chp1\PageIndex\results\RMT_Manual_structure.json", "r", encoding="utf-8") as f:
#         tree = json.load(f)

#     # =========================
#     # Step 2: Tree Search
#     # =========================
#     query = "what is isolator?"

#     tree_without_text = utils.remove_fields(tree.copy(), fields=['text'])

#     search_prompt = f"""
# You are given a question and a tree structure of a document.
# Each node contains a node id, node title, and a corresponding summary.
# Your task is to find all nodes that are likely to contain the answer to the question.

# Question: {query}

# Document tree structure:
# {json.dumps(tree_without_text, indent=2)}

# Please reply in the following JSON format:
# {{
#     "thinking": "<Your thinking process on which nodes are relevant to the question>",
#     "node_list": ["node_id_1", "node_id_2", ..., "node_id_n"]
# }}
# Directly return the final JSON structure. Do not output anything else.
# """

#     # tree_search_result = await call_llm(search_prompt)
#     tree_search_result = await call_llm(search_prompt, tree=tree, mode="search")
#     # =========================
#     # Step 2.2: Print retrieved nodes
#     # =========================
#     node_map = create_node_mapping(tree)
#     print("\nAvailable Node IDs (first 20):")
#     for i, key in enumerate(node_map.keys()):
#      if i >= 20:
#         break
#      print(key)

#     tree_search_result_json = json.loads(tree_search_result)

#     print("Reasoning Process:")
#     print_wrapped(tree_search_result_json["thinking"])

#     print("\nRetrieved Nodes:")
#     for node_id in tree_search_result_json["node_list"]:
#         node = node_map[node_id]
#         print(
#         f"Node ID: {node.get('node_id', 'N/A')}\t "
#         f"Page: {node.get('page_index', 'N/A')}\t "
#         f"Title: {node.get('title', 'N/A')}"
# )

#     # =========================
#     # Step 3.1: Extract relevant content
#     # =========================
#     node_list = tree_search_result_json["node_list"]
#     # relevant_content = "\n\n".join(node_map[node_id]["text"] for node_id in node_list)
#     relevant_chunks = []
#     for node_id in node_list:
#         if node_id in node_map and "text" in node_map[node_id]:
#             relevant_chunks.append(node_map[node_id]["text"])

#     relevant_content = "\n\n".join(relevant_chunks)
#     print("\nRetrieved Context:\n")
#     print_wrapped(relevant_content[:1000] + "...")

#     # =========================
#     # Step 3.2: Generate answer
#     # =========================
#     answer_prompt = f"""
# Answer the question based on the context:

# Question: {query}
# Context: {relevant_content}

# Provide a clear, concise answer based only on the context provided.
# """

#     print("\nGenerated Answer:\n")
#     # answer = await call_llm(answer_prompt)
#     answer = await call_llm(answer_prompt, mode="answer")
#     print_wrapped(answer)


# if __name__ == "__main__":
#     asyncio.run(main())














# from pageindex import page_index
# from pageindex.rag.pageindex_rag import pageindex_rag

# PDF_PATH = r"D:\AIAgents\class10ScienceBook\chp1\RMT_Manual.pdf"

# QUESTION = "What is the difference between physical and chemical change?"

# # Step 1: Build / reuse index
# index_result = page_index(
#     PDF_PATH,
#     model="gpt-4o-mini"
# )

# # Step 2: Ask question using RAG
# answer = pageindex_rag(
#     index_result=index_result,
#     question=QUESTION,
#     model="gpt-4o-mini"
# )

# print("\nAnswer:\n")
# print(answer)
