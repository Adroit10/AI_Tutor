from __future__ import annotations
from rag.retriever          import retrieve
from llm.tutor_model         import generate_answer, generate_quiz, generate_followup, generate_diagram_prompt
from llm.diagram_generator  import generate_diagram


def run_retrieval(query: str, use_hyde: bool = True) -> str:
    print("\n[1/5] Retrieving context (HyDE + BM25 + MMR)…")
    context = retrieve(query, use_hyde=use_hyde)
    print(f"      ✔ {len(context)} chars retrieved")
    return context


def run_answer(query: str, context: str, level: str) -> str:
    print("\n[2/5] Generating answer (Groq Llama-3.3-70B)…")
    answer = generate_answer(query, context, level=level)
    print("      ✔ Answer ready")
    return answer


def run_quiz(query: str, context: str) -> str:
    print("\n[3/5] Generating quiz…")
    quiz = generate_quiz(query, context)
    print("      ✔ Quiz ready")
    return quiz


def run_followup(query: str, answer: str) -> str:
    print("\n[4/5] Generating follow-up questions…")
    followups = generate_followup(query, answer)
    print("      ✔ Follow-ups ready")
    return followups


def run_diagram(query: str, context: str, level: str) -> str:
    print("\n[5/5] Generating diagram (Mermaid → Kroki.io SVG)…")
   
    mermaid_def = generate_diagram_prompt(query, context, level)
    print(f"      Mermaid definition:\n{mermaid_def[:300]}…")
   
    image_path  = generate_diagram(
        query,
        mermaid_definition=mermaid_def,
        context=context,
        level=level,
    )
    print(f"      ✔ Diagram saved → {image_path}")
    return image_path


def run_tutor_pipeline(
    query:           str  = "Explain Gradient Descent",
    level:           str  = "beginner",
    include_quiz:    bool = True,
    include_followup: bool = True,
    include_diagram: bool = True,
    use_hyde:        bool = True,
) -> dict:

    sep = "=" * 55
    print(f"\n{sep}")
    print(f"  AI Tutor Pipeline  (upgraded)")
    print(f"  Query : {query}")
    print(f"  Level : {level}")
    print(f"{sep}")

    result = {"query": query, "level": level}

    result["context"] = run_retrieval(query, use_hyde=use_hyde)
    result["answer"]  = run_answer(query, result["context"], level)

    if include_quiz:
        result["quiz"]      = run_quiz(query, result["context"])

    if include_followup:
        result["followups"] = run_followup(query, result["answer"])

    if include_diagram:
        result["diagram_path"] = run_diagram(query, result["context"], level)

    print(f"\n{sep}")
    print("  Pipeline complete.")
    print(f"{sep}\n")
    return result


def print_results(result: dict) -> None:
    sep = "=" * 55
    print(f"\n{sep}")
    print(f"QUERY  : {result['query']}")
    print(f"LEVEL  : {result['level']}")

    print("\n--- ANSWER " + "-" * 44)
    print(result["answer"])

    if "quiz" in result:
        print("\n--- QUIZ " + "-" * 46)
        print(result["quiz"])

    if "followups" in result:
        print("\n--- FOLLOW-UP QUESTIONS " + "-" * 31)
        print(result["followups"])

    if "diagram_path" in result:
        print("\n--- DIAGRAM " + "-" * 43)
        print(f"Saved to: {result['diagram_path']}")

    print(f"{sep}\n")

if __name__ == "__main__":
    result = run_tutor_pipeline(
        query="Explain Gradient Descent",
        level="beginner",
        include_quiz=True,
        include_followup=True,
        include_diagram=True,
        use_hyde=True,      
    )
    print_results(result)