"""Eval harness: runs every case in cases.json through the agent and scores it.

Usage (from the repo root):  python3 evals/run_evals.py [label]
Results are saved to evals/results/<label>.json so runs can be compared.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CASES_PATH = os.path.join(os.path.dirname(__file__), "cases.json")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def check_case(case: dict, answer: str, tools_used: list[str], source_titles: list[str],
               truncated: bool = False) -> list[str]:
    """Return a list of failure reasons. An empty list means the case passed."""
    failures = []
    answer_lower = answer.lower()

    if truncated:
        failures.append("answer was cut off (hit the max_tokens limit)")

    for tool in case.get("expect_tools", []):
        if tool not in tools_used:
            failures.append(f"expected tool '{tool}' was not called")
    for tool in case.get("forbid_tools", []):
        if tool in tools_used:
            failures.append(f"forbidden tool '{tool}' was called")

    any_of = case.get("must_contain_any")
    if any_of and not any(s.lower() in answer_lower for s in any_of):
        failures.append(f"answer contains none of {any_of}")
    for s in case.get("must_contain_all", []):
        if s.lower() not in answer_lower:
            failures.append(f"answer is missing required text '{s}'")

    max_words = case.get("max_words")
    if max_words and len(answer.split()) > max_words:
        failures.append(f"answer has {len(answer.split())} words, limit is {max_words}")
    for s in case.get("must_not_contain", []):
        if s.lower() in answer_lower:
            failures.append(f"answer contains forbidden text '{s}'")

    wanted_titles = case.get("expect_source_titles_any")
    if wanted_titles and not any(t in source_titles for t in wanted_titles):
        failures.append(f"none of {wanted_titles} found in sources {source_titles}")
    if wanted_titles and case.get("must_cite_source") and not any(t.lower() in answer_lower for t in wanted_titles):
        failures.append(f"answer does not name its source document {wanted_titles}")

    max_calls = case.get("max_tool_calls")
    if max_calls is not None and len(tools_used) > max_calls:
        failures.append(f"used {len(tools_used)} tool calls, limit is {max_calls}")

    return failures


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    if len(sys.argv) > 2:
        # Must be set before importing app, which reads it at import time.
        os.environ["PROMPT_VERSION"] = sys.argv[2]

    import app  # heavy import: loads the embedding model and indexes the PDFs
    print(f"Running with prompt version: {app.PROMPT_VERSION}")

    with open(CASES_PATH) as f:
        cases = json.load(f)

    results = []
    for case in cases:
        started = time.perf_counter()
        try:
            response = app.run_agent(case["query"])
            answer = response.answer
            tools_used = [c.name for c in response.tool_calls]
            titles = [s.title for s in response.sources]
            failures = check_case(case, answer, tools_used, titles, response.truncated)
        except Exception as e:
            answer, tools_used, titles, failures = "", [], [], [f"agent crashed: {e}"]
        seconds = round(time.perf_counter() - started, 1)

        results.append({
            "id": case["id"],
            "passed": not failures,
            "failures": failures,
            "tools_used": tools_used,
            "source_titles": titles,
            "seconds": seconds,
            "answer": answer,
        })
        print(f"{'PASS' if not failures else 'FAIL'}  {case['id']:<32} {seconds:>5}s  tools={tools_used}")
        for reason in failures:
            print(f"        - {reason}")

    passed = sum(r["passed"] for r in results)
    metrics = {
        "tool_calls": sum(len(r["tools_used"]) for r in results),
        "words": sum(len(r["answer"].split()) for r in results),
        "seconds": round(sum(r["seconds"] for r in results), 1),
    }
    print(f"\n{passed}/{len(results)} passed  (label: {label}, prompt: {app.PROMPT_VERSION})")
    print(f"metrics: {metrics['tool_calls']} tool calls, {metrics['words']} words, {metrics['seconds']}s total")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, f"{label}.json"), "w") as f:
        json.dump({"label": label, "prompt_version": app.PROMPT_VERSION, "passed": passed,
                   "total": len(results), "metrics": metrics, "results": results}, f, indent=2)

    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
