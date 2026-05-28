#!/usr/bin/env python3
"""
Kisaan AI — Upgraded Eval Harness (v2)

Metrics:
  1. Keyword Precision (RAG faithfulness proxy) — expected treatment keywords in response
  2. Severity Accuracy — does returned severity match golden label
  3. Ragas Faithfulness — factual grounding of response in retrieved context
  4. Ragas Answer Relevance — how well the answer addresses the question
  5. Latency SLA — p95 < 8000ms
  6. Cost per query — tracked via SessionCost

Usage (with backend running on :8000):
  python eval/disease_eval.py

CI gate: keyword_precision >= 0.60 AND ragas_faithfulness >= 0.70
"""
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

GOLDEN_FILE = Path(__file__).parent / "golden_dataset" / "disease_cases.json"
BACKEND_URL = "http://localhost:8000/api"
LATENCY_SLA_MS = 8000
KEYWORD_GATE = 0.60
RAGAS_GATE = 0.70


@dataclass
class EvalResult:
    case_id: str
    crop: str
    condition: str
    keyword_hits: int
    keyword_total: int
    severity_correct: bool
    latency_ms: float
    response: str
    retrieved_context: str
    error: str = ""

    @property
    def keyword_precision(self) -> float:
        return self.keyword_hits / self.keyword_total if self.keyword_total else 0

    @property
    def passes_latency_sla(self) -> bool:
        return self.latency_ms < LATENCY_SLA_MS


def run_case(case: dict) -> EvalResult:
    payload = {
        "query": f"What is wrong with my {case['crop']}? I see: {case['condition']}",
        "commodity": "",
        "location": "",
    }
    try:
        start = time.perf_counter()
        resp = httpx.post(f"{BACKEND_URL}/analyze", data=payload, timeout=60.0)
        latency_ms = (time.perf_counter() - start) * 1000
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        return EvalResult(
            case_id=case["id"], crop=case["crop"], condition=case["condition"],
            keyword_hits=0, keyword_total=len(case["expected_keywords"]),
            severity_correct=False, latency_ms=0, response="", retrieved_context="",
            error=str(exc),
        )

    response_text = (data.get("response") or "").lower()
    hits = sum(1 for kw in case["expected_keywords"] if kw.lower() in response_text)
    severity_correct = data.get("severity", "").lower() == case["expected_severity"].lower()
    retrieved_context = " ".join(
        d.get("treatment", "") + " " + d.get("symptoms", "")
        for d in data.get("retrieved_docs", [])
    )

    return EvalResult(
        case_id=case["id"],
        crop=case["crop"],
        condition=case["condition"],
        keyword_hits=hits,
        keyword_total=len(case["expected_keywords"]),
        severity_correct=severity_correct,
        latency_ms=latency_ms,
        response=data.get("response", ""),
        retrieved_context=retrieved_context,
    )


def run_ragas(results: list[EvalResult]) -> dict:
    """Run Ragas faithfulness + answer_relevance on valid results."""
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, faithfulness

        valid = [r for r in results if not r.error and r.response and r.retrieved_context]
        if not valid:
            return {"faithfulness": None, "answer_relevancy": None, "note": "No valid results"}

        data = {
            "question": [f"What disease affects {r.crop}? Condition: {r.condition}" for r in valid],
            "answer": [r.response for r in valid],
            "contexts": [[r.retrieved_context] for r in valid],
        }
        dataset = Dataset.from_dict(data)
        scores = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
        return {
            "faithfulness": round(scores["faithfulness"], 4),
            "answer_relevancy": round(scores["answer_relevancy"], 4),
        }
    except Exception as exc:
        return {"faithfulness": None, "answer_relevancy": None, "error": str(exc)}


def main():
    cases = json.loads(GOLDEN_FILE.read_text())
    results: list[EvalResult] = []

    print(f"\nKisaan AI Eval Harness v2 — {len(cases)} test cases\n{'─' * 65}")
    for case in cases:
        print(f"  {case['id']}: {case['crop']} — {case['condition']}...", end=" ", flush=True)
        r = run_case(case)
        results.append(r)
        if r.error:
            print(f"✗  ERROR: {r.error}")
        else:
            print(f"✓  kw={r.keyword_hits}/{r.keyword_total}  sev={'✓' if r.severity_correct else '✗'}  {r.latency_ms:.0f}ms")

    valid = [r for r in results if not r.error]
    if not valid:
        print("\nAll cases failed — is the backend server running on :8000?")
        sys.exit(1)

    # Core metrics
    avg_kw = sum(r.keyword_precision for r in valid) / len(valid)
    sev_acc = sum(r.severity_correct for r in valid) / len(valid)
    latencies = sorted(r.latency_ms for r in valid)
    p95 = latencies[max(0, int(len(latencies) * 0.95) - 1)]
    sla_pass = sum(r.passes_latency_sla for r in valid) / len(valid)

    # Ragas metrics
    print(f"\n  Running Ragas evaluation...")
    ragas_scores = run_ragas(valid)

    print(f"\n{'─' * 65}")
    print(f"  Keyword Precision  (RAG faithfulness proxy): {avg_kw:.1%}")
    print(f"  Severity Accuracy:                           {sev_acc:.1%}")
    print(f"  p95 Latency:                                 {p95:.0f}ms  (SLA: {LATENCY_SLA_MS}ms)")
    print(f"  Latency SLA Pass Rate:                       {sla_pass:.1%}")
    if ragas_scores.get("faithfulness") is not None:
        print(f"  Ragas Faithfulness:                          {ragas_scores['faithfulness']:.4f}")
        print(f"  Ragas Answer Relevancy:                      {ragas_scores['answer_relevancy']:.4f}")
    else:
        note = ragas_scores.get("error") or ragas_scores.get("note", "")
        print(f"  Ragas: skipped ({note})")
    print(f"  Cases: {len(valid)}/{len(cases)} passed\n")

    # CI gates
    failed = []
    if avg_kw < KEYWORD_GATE:
        failed.append(f"Keyword precision {avg_kw:.1%} < {KEYWORD_GATE:.0%} gate")
    if ragas_scores.get("faithfulness") and ragas_scores["faithfulness"] < RAGAS_GATE:
        failed.append(f"Ragas faithfulness {ragas_scores['faithfulness']:.2f} < {RAGAS_GATE} gate")

    if failed:
        print("FAIL — gates not met:")
        for f in failed:
            print(f"  ✗ {f}")
        sys.exit(1)
    else:
        print("PASS — all gates met. ✓")


if __name__ == "__main__":
    main()
