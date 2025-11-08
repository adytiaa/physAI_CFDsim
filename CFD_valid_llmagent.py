# CFD_valid_llmagent.py
import json, time

def make_prompt(report: dict):
    # brief prompt to LLM: include metrics and ask for verdict + recommended checks
    prompt = f"""
You are a CFD verification assistant. Given the following verification metrics, produce:
1) a short PASS/FAIL verdict,
2) the top 3 reasons if FAIL,
3) suggested next checks.
Metrics: {json.dumps(report['metrics'])}
Evidence IPFS: {report.get('evidence',{})}
"""
    return prompt

def call_llm(prompt: str) -> str:
    # placeholder: integrate with whichever LLM API you use (OpenAI, local Llama, etc.)
    # For demo, we construct a simple heuristic reply:
    if report_metrics_trouble(prompt):
        return "VERDICT: FAIL. Reason: high residuals. Next checks: increase collocation density, examine BCs, retrain with more data."
    return "VERDICT: PASS. Confidence high."

def report_metrics_trouble(prompt_text):
    # naive heuristic for offline demo: look for numeric tokens > threshold
    return "0.01" in prompt_text or "0.1" in prompt_text

if __name__ == "__main__":
    import argparse, sys
    parser = argparse.ArgumentParser()
    parser.add_argument("--report_ipfs", required=True)
    parser.add_argument("--job_hash", required=True)
    args = parser.parse_args()

    # fetch the report from IPFS (assumes local ipfs daemon)
    client = ipfshttpclient.connect(IPFS_API)
    blob = client.cat(args.report_ipfs).decode()
    report = json.loads(blob)

    prompt = make_prompt(report)
    # real implementation: call your LLM API here
    llm_reply = "VERDICT: PASS. No obvious issues detected. Confidence: 0.86"

    verdict_obj = {
      "job_hash": args.job_hash,
      "agent_type": "llm_agent",
      "llm_reply": llm_reply,
      "timestamp": time.time()
    }

    tmp = "/tmp/llm_verdict.json"
    Path(tmp).write_text(json.dumps(verdict_obj, indent=2))
    res = client.add(tmp)
    print("LLM verdict:", res["Hash"])
