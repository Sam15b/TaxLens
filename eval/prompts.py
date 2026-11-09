"""Eval prompts (docs/06-evaluation.md § 6.2)."""

JUDGE_PROMPT = """You grade answers about Indian tax law.
Question: {question}
Reference answer: {expected_answer}
Candidate answer: {answer}

Score 1 if the candidate states the same key facts as the reference (numbers, sections, conditions)
and nothing contradicting it. Score 0.5 if partially correct or missing an important condition.
Score 0 if wrong, contradicting, or not answering.
Return JSON: {{"score": 0|0.5|1, "reason": "<one sentence>"}}"""
