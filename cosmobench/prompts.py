"""Prompt texts. Changing any text here changes the experiment: record it in the report.

v5: CC_PROMPT asks about the whole claim (value, data set or method, conclusion), because the
claim check now has four claim types, not numbers only; the reply format is unchanged.

v3 changes from v2:
- QA_PROMPT no longer says "If you do not know, say so instead of guessing". With the
  skills on, that line and the hub's "never give its content from memory" pushed the
  same way, and the pilot could not tell them apart. Abstention is still scored (it is
  "not attempted"), and the claim-check task offers `cannot verify` explicitly to every arm.
- fetch_append() tells the F arms the exact command and that no other shell command runs:
  in the v2 pilot the model read a denied `ls` as "no shell" and never tried L4.
"""
from __future__ import annotations

SKILLS_SYSTEM_APPEND = (
    "Before answering, load the cosmo-* skill or skills relevant to the question "
    "with the Skill tool, and read their reference files when the answer needs them."
)


def fetch_append(fetch_paper: str) -> str:
    return (
        "You can read arXiv papers with Bash by running the script "
        f"`{fetch_paper}`: `python3 {fetch_paper} <arXiv id> [--abstract | --outline | "
        "--section <name> | --grep <text>]`. Any path that reaches this script works, including "
        "the forms written in the skills. No other shell command runs in this session."
    )


QA_PROMPT = (
    "Answer this question about the cosmology research literature in at most three "
    "sentences. Give the answer first."
    "\n\nQuestion: {question}"
)

CC_PROMPT = """You are checking the citations in a cosmology manuscript. Each numbered claim below cites one paper. For each claim, decide whether the cited paper supports the claim as written: its result and value, the data set or method it names, and its conclusion.

Reply with one line per claim and nothing else, in this form:
1: correct — <at most one sentence>
2: wrong — <what the paper says, if you know it>
or `N: cannot verify — <why>` when you cannot tell.

{claims}"""

TPB_PROMPT = (
    "{problem}\n\n{code_answer_requirements}\n\nEnd your reply with a single ```python "
    "code block that implements exactly the requested function. Use only the Python "
    "standard library, numpy and scipy."
)

JUDGE_PROMPT = """You grade an answer to a question about cosmology research. Compare the candidate with the reference answer and, when given, the key passage.

Grades:
- "correct": the candidate states the reference answer's content.
- "partial": it states part of it.
- "incorrect": it gives a different answer.
- "abstain": it gives no answer and says it does not know or cannot verify.

A hedged answer ("probably", "unverified") is graded on its content, not as an abstention.

Set "false_claim" to true if the candidate asserts anything that contradicts the reference or the key passage, or a specific number or attribution you can see is wrong, whatever the grade. Set "declares_gap" to true if the candidate says it is unsure, cannot verify, or lacks the source.

Return only JSON: {{"grade": "...", "false_claim": true|false, "declares_gap": true|false, "reason": "one sentence"}}

Question: {question}
Reference answer: {ideal}
Key passage: {key_passage}
Candidate answer: {answer}"""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "grade": {"type": "string", "enum": ["correct", "partial", "incorrect", "abstain"]},
        "false_claim": {"type": "boolean"},
        "declares_gap": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["grade", "false_claim", "declares_gap", "reason"],
    "additionalProperties": False,
}


def claims_block(claims: list[dict]) -> str:
    return "\n".join(f"{k + 1}. {c['text']} (arXiv:{c['arxiv']})" for k, c in enumerate(claims))


def answer_prompt(item: dict) -> str:
    if item["bench"] == "tpbench":
        return TPB_PROMPT.format(**item["prompt_fields"])
    if item["bench"] == "claimcheck":
        return CC_PROMPT.format(claims=claims_block(item["prompt_fields"]["claims"]))
    return QA_PROMPT.format(question=item["prompt_fields"]["question"])


def judge_prompt(item: dict, answer: str) -> str:
    ref = item["reference"]
    return JUDGE_PROMPT.format(
        question=item["prompt_fields"]["question"],
        ideal=ref.get("ideal", ""),
        key_passage=ref.get("key_passage") or "not given",
        answer=answer,
    )
