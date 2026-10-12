#!/usr/bin/env python3
"""
Video-MME-v2 evaluation script: extract model responses from agent inference traces,
match questions in test.jsonl, and compute metrics using the official evaluate function.

Usage:
    python scripts/eval_videomme_v2.py \
        --sessions-file conversations/all_data_path_xxx.json \
        --data-file data/Video-MME-v2/test/test.jsonl \
        --output results/videomme_v2_eval.tsv
"""

import argparse
import json
import os
import re
import sys

import pandas as pd

# ---------------------------------------------------------------------------
# Reuse helper functions from html_show/add_gt_answer.py
# ---------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "html_show"))
from add_gt_answer import (
    extract_video_and_question,
    find_matching_item,
    normalize_video_path,
    parse_session_jsonl,
)

# ---------------------------------------------------------------------------
# Reuse evaluation functions from test_video_mme_v2.py
# ---------------------------------------------------------------------------
EVAL_DIR = os.path.join(
    ROOT_DIR, "scripts"
)
sys.path.insert(0, EVAL_DIR)
from test_video_mme_v2 import evaluate, extract_characters_regex_v2

# ---------------------------------------------------------------------------
# Answer extraction for agent output (overrides the original extract_characters_regex_v2)
# The original function takes the first A-H letter, which can incorrectly match the first letter of words such as "Analysis" or "According".
# Agent output usually ends with "Answer: X", so prioritize structured answer patterns.
# ---------------------------------------------------------------------------
_ANSWER_PATTERN = re.compile(
    r'(?:Final\s+Answer|Answer|Best\s+option|The\s+answer\s+is|'
    r'The\s+correct\s+answer\s+is|The\s+best\s+answer\s+is)'
    r'\s*[:：]?\s*\(?([A-H])\)?',
    re.IGNORECASE,
)


def extract_answer_for_agent(s):
    """Extract an answer letter A-H from a long agent response.

    Strategy:
    1. Prioritize structured patterns such as "Answer: X" (use the last match).
    2. Try the last A-H letter in the final 50 characters, then the original extract_characters_regex_v2.
    """
    if not s or not s.strip():
        return ''
    matches = _ANSWER_PATTERN.findall(s)
    if matches:
        return matches[-1].upper()
    # Fallback: take the last A-H letter in the final 50 characters
    tail = s.strip()[-50:]
    m = re.findall(r'[A-H]', tail)
    if m:
        return m[-1]
    return extract_characters_regex_v2(s)


# Monkey-patch evaluate to use the extraction logic adapted for agent output
import test_video_mme_v2
test_video_mme_v2.extract_characters_regex_v2 = extract_answer_for_agent


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def load_jsonl(filepath):
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def build_index(qa_data):
    """Build a (video_name, question) -> item index for faster matching."""
    index = {}
    for item in qa_data:
        vid = normalize_video_path(item.get("video_path", ""))
        q = item.get("question", "").strip()
        if vid and q:
            index[(vid, q)] = item
    return index


def fast_find(query, index, qa_data):
    """Try an exact index match first, then fall back to scanning the records."""
    video_path, question = extract_video_and_question(query)
    if video_path and question:
        vid = normalize_video_path(video_path)
        key = (vid, question.strip())
        if key in index:
            return index[key]
    # fallback
    return find_matching_item(query, qa_data)


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Video-MME-v2 evaluation script")
    parser.add_argument(
        "--sessions-file",
        required=True,
        help="Agent inference results index file, e.g., conversations/all_data_path_xxx.json",
    )
    parser.add_argument(
        "--data-file",
        default="data/Video-MME-v2/test/test.jsonl",
        help="Path to Video-MME-v2 test.jsonl",
    )
    parser.add_argument(
        "--output",
        default="results/videomme_v2_eval.tsv",
        help="Output path for evaluation results (.tsv)",
    )
    args = parser.parse_args()

    # 1. Load test.jsonl and build the index
    print(f"Loading data file: {args.data_file}")
    qa_data = load_jsonl(args.data_file)
    print(f"  Loaded {len(qa_data)} items")
    index = build_index(qa_data)

    # 2. Load the sessions file
    print(f"Loading sessions file: {args.sessions_file}")
    with open(args.sessions_file, "r", encoding="utf-8") as f:
        sessions_data = json.load(f)
    sessions = sessions_data.get("sessions", [])
    print(f"  Found {len(sessions)} sessions")

    # 3. Iterate over sessions, match questions, and extract model responses
    # Build a question_id -> prediction mapping
    predictions = {}
    matched = 0
    for session in sessions:
        init_query = session.get("init_user_query", "")
        item = fast_find(init_query, index, qa_data)
        if item is None:
            continue
        matched += 1

        # Extract the final model response
        transcript_path = session.get("transcript_path", "")
        prediction = ""
        if transcript_path and os.path.exists(transcript_path):
            model_output = parse_session_jsonl(transcript_path)
            if model_output:
                prediction = model_output

        qid = item.get("question_id", "")
        predictions[qid] = prediction

    print(f"\nMatched {matched} / {len(sessions)} sessions to test data")

    if not predictions:
        print("No matched records, nothing to evaluate.")
        return

    # 4. Build a DataFrame from the full qa_data, preserving the original order and complete groups
    #    get_final_rating requires groups of 4 records; use empty predictions for missing sessions
    records = []
    for item in qa_data:
        qid = item.get("question_id", "")
        records.append(
            {
                "question_id": qid,
                "answer": item.get("answer", ""),
                "prediction": predictions.get(qid, ""),
                "level": item.get("level"),
                "group_type": item.get("group_type", ""),
                "group_structure": item.get("group_structure", ""),
                "second_head": item.get("second_head"),
                "third_head": item.get("third_head"),
            }
        )
    df = pd.DataFrame(records)

    # Ensure the output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 5. Call the official evaluate function
    print(f"\n── Evaluation ({len(df)} questions) ──")
    evaluate(df, args.output)


if __name__ == "__main__":
    main()
