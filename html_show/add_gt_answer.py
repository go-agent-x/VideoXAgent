#!/usr/bin/env python3
"""
Match ground truth answers from videomme data files and add them to conversations/all_data_path.json.
Also parse model responses and check whether the answers are correct.
"""

import argparse
import json
import os
import re

def load_jsonl(filepath):
    """Load a JSONL file and return a list of records."""
    data = []
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    return data

def extract_video_and_question(query):
    """Extract the video path and question from init_user_query."""
    if not query:
        return None, None
    
    # Normalize line breaks: replace literal \\n sequences with actual newlines
    query = query.replace('\\n', '\n')
    
    # Try matching multiple video reference formats
    # 1. "Given a video: xxx.mp4" format (VideoMME/LVBench/LongVideoBench)
    video_match = re.search(r'Given a video:\s*([^\n]+\.mp4)', query)
    # 2. "Given a youtube video ID: xxx" format (Minerva)
    if not video_match:
        video_match = re.search(r'Given a youtube video ID:\s*([^\n]+)', query)
    
    question_match = re.search(r'Question:\s*([^\n]+)', query)
    
    video_path = video_match.group(1).strip() if video_match else None
    question = question_match.group(1).strip() if question_match else None
    
    return video_path, question

def normalize_video_path(path):
    """Normalize a video path for matching."""
    if not path:
        return None
    # Extract the video filename before removing the extension
    basename = os.path.basename(path)
    # Remove the file extension, such as .mp4
    name_without_ext = os.path.splitext(basename)[0]
    return name_without_ext

def parse_session_jsonl(filepath):
    """Parse a session JSONL file and extract the final model response."""
    if not os.path.exists(filepath):
        return None
    
    entries = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    
    # Search backward for the last assistant text response
    for entry in reversed(entries):
        if entry.get('type') == 'assistant':
            message = entry.get('message', {})
            content = message.get('content', '')
            if isinstance(content, str) and content.strip():
                return content
            if isinstance(content, list):
                for part in content:
                    if part.get('type') == 'text':
                        text = part.get('text', '')
                        if text.strip():
                            return text
    return None

def extract_model_answer(text):
    """Extract an answer in Answer: X format from the model response."""
    if not text:
        return None
    
    # Match letter answers in multiple formats, ordered by priority (A-E, supporting five options)
    patterns = [
        # 1. Explicit answer formats, with parentheses (Answer: (B)) or without (Answer: B)
        r'\*?\*?Answer\*?\*?\s*[:：]\s*\*?\*?\(?([A-Ea-e])\)?\*?\*?',  # Answer: A / Answer: (A) / **Answer: (A)**
        r'\u7b54\u6848\s*[:：]\s*\(?([A-Ea-e])\)?',  # Chinese answer label followed by a letter, with optional parentheses
        r'[Ff]inal [Aa]nswer\s*[:：]\s*\(?([A-Ea-e])\)?',  # Final Answer: A or Final Answer: (A)
        r'\u9009\u62e9\s*[:：]?\s*\(?([A-Ea-e])\)?',  # Chinese selection label followed by a letter, with optional colon or parentheses
        # 2. Markdown bold option formats: **A. xxx** or **A**
        r'\*\*([A-Ea-e])[\.\s]',  # **A. or **A
        # 3. Option at the start of a line (possibly following the analysis)
        r'(?:^|\n)\s*\*?\*?([A-Ea-e])[\.\s]\s*[A-Z]',  # A. Xxx or **A. Xxx
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.MULTILINE)
        if matches:
            # Use the last matching answer
            return matches[-1].upper()
    
    # Numeric answer formats (datasets such as Minerva): Answer: 0/1/2/3/4
    num_patterns = [
        r'\*?\*?Answer\*?\*?\s*[:：]\s*\*?\*?\(?([0-4])\)?\*?\*?',  # Answer: 3 / Answer: (3)
        r'\u7b54\u6848\s*[:：]\s*\(?([0-4])\)?',  # Chinese answer label followed by a digit
        r'[Ff]inal [Aa]nswer\s*[:：]\s*\(?([0-4])\)?',  # Final Answer: 3
    ]
    
    for pattern in num_patterns:
        matches = re.findall(pattern, text, re.MULTILINE)
        if matches:
            num = int(matches[-1])
            return chr(ord('A') + num)  # 0→A, 1→B, 2→C, 3→D, 4→E
    
    return None

def compare_answers(model_answer, gt_answer):
    """Check whether the model answer matches the ground truth answer."""
    if not model_answer or not gt_answer:
        return None
    return model_answer.upper() == gt_answer.upper()

def find_matching_item(query, all_qa_data):
    """Find the complete matching record in the QA data."""
    if not query:
        return None
    
    video_path, question = extract_video_and_question(query)
    if not video_path:
        return None
    
    video_id = normalize_video_path(video_path)
    
    for item in all_qa_data:
        # Try multiple matching methods
        item_video_id = normalize_video_path(item.get('video_path', ''))
        item_videoID = item.get('videoID', '')  # videoID field used by Minerva and other datasets
        
        # Match the video by filename or videoID
        video_matched = (item_video_id == video_id) or (item_videoID == video_id)
        
        if video_matched:
            # Then match the question after trimming whitespace
            item_question = item.get('question', '').strip()
            if question and item_question:
                # Compare directly after trimming leading and trailing whitespace
                if question.strip() == item_question:
                    return item
                # Also check whether the query contains the full prompt
                item_prompt = item.get('prompt', '')
                if item_prompt and item_prompt.strip() in query.replace('\\n', '\n'):
                    return item
    
    return None

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Match ground truth answers from videomme/LVBench/LongVideoBench data and add them to a JSON file')
    parser.add_argument('json_file', nargs='?', default='conversations/all_data_path.json',
                        help='Path to the JSON file to process (default: conversations/all_data_path.json)')
    args = parser.parse_args()
    
    # File path
    all_data_path = args.json_file
    
    # Supported dataset directories
    dataset_dirs = [
        ('VideoMME', 'data/videomme/test'),
        ('LVBench', 'data/LVBench/test'),
        ('LongVideoBench', 'data/LongVideoBench/test'),
        ('Minerva', 'data/minerva/test'),
    ]
    
    # Load all QA data from every JSONL file in the supported dataset directories
    print("Loading QA data...")
    all_qa_data = []
    for dataset_name, dataset_dir in dataset_dirs:
        if os.path.isdir(dataset_dir):
            jsonl_files = sorted([f for f in os.listdir(dataset_dir) if f.endswith('.jsonl')])
            dataset_count = 0
            for fname in jsonl_files:
                fpath = os.path.join(dataset_dir, fname)
                items = load_jsonl(fpath)
                if items:
                    dataset_count += len(items)
                    all_qa_data.extend(items)
            if dataset_count > 0:
                print(f"  [{dataset_name}] Loaded {dataset_count} items from {len(jsonl_files)} files")
        else:
            print(f"  [{dataset_name}] Directory not found: {dataset_dir}")
    print(f"Total: {len(all_qa_data)} QA items")
    
    # Load all_data_path.json
    print(f"Loading {all_data_path}...")
    with open(all_data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Process each session
    matched_count = 0
    for session in data.get('sessions', []):
        init_query = session.get('init_user_query')
        matched_item = find_matching_item(init_query, all_qa_data)
        
        if matched_item:
            # Add all metadata fields
            session['gt_answer'] = matched_item.get('answer')
            session['video_url'] = matched_item.get('video_url')
            session['video_id'] = matched_item.get('video_id')
            session['videoID'] = matched_item.get('videoID')
            session['question_id'] = matched_item.get('question_id')
            session['duration'] = matched_item.get('duration')
            session['domain'] = matched_item.get('domain')
            session['sub_category'] = matched_item.get('sub_category')
            session['task_type'] = matched_item.get('task_type')
            session['options'] = matched_item.get('options')
            
            # Parse the session JSONL file to extract the model answer
            transcript_path = session.get('transcript_path')
            if transcript_path:
                model_output = parse_session_jsonl(transcript_path)
                model_answer = extract_model_answer(model_output)
                session['model_answer'] = model_answer
                
                # Compare the answers
                gt_answer = matched_item.get('answer')
                is_correct = compare_answers(model_answer, gt_answer)
                session['is_correct'] = is_correct
                
                correct_str = "✓" if is_correct else ("✗" if is_correct is False else "?")
                print(f"  Matched session {session.get('session_id', 'unknown')}: GT={gt_answer}, Model={model_answer}, {correct_str}")
            else:
                print(f"  Matched session {session.get('session_id', 'unknown')}: GT={matched_item.get('answer')}, no transcript")
            
            matched_count += 1
    
    # Save the updated file
    print(f"\nSaving updated file...")
    with open(all_data_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    # Compute accuracy
    correct_count = sum(1 for s in data.get('sessions', []) if s.get('is_correct') is True)
    incorrect_count = sum(1 for s in data.get('sessions', []) if s.get('is_correct') is False)
    total_evaluated = correct_count + incorrect_count
    
    print(f"\nDone! Matched {matched_count} sessions with ground truth answers.")
    if total_evaluated > 0:
        accuracy = correct_count / total_evaluated * 100
        print(f"Accuracy: {correct_count}/{total_evaluated} = {accuracy:.2f}%")

if __name__ == '__main__':
    main()
