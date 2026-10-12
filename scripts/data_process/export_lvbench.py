import json
import os
import re
import argparse

VALID_TYPES = ['all', 'cartoon', 'documentary', 'live', 'selfmedia', 'sport', 'tv']

parser = argparse.ArgumentParser(description='Export LVBench data by video type')
parser.add_argument('category', choices=VALID_TYPES,
                    default='all', nargs='?',
                    help='Video type: all/cartoon/documentary/live/selfmedia/sport/tv')
args = parser.parse_args()

category_name = args.category

# Path configuration
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_dir = os.path.join(project_root, "data", "LVBench")
video_dir = os.path.join(data_dir, "videos")
annotation_file = os.path.join(data_dir, "video_info.meta.jsonl")
os.makedirs(os.path.join(data_dir, "test"), exist_ok=True)
output_file = os.path.join(data_dir, "test", f"{category_name}_data.jsonl")

# Read the data
with open(annotation_file, 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f if line.strip()]
print(f"Original video count: {len(data)}")

# Parse the options: extract question and options from the question text
def parse_question(raw_question):
    """Split the question and options from the 'question\\n(A) opt1\\n(B) opt2...' format"""
    match = re.split(r'\n\(A\)\s*', raw_question, maxsplit=1)
    if len(match) == 2:
        question_text = match[0].strip()
        options_raw = '(A) ' + match[1]
        options = re.findall(r'\(([A-Z])\)\s*(.+?)(?=\n\([A-Z]\)|$)', options_raw, re.DOTALL)
        options_list = [opt.strip() for _, opt in options]
        options_str = '\n'.join(f"({letter}) {opt.strip()}" for letter, opt in options)
        return question_text, options_list, options_str
    # Fallback: no standard option format found
    return raw_question.strip(), [], raw_question.strip()

# Filter by type
if category_name == 'all':
    filtered = data
else:
    filtered = [item for item in data if item['type'] == category_name]
print(f"Video count for type {category_name}: {len(filtered)}")

# Write JSONL (one record per QA pair)
count = 0
skipped_no_video = 0
with open(output_file, 'w', encoding='utf-8') as f:
    for item in filtered:
        video_key = item['key']
        video_path = os.path.join(video_dir, f"{video_key}.mp4")

        # Skip missing video files
        if not os.path.exists(video_path):
            skipped_no_video += len(item['qa'])
            continue

        video_info = item['video_info']

        for qa in item['qa']:
            question_text, options_list, options_str = parse_question(qa['question'])

            prompt = f"""Given a video: {video_path}
Question: {question_text}
Options:
{options_str}
Please answer the question based on the video and options. Give an analysis and a final answer with format: \nAnswer: X"""

            record = {
                'video_id': video_key,
                'video_path': video_path,
                'question_id': qa['uid'],
                'type': item['type'],
                'duration_minutes': video_info['duration_minutes'],
                'question': question_text,
                'options': options_list,
                'answer': qa['answer'],
                'question_type': qa['question_type'],
                'time_reference': qa.get('time_reference', ''),
                'prompt': prompt,
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            count += 1

print(f"\nData written to: {output_file}")
print(f"Wrote {count} QA records")
if skipped_no_video:
    print(f"Skipped {skipped_no_video} records (video files not found)")