import json
import os
import argparse

# short/medium/long categories aligned with VideoMME
# short: ≤60s (duration_group 15, 60)
# medium: 3-10min (duration_group 600)
# long: 15-60min (duration_group 3600)
CATEGORY_TO_GROUPS = {
    'short': [15, 60],
    'medium': [600],
    'long': [3600],
}

CHOICE_LETTERS = ['A', 'B', 'C', 'D', 'E']

parser = argparse.ArgumentParser(description='Export LongVideoBench data for the specified duration category')
parser.add_argument('category', choices=['short', 'medium', 'long'],
                    default='medium', nargs='?',
                    help='Duration category: short/medium/long (aligned with VideoMME)')
args = parser.parse_args()

category_name = args.category

# Path configuration
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_dir = os.path.join(project_root, "data", "LongVideoBench")
video_dir = os.path.join(data_dir, "videos")
annotation_file = os.path.join(data_dir, "lvb_val.json")
os.makedirs(os.path.join(data_dir, "test"), exist_ok=True)
output_file = os.path.join(data_dir, "test", f"{category_name}_duration_data.jsonl")

# Read the data
with open(annotation_file, 'r', encoding='utf-8') as f:
    data = json.load(f)
print(f"Original record count: {len(data)}")

# Filter by category
target_groups = CATEGORY_TO_GROUPS[category_name]
filtered = [item for item in data if item['duration_group'] in target_groups]
print(f"Record count for duration={category_name}: {len(filtered)} (duration_group in {target_groups})")

# Write JSONL
with open(output_file, 'w', encoding='utf-8') as f:
    for item in filtered:
        video_path = os.path.join(video_dir, item['video_path'])

        # Build the option text (A/B/C/D/E)
        candidates = item['candidates']
        options_lines = []
        for i, opt in enumerate(candidates):
            options_lines.append(f"{CHOICE_LETTERS[i]}. {opt}")
        options_str = '\n'.join(options_lines)

        # Letter of the correct answer
        answer_letter = CHOICE_LETTERS[item['correct_choice']]

        prompt = f"""Given a video: {video_path}
Question: {item['question']}
Options:
{options_str}
Please answer the question based on the video and options. Give an analysis and a final answer with format: \nAnswer: X"""

        record = {
            'video_id': item['video_id'],
            'question_id': item['id'],
            'video_path': video_path,
            'duration': item['duration'],
            'duration_group': item['duration_group'],
            'question_category': item['question_category'],
            'topic_category': item['topic_category'],
            'level': item['level'],
            'question': item['question'],
            'options': candidates,
            'answer': answer_letter,
            'prompt': prompt,
        }
        f.write(json.dumps(record, ensure_ascii=False) + '\n')

print(f"\nData written to: {output_file}")
print(f"Wrote {len(filtered)} records")