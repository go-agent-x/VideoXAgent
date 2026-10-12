import pandas as pd
import os
import json
import re

# Get the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_dir = os.path.join(project_root, "data", "Video-MME-v2")
video_dir = os.path.join(data_dir, "videos")
annotation_file = os.path.join(data_dir, "test.parquet")
os.makedirs(os.path.join(data_dir, "test"), exist_ok=True)
output_file = os.path.join(data_dir, "test", "test.jsonl")

# Read the data
df = pd.read_parquet(annotation_file)
print(f"Original record count: {df.shape[0]}")
print(f"Fields: {df.columns.tolist()}")

# Extract videoID from a YouTube URL
def extract_video_id(url):
    """Extract the video ID from a YouTube URL"""
    match = re.search(r'v=([a-zA-Z0-9_-]+)', url)
    if match:
        return match.group(1)
    return None

# Extract records and write them to a JSONL file
with open(output_file, 'w', encoding='utf-8') as f:
    for idx, (_, row) in enumerate(df.iterrows()):
        videoID = extract_video_id(row['url'])
        # Build the path using video_id, which has the format "001", "002", ...
        video_path = os.path.join(video_dir, f"{row['video_id']}.mp4")

        # Process the options field (originally a newline-delimited string)
        options_str = row['options']
        options_list = [opt.strip() for opt in options_str.split('\n') if opt.strip()]
        options_formatted = '\n'.join(options_list)

        prompt = f"""Given a video: {video_path}
Question: {row['question']}
Options:
{options_formatted}
Please answer the question based on the video and options. Give an analysis and a final answer with format: \nAnswer: X"""

        item = {
            'video_id': row['video_id'],
            'video_path': video_path,
            'video_url': row['url'],
            'videoID': videoID,
            'question_id': row['question_id'],
            'group_type': row['group_type'],
            'group_structure': row['group_structure'],
            'question': row['question'],
            'options': options_list,
            'answer': row['answer'],
            'level': row['level'] if pd.notna(row['level']) else None,
            'second_head': row['second_head'] if pd.notna(row['second_head']) else None,
            'third_head': row['third_head'] if pd.notna(row['third_head']) else None,
            'prompt': prompt
        }
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"\nData written to: {output_file}")
print(f"Wrote {df.shape[0]} records")
