import pandas as pd
import os
import json
import argparse

parser = argparse.ArgumentParser(description='Export video data for the specified duration category')
parser.add_argument('category', choices=['short', 'medium', 'long'], 
                    default='medium', nargs='?', help='Duration category: short/medium/long')
args = parser.parse_args()

category_name = args.category

# Get the script directory
# script_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_dir = os.path.join(project_root, "data", "videomme")
video_dir = os.path.join(data_dir, "videos")
annotation_file = os.path.join(data_dir, "test-00000-of-00001.parquet")
os.makedirs(os.path.join(data_dir, "test"), exist_ok=True)
output_file = os.path.join(data_dir, "test", f"{category_name}_duration_data.jsonl")

# Read the data
df = pd.read_parquet(annotation_file)
print(f"Original record count: {df.shape[0]}")

# Filter records by duration category
df_long = df[df['duration'] == category_name]
print(f"Record count for duration={category_name}: {df_long.shape[0]}")

# Extract records and write them to a JSONL file
with open(output_file, 'w', encoding='utf-8') as f:
    for _, row in df_long.iterrows():
        videoID = row['videoID']
        video_path = os.path.join(video_dir, f"{videoID}.mp4")
        
        options_list = row['options'].tolist() if hasattr(row['options'], 'tolist') else row['options']
        options_str = '\n'.join(options_list)
        
        prompt = f"""Given a video: {video_path}
Question: {row['question']}
Options:
{options_str} 
Please answer the question based on the video and options. Give an analysis and a final answer with format: \nAnswer: X"""
        
        item = {
            'video_id': row['video_id'],
            'video_path': video_path,
            'video_url': row['url'],  # YouTube URL
            'videoID': row['videoID'],
            'question_id': row['question_id'],
            'duration': row['duration'],
            'domain': row['domain'],
            'sub_category': row['sub_category'],
            'task_type': row['task_type'],
            'question': row['question'],
            'options': options_list,
            'answer': row['answer'],
            'prompt': prompt
        }
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"\nData written to: {output_file}")
print(f"Wrote {df_long.shape[0]} records")