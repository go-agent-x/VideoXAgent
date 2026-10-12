"""Export MINERVA questions for already prepared local videos."""

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile


STRING_FIELDS = (
    'key', 'video_id', 'question', 'answer', 'question_type', 'split',
    'category', 'reasoning', *(f'answer_choice_{i}' for i in range(5)),
)


def load_annotations(input_file):
    if not input_file.is_file():
        raise ValueError(f"Missing MINERVA annotation file: {input_file}")
    try:
        with input_file.open('r', encoding='utf-8') as handle:
            data = json.load(handle)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"Invalid MINERVA JSON in {input_file}: {exc}") from exc
    if not isinstance(data, list):
        raise ValueError(f"MINERVA annotations must be a JSON array: {input_file}")
    for index, row in enumerate(data, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"MINERVA record {index} must be a JSON object")
        for field in STRING_FIELDS:
            if field not in row or not isinstance(row[field], str):
                raise ValueError(
                    f"MINERVA record {index}: '{field}' must be a string"
                )
        if not row['video_id'].strip():
            raise ValueError(f"MINERVA record {index}: 'video_id' must not be empty")
        if type(row.get('answer_id')) is not int or not 0 <= row['answer_id'] <= 4:
            raise ValueError(f"MINERVA record {index}: 'answer_id' must be an integer from 0 to 4")
    return data


def validate_videos(data, video_dir):
    missing = sorted({
        video_dir / f"{row['video_id']}.mp4"
        for row in data
        if not (video_dir / f"{row['video_id']}.mp4").is_file()
    })
    if missing:
        paths = '\n'.join(f"  {path}" for path in missing)
        raise ValueError(
            f"Missing {len(missing)} local video file(s) required by the selected "
            f"MINERVA records:\n{paths}\n"
            "Prepare these videos before exporting. No output was written."
        )


def make_item(row, index, video_dir):
    video_id = row['video_id']
    video_path = str(video_dir / f"{video_id}.mp4")
    # Preserve the existing option labels and post-filter question numbering.
    options_list = [f"{i}. {row[f'answer_choice_{i}']}" for i in range(5)]
    options_str = '\n'.join(options_list)
    answer_letter = chr(ord('A') + row['answer_id'])
    prompt = f"""Given a video: {video_path}
Question: {row['question']}
Options:
{options_str}
Please answer the question based on the video and options. Give an analysis and a final answer with format: \nAnswer: X"""
    return {
        'key': row['key'],
        'video_id': video_id,
        'video_path': video_path,
        'video_url': f"https://www.youtube.com/watch?v={video_id}",
        'videoID': video_id,
        'question_id': f"{video_id}-{index}",
        'duration': row['split'],
        'question': row['question'],
        'options': options_list,
        'answer_id': row['answer_id'],
        'answer_letter': answer_letter,
        'answer': answer_letter,
        'answer_text': row['answer'],
        'question_type': row['question_type'],
        'sub_category': row['category'],
        'task_type': row['question_type'],
        'split': row['split'],
        'category': row['category'],
        'reasoning': row['reasoning'],
        'prompt': prompt,
    }


def write_output(data, output_file, video_dir):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w', encoding='utf-8', dir=output_file.parent,
            prefix=f'.{output_file.name}.', suffix='.tmp', delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            for index, row in enumerate(data, start=1):
                item = make_item(row, index, video_dir)
                handle.write(json.dumps(item, ensure_ascii=False) + '\n')
        os.replace(temporary_path, output_file)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Convert MINERVA JSON data to JSONL format')
    parser.add_argument('--split', choices=['Sports and Board Games', 'VLOG Style', 'STEM', 'Short Films'],
                        default=None, help='Filter data by split (optional)')
    parser.add_argument('--question_type', default=None, help='Filter data by question_type (optional)')
    parser.add_argument('--output', default=None,
                        help='Output filename under data/minerva/test, or an absolute path')
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / 'data' / 'minerva'
    video_dir = project_root / 'data' / 'youtube_cache'
    input_file = data_dir / 'minerva.json'
    if args.output:
        output_file = data_dir / 'test' / args.output
    elif args.split:
        safe_name = args.split.lower().replace(' ', '_').replace('and_', '')
        output_file = data_dir / 'test' / f'{safe_name}_data.jsonl'
    elif args.question_type:
        safe_name = args.question_type.lower().replace(' ', '_').replace('and_', '')
        output_file = data_dir / 'test' / f'{safe_name}_data.jsonl'
    else:
        output_file = data_dir / 'test' / 'minerva_all_data.jsonl'

    try:
        data = load_annotations(input_file)
        print(f"Original record count: {len(data)}")
        if args.split:
            data = [row for row in data if row['split'] == args.split]
            print(f"Record count for split='{args.split}': {len(data)}")
        if args.question_type:
            data = [row for row in data if row['question_type'] == args.question_type]
            print(f"Record count for question_type='{args.question_type}': {len(data)}")
        validate_videos(data, video_dir)
        write_output(data, output_file, video_dir)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"\nData written to: {output_file}")
    print(f"Wrote {len(data)} records")
    return 0


if __name__ == '__main__':
    sys.exit(main())
