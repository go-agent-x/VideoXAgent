#!/usr/bin/env python3
"""Write experiment configuration to all_data_path.json"""

import json
import yaml
import argparse
import fcntl
from pathlib import Path
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description="Write experiment configuration to all_data_path.json")
    parser.add_argument('--config', type=str,
                        help='Path to the experiment configuration YAML file (takes precedence)')
    parser.add_argument('--system-prompt-file', default='system_prompt/system_prompt_video.txt',
                        help='Path to the system prompt file')
    # Individual arguments (required when --config is not used)
    parser.add_argument('--max-samples', type=int,
                        help='Total number of test samples (-n)')
    parser.add_argument('--sample-rate', type=int,
                        help='Sampling interval: take one record every N records (-r)')
    parser.add_argument('--workers', type=int,
                        help='Number of parallel processes (-w)')
    parser.add_argument('--skip-completed', action='store_true',
                        help='Skip completed samples')
    parser.add_argument('--output', default='conversations/all_data_path.json',
                        help='Path to the output JSON file')
    args = parser.parse_args()

    # Load parameters from the configuration file if provided
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"❌ Configuration file not found: {config_path}")
            return
        with open(config_path, 'r', encoding='utf-8') as f:
            exp_config = yaml.safe_load(f)

        max_samples = exp_config.get('max_samples')
        sample_rate = exp_config.get('sample_rate')
        workers = exp_config.get('workers')
        skip_completed = exp_config.get('skip_completed', False)
        system_prompt_file = exp_config.get('system_prompt_file', args.system_prompt_file)
        data_files = exp_config.get('data_files', [])

        # Read the model configuration from the model section in the YAML file
        model_config = exp_config.get('model', {})
        model = model_config.get('model', model_config.get('model_name', 'unknown'))

        # Read exp_name to distinguish result files for different experiments
        exp_name = exp_config.get('exp_name')
    else:
        # Use command-line arguments
        if not all([args.max_samples, args.sample_rate, args.workers]):
            parser.error("When --config is not used, --max-samples, --sample-rate, and --workers are required")
        max_samples = args.max_samples
        sample_rate = args.sample_rate
        workers = args.workers
        skip_completed = args.skip_completed
        system_prompt_file = args.system_prompt_file
        data_files = []
        model = 'unknown'
        exp_name = None

    # Read the system_prompt content
    prompt_path = Path(system_prompt_file)
    system_prompt = ""
    if prompt_path.exists():
        system_prompt = prompt_path.read_text(encoding='utf-8').strip()

    # Build the experiment configuration
    config = {
        "model": model,
        "system_prompt_file": str(prompt_path),
        "system_prompt": system_prompt,
        "params": {
            "max_samples": max_samples,
            "sample_rate": sample_rate,
            "workers": workers,
            "skip_completed": skip_completed
        },
        "data_files": data_files,
        "timestamp": datetime.now().isoformat()
    }

    # Write JSON using a file lock to prevent concurrent write conflicts
    # Determine the output path based on exp_name
    if exp_name:
        output_path = Path(f"conversations/all_data_path_{exp_name}.json")
    else:
        output_path = Path(args.output)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'a+', encoding='utf-8') as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # Acquire an exclusive lock
        try:
            f.seek(0)
            content = f.read()
            if content:
                data = json.loads(content)
            else:
                data = {"sessions": []}

            data["experiment_config"] = config

            f.seek(0)
            f.truncate()
            json.dump(data, f, indent=2, ensure_ascii=False)
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)  # Release the lock

    print(f"✅ Experiment configuration written to: {output_path}")
    print(f"   Model: {model}")
    print(f"   System Prompt file: {prompt_path}")
    print(f"   Parameters: max_samples={max_samples}, sample_rate={sample_rate}, workers={workers}, skip_completed={skip_completed}")


if __name__ == "__main__":
    main()
