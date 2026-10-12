#!/usr/bin/env python3
"""Batch-process JSONL files and run agent commands."""

import json
import subprocess
import sys
import argparse
import os
import glob
import time
import re
import yaml
import uuid
import tempfile
from multiprocessing import Pool, cpu_count
from functools import partial

# Retry settings
MAX_RETRIES = 4           # Maximum number of retries
RETRY_BASE_DELAY = 10     # Base delay in seconds
RETRY_BACKOFF = 2         # Exponential backoff multiplier

# Default file recording completed samples
DEFAULT_COMPLETED_FILE = "conversations/all_data_path.json"


def load_completed_question_ids(file_path):
    """
    Read the set of completed question IDs from all_data_path.json.
    A sample is complete when is_correct is non-null and either true or false.
    """
    completed_ids = set()
    if not os.path.exists(file_path):
        print(f"Warning: Completed-sample file not found: {file_path}; no samples will be skipped")
        return completed_ids

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        sessions = data.get('sessions', [])
        for session in sessions:
            is_correct = session.get('is_correct')
            # is_correct must be non-null and a boolean (true or false)
            if is_correct is not None and isinstance(is_correct, bool):
                qid = session.get('question_id')
                if qid:
                    completed_ids.add(str(qid))

        print(f"Loaded {len(completed_ids)} completed question IDs (these samples will be skipped)")
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Failed to read completed-sample file {file_path}: {e}")

    return completed_ids


def is_rate_limit_error(output: str) -> bool:
    """Check whether the output contains a 429 rate limit error."""
    patterns = [
        r'Error code: 429',
        r'ThrottlingException: Too many tokens, please wait before trying again.',
    ]
    for pattern in patterns:
        if re.search(pattern, output, re.IGNORECASE):
            return True
    return False

# Default configuration
DEFAULT_FILE = "data/videomme/test/medium_duration_data.jsonl"
DEFAULT_MAX_LINES = 10000
DEFAULT_JSONL_DIR = "data/videomme/test"


def normalize_gpu_ids(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        gpu_ids = [str(v).strip() for v in value if str(v).strip()]
    else:
        gpu_ids = [part.strip() for part in str(value).split(',') if part.strip()]
    return gpu_ids or None


def find_by_question_id(question_id, jsonl_dir=DEFAULT_JSONL_DIR):
    """Find the matching record by question_id across all JSONL files."""
    jsonl_files = glob.glob(os.path.join(jsonl_dir, "*.jsonl"))

    for jsonl_file in jsonl_files:
        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    if data.get('question_id') == question_id:
                        return data, jsonl_file
        except (FileNotFoundError, json.JSONDecodeError):
            continue
    return None, None

def run_single_question(question_id, jsonl_dir=DEFAULT_JSONL_DIR, config=None, system_prompt=None):
    """Run the sample matching the specified question_id."""
    data, source_file = find_by_question_id(question_id, jsonl_dir)

    if data is None:
        print(f"Error: question_id={question_id} not found", file=sys.stderr)
        sys.exit(1)

    prompt = data.get('prompt', '').strip()
    if not prompt:
        print(f"Error: question_id={question_id} has a missing or empty prompt field", file=sys.stderr)
        sys.exit(1)

    print(f"Found sample: question_id={question_id}")
    print(f"Source file: {source_file}")
    print(f"video_path: {data.get('video_path', 'N/A')}")
    print(f"question: {data.get('question', 'N/A')}")
    print(f"prompt: {prompt}")
    print(f"{'='*40}>  agent is working ...")

    # subprocess.run(['ducc', '-p', prompt], stdin=subprocess.DEVNULL)
    subprocess.run(['python', 'video-agent.py', '-p', prompt, '-c', config, '-sys', system_prompt], stdin=subprocess.DEVNULL)

            # ['deepagents', '-n', prompt, "--model", "anthropic:claude-sonnet-4-6", "--trust-project-mcp"],
            # ['python', 'video-agent.py', '-p', prompt],

def process_single_task(task, total, config=None, system_prompt=None, stream=False, gpu_ids=None):
    """Process one task with rate limit retries for use with multiprocessing.

    Args:
        task: An (idx, prompt) tuple.
        total: Total number of tasks.
        config: Path to the configuration file.
        system_prompt: Path to the system prompt file.
        stream: Whether to print output in real time (default: False).
        gpu_ids: Optional GPU IDs assigned in round-robin order by task index.
    """
    idx, prompt = task
    env = None
    assigned_gpu = None
    if gpu_ids:
        assigned_gpu = gpu_ids[(idx - 1) % len(gpu_ids)]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(assigned_gpu)
    for attempt in range(MAX_RETRIES + 1):
        try:
            if attempt > 0:
                print(f"\n{'='*10} Retry {attempt} (sample {idx}) {'='*10}")
            else:
                print(f"\n{'='*10} Processing sample {idx} / {total} {'='*10}")
            print(prompt)
            print(f"{'='*40}> agent is working ...")
            if assigned_gpu is not None:
                print(f"[GPU] Task {idx} uses GPU {assigned_gpu} (CUDA_VISIBLE_DEVICES={assigned_gpu})")

            # cmd = ['ducc', '-p', prompt, "--disable-model-proxy", "--verbose"]
            # -u disables Python output buffering for real-time output
            cmd = ['python', '-u', 'video-agent.py', '-p', prompt, '-c', config, '-sys', system_prompt]

            if stream:
                # Streaming mode: read output incrementally with Popen
                process = subprocess.Popen(
                    cmd,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,  # Merge stderr into stdout
                    text=True,
                    bufsize=1,  # Line buffering
                    env=env,
                )

                combined_output = []
                for line in process.stdout:
                    print(line, end='', flush=True)  # Print output as it arrives
                    combined_output.append(line)

                process.wait()
                combined_output = ''.join(combined_output)
            else:
                # Buffered-output mode: wait for completion before printing output
                result = subprocess.run(
                    cmd,
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    text=True,
                    env=env,
                )
                combined_output = (result.stdout or '') + (result.stderr or '')

            # Check for rate limit errors
            if is_rate_limit_error(combined_output):
                if attempt < MAX_RETRIES:
                    delay = RETRY_BASE_DELAY * (RETRY_BACKOFF ** attempt)
                    print(f"[RATE LIMIT] Detected a 429 rate limit error; retrying in {delay} seconds...")
                    time.sleep(delay)
                    continue
                else:
                    print(f"[RATE LIMIT] Reached the retry limit of {MAX_RETRIES}; skipping this task")
                    if not stream:
                        # Print rate limit output in buffered-output mode
                        print(combined_output)
                    return (idx, False, "Rate limit exceeded after max retries")

            # Print output after completion in buffered-output mode
            if not stream and combined_output:
                print(combined_output)
            return (idx, True, None)

        except Exception as e:
            return (idx, False, str(e))

    return (idx, False, "Unknown error after retries")


def run_parallel_workers(indexed_tasks, workers, config, system_prompt, gpu_ids, stream=False):
    """Run tasks using persistent workers and a dynamic task queue.

    All tasks share one JSONL queue file; workers fetch tasks using a byte-offset cursor protected by fcntl locks.
    As each worker finishes, it fetches the next task while the queue still has work.

    Returns:
        (success_count, fail_count): Counts of successful and failed workers.
    """
    run_id = f"{int(time.time())}_{uuid.uuid4().hex[:6]}"
    queue_file = f"/tmp/batch_queue_{run_id}.jsonl"
    cursor_file = f"/tmp/batch_queue_{run_id}.cursor"
    lock_file = f"/tmp/batch_queue_{run_id}.lock"

    # 1. Write all tasks to a single shared queue file
    with open(queue_file, 'w', encoding='utf-8') as f:
        for idx, prompt in indexed_tasks:
            f.write(json.dumps({"idx": idx, "prompt": prompt}, ensure_ascii=False) + '\n')

    # 2. Initialize the cursor to zero
    with open(cursor_file, 'w') as f:
        f.write('0')

    # 3. Start worker processes that share the same queue
    processes = []
    for wid in range(workers):
        assigned_gpu = gpu_ids[wid % len(gpu_ids)] if gpu_ids else None
        env = os.environ.copy()
        if assigned_gpu:
            env["CUDA_VISIBLE_DEVICES"] = str(assigned_gpu)

        cmd = ['python', '-u', 'batch_worker.py',
               '-c', config,
               '-sys', system_prompt,
               '--queue-file', queue_file,
               '--worker-id', str(wid),
               '--total-tasks', str(len(indexed_tasks))]
        if stream:
            cmd.append('--stream')

        print(f"  Starting Worker {wid}" + (f" (GPU {assigned_gpu})" if assigned_gpu else ""))
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=sys.stdout,
            stderr=sys.stderr,
            text=True,
            bufsize=1,
            env=env,
        )
        processes.append(proc)

    # 4. Wait for all workers to finish
    print(f"\n{'='*44}")
    print(f"Started {workers} persistent workers (dynamic task queue with automatic load balancing)")
    print(f"Queue file: {queue_file}")
    print(f"{'='*44}\n")

    for wid, proc in enumerate(processes):
        retcode = proc.wait()
        if retcode != 0:
            print(f"  Worker {wid} exit code: {retcode}", file=sys.stderr)

    # 5. Clean up temporary files
    for tf_path in [queue_file, cursor_file, lock_file]:
        try:
            os.unlink(tf_path)
        except OSError:
            pass

    success_count = sum(1 for p in processes if p.returncode == 0)
    fail_count = sum(1 for p in processes if p.returncode != 0)
    return success_count, fail_count


def main():
    parser = argparse.ArgumentParser(description='Batch-run video agent commands')
    parser.add_argument('-c', '--config', type=str, help='Path to the experiment YAML configuration file')
    parser.add_argument('-f', '--file', default=DEFAULT_FILE, help='Path to the JSONL file')
    parser.add_argument('-n', '--max-datas', type=int, default=10000, help='Maximum number of samples to process')
    parser.add_argument('-s', '--start-line', type=int, default=0, help='Starting line number (zero-based)')
    parser.add_argument('-r', '--sample-rate', type=int, default=1, help='Sampling interval (take one sample every N lines)')
    parser.add_argument('-q', '--question-id', type=str, help='Run a single sample by question_id (e.g., 304-3)')
    parser.add_argument('-d', '--jsonl-dir', default=DEFAULT_JSONL_DIR, help='Directory containing JSONL files (searched when using -q)')
    parser.add_argument('-w', '--workers', type=int, default=1, help='Number of parallel processes (default: 1 for sequential execution)')
    parser.add_argument('--qid-file', type=str, default=None, help='Path to a question_id filter file (one ID per line; keep only matching samples)')
    parser.add_argument('--system-prompt', type=str, default=None, help='Path to the system prompt file passed to the video agent')
    parser.add_argument('--skip-completed', action='store_true', help='Skip completed samples (read from conversations/all_data_path.json)')
    parser.add_argument('--completed-file', type=str, default=DEFAULT_COMPLETED_FILE, help='Path to the completed-sample record file (default: conversations/all_data_path.json)')
    parser.add_argument('--stream', action='store_true', help='Stream inference output in real time (disabled by default; print output after completion)')
    parser.add_argument('--gpu-ids', type=str, default=None, help='Comma-separated physical GPU IDs to make visible, e.g., 0,1,2')
    parser.add_argument('--no-persistent-worker', action='store_true', help='Disable persistent workers and use the legacy subprocess-per-sample mode')
    args = parser.parse_args()

    gpu_ids = normalize_gpu_ids(args.gpu_ids)

    # Load parameters from the configuration file if provided
    if args.config:
        try:
            with open(args.config, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f)
            print(f"📄 Reading configuration file: {args.config}")

            # Apply values from the configuration file to the parsed arguments
            args.max_datas = cfg.get('max_samples', args.max_datas)
            args.sample_rate = cfg.get('sample_rate', args.sample_rate)
            args.workers = cfg.get('workers', args.workers)
            args.skip_completed = cfg.get('skip_completed', args.skip_completed)
            if gpu_ids is None:
                gpu_ids = normalize_gpu_ids(cfg.get('gpu_ids'))

            # Read the system prompt file path from the configuration
            prompt_file = cfg.get('system_prompt_file')
            args.system_prompt = prompt_file

            # Set the completed_file path based on exp_name
            exp_name = cfg.get('exp_name')
            if exp_name:
                args.completed_file = f"conversations/all_data_path_{exp_name}.json"

            # if not args.system_prompt and prompt_file and os.path.exists(prompt_file):
            #     with open(prompt_file, 'r', encoding='utf-8') as pf:
            #         args.system_prompt = pf.read().strip()
            #     print(f"   system_prompt_file: {prompt_file}")

            print(f"   max_samples={args.max_datas}, sample_rate={args.sample_rate}, workers={args.workers}, skip_completed={args.skip_completed}")
        except (FileNotFoundError, yaml.YAMLError) as e:
            print(f"Warning: Could not read configuration file {args.config}: {e}", file=sys.stderr)

    # Run a single sample if question_id is specified
    if args.question_id:
        run_single_question(args.question_id, args.jsonl_dir, args.config, args.system_prompt)
        return

    # Load completed question IDs so those samples can be skipped
    completed_ids = set()
    if args.skip_completed:
        completed_ids = load_completed_question_ids(args.completed_file)

    # Load the question_id filter set
    qid_set = None
    if args.qid_file:
        try:
            with open(args.qid_file, 'r', encoding='utf-8') as f:
                qid_set = set(line.strip() for line in f if line.strip())
            print(f"Loaded {len(qid_set)} question IDs for filtering: {args.qid_file}")
        except FileNotFoundError:
            print(f"Error: Question ID file not found: {args.qid_file}", file=sys.stderr)
            sys.exit(1)

    total_line_nums = args.sample_rate * args.max_datas
    print(f"Processing file: {args.file} (start at line {args.start_line}, up to {args.max_datas} samples, sample every {args.sample_rate} lines, stop before line {args.start_line+total_line_nums})")

    try:
        with open(args.file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for i, line in enumerate(f)
                     if i >= args.start_line and i < args.start_line + total_line_nums
                     and (i - args.start_line) % args.sample_rate == 0]
    except FileNotFoundError:
        print(f"Error: File not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    # Preprocess all tasks
    tasks = []
    skipped_count = 0
    for line in lines:
        if not line:
            continue
        try:
            data = json.loads(line)
            qid = data.get('question_id', '')
            # Skip completed samples
            if completed_ids and str(qid) in completed_ids:
                skipped_count += 1
                continue
            if qid_set and str(qid) not in qid_set:
                continue
            prompt = data.get('prompt', '').strip()
            if prompt:
                tasks.append(prompt)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}", file=sys.stderr)

    if skipped_count > 0:
        print(f"Skipped {skipped_count} completed samples")

    total = len(tasks)
    # Add task indices
    indexed_tasks = [(i + 1, prompt) for i, prompt in enumerate(tasks)]

    if total == 0:
        print("No samples to process; exiting")
        return

    workers = min(args.workers, total) if args.workers > 1 else 1
    print(f"Using {workers} processes to handle {total} samples...")
    if gpu_ids:
        print(f"Round-robin GPU assignment enabled: gpu_ids={gpu_ids}")
        if workers > len(gpu_ids):
            print(f"⚠️  Warning: workers={workers} exceeds the GPU count of {len(gpu_ids)}; multiple concurrent tasks will share a GPU")
    else:
        print("Round-robin GPU assignment disabled; using default CUDA device behavior")

    success_count = 0
    fail_count = 0

    if workers == 1:
        # Run tasks sequentially
        for task in indexed_tasks:
            idx, success, error = process_single_task(task, total, args.config, args.system_prompt, args.stream, gpu_ids)
            if success:
                success_count += 1
            else:
                fail_count += 1
                print(f"Processing error (sample {idx}): {error}", file=sys.stderr)
    else:
        # Run tasks in parallel using multiple processes
        # Use persistent workers by default (load MCP once per worker)
        # Use --no-persistent-worker to fall back to a subprocess per sample
        if args.no_persistent_worker:
            # Legacy mode: start a subprocess per sample and reload MCP each time
            if args.stream:
                print("⚠️  Warning: --stream is ignored in legacy multiprocessing mode; output is printed after each task completes")
            worker_func = partial(
                process_single_task,
                total=total,
                config=args.config,
                system_prompt=args.system_prompt,
                stream=False,
                gpu_ids=gpu_ids,
            )
            with Pool(processes=workers) as pool:
                results = pool.map(worker_func, indexed_tasks, chunksize=1)

            for idx, success, error in results:
                if success:
                    success_count += 1
                else:
                    fail_count += 1
                    print(f"Processing error (sample {idx}): {error}", file=sys.stderr)
        else:
            # Persistent worker mode: load MCP once per worker
            print(f"Using persistent workers with a dynamic task queue ({workers} workers, automatic load balancing)")
            print(f"   Tip: Use --no-persistent-worker to fall back to the legacy per-sample mode")
            success_count, fail_count = run_parallel_workers(
                indexed_tasks, workers,
                config=args.config,
                system_prompt=args.system_prompt,
                gpu_ids=gpu_ids,
                stream=args.stream,
            )

    print(f"\n{'='*44}")
    print(f"Done! Succeeded: {success_count}, failed: {fail_count}")
    print(f"{'='*44}")

if __name__ == "__main__":
    main()
