"""CLI regression tests using an isolated, temporary checkout and fake videos."""

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


EXPORTER = Path(__file__).resolve().parents[1] / 'export_minerva_data.py'


def annotation(video_id, split='STEM', question_type='Reasoning'):
    return {
        'key': f'key-{video_id}',
        'video_id': video_id,
        'question': 'What happens next?',
        **{f'answer_choice_{i}': f'Choice {i}' for i in range(5)},
        'answer_id': 2,
        'answer': 'Choice 2',
        'question_type': question_type,
        'split': split,
        'category': 'Example',
        'reasoning': 'Example explanation.',
    }


class ExportMinervaTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='minerva export ')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.script = self.root / 'scripts' / 'data_process' / EXPORTER.name
        self.script.parent.mkdir(parents=True)
        shutil.copy2(EXPORTER, self.script)
        self.data_dir = self.root / 'data' / 'minerva'
        self.data_dir.mkdir(parents=True)
        self.video_dir = self.root / 'data' / 'youtube_cache'
        self.video_dir.mkdir()
        self.output = self.data_dir / 'test' / 'minerva_all_data.jsonl'

    def prepare(self, rows, video_ids=()):
        (self.data_dir / 'minerva.json').write_text(json.dumps(rows), encoding='utf-8')
        for video_id in video_ids:
            (self.video_dir / f'{video_id}.mp4').write_bytes(b'fixture, not decoded')

    def run_export(self, *args):
        return subprocess.run(
            [sys.executable, str(self.script), *args],
            # Paths must be based on the script, not the invoking directory.
            cwd=self.root.parent, capture_output=True, text=True, check=False,
        )

    def read_rows(self, path=None):
        return [json.loads(line) for line in (path or self.output).read_text(encoding='utf-8').splitlines()]

    def preserve_existing_output(self):
        self.output.parent.mkdir(parents=True)
        self.output.write_text('existing output\n', encoding='utf-8')

    def test_full_export_preserves_schema_prompt_and_ids(self):
        rows = [annotation('video-one'), annotation('video-two', 'VLOG Style')]
        self.prepare(rows, ['video-one', 'video-two'])
        result = self.run_export()
        self.assertEqual(result.returncode, 0, result.stderr)
        items = self.read_rows()
        self.assertEqual([item['question_id'] for item in items], ['video-one-1', 'video-two-2'])
        self.assertEqual(set(items[0]), {
            'key', 'video_id', 'video_path', 'video_url', 'videoID', 'question_id',
            'duration', 'question', 'options', 'answer_id', 'answer_letter',
            'answer', 'answer_text', 'question_type', 'sub_category', 'task_type',
            'split', 'category', 'reasoning', 'prompt',
        })
        self.assertEqual(items[0]['video_path'], str(self.video_dir / 'video-one.mp4'))
        self.assertEqual(items[0]['videoID'], rows[0]['video_id'])
        self.assertEqual(items[0]['options'], [f'{i}. Choice {i}' for i in range(5)])
        self.assertEqual(items[0]['answer'], 'C')
        self.assertEqual(items[0]['answer_letter'], 'C')
        self.assertEqual(items[0]['answer_id'], 2)
        self.assertEqual(items[0]['answer_text'], 'Choice 2')
        self.assertTrue(items[0]['prompt'].startswith(f"Given a video: {items[0]['video_path']}\n"))
        self.assertTrue(items[0]['prompt'].endswith('format: \nAnswer: X'))
        self.assertEqual(list(self.output.parent.glob('*.tmp')), [])

    def test_missing_videos_report_unique_paths_and_preserve_output(self):
        self.prepare([annotation('missing-b'), annotation('missing-a'), annotation('missing-b')])
        self.preserve_existing_output()
        result = self.run_export()
        self.assertEqual(result.returncode, 1)
        self.assertIn('Missing 2 local video file(s)', result.stderr)
        for video_id in ('missing-a', 'missing-b'):
            self.assertEqual(result.stderr.count(str(self.video_dir / f'{video_id}.mp4')), 1)
        self.assertEqual(self.output.read_text(encoding='utf-8'), 'existing output\n')

    def test_missing_annotation_preserves_output(self):
        self.preserve_existing_output()
        result = self.run_export()
        self.assertEqual(result.returncode, 1)
        self.assertIn('Missing MINERVA annotation file', result.stderr)
        self.assertEqual(self.output.read_text(encoding='utf-8'), 'existing output\n')

    def test_bad_annotations_fail_before_replacing_output(self):
        self.preserve_existing_output()
        bad_row = annotation('valid')
        del bad_row['answer_choice_4']
        invalid_answer = annotation('valid')
        invalid_answer['answer_id'] = 5
        for data, expected in [
            ({'unexpected': 'object'}, 'must be a JSON array'),
            (['not an object'], 'must be a JSON object'),
            ([bad_row], "'answer_choice_4' must be a string"),
            ([invalid_answer], "'answer_id' must be an integer from 0 to 4"),
        ]:
            with self.subTest(expected=expected):
                self.prepare(data, ['valid'])
                result = self.run_export()
                self.assertEqual(result.returncode, 1)
                self.assertIn(expected, result.stderr)
                self.assertEqual(self.output.read_text(encoding='utf-8'), 'existing output\n')

    def test_invalid_json_preserves_output(self):
        (self.data_dir / 'minerva.json').write_text('{invalid', encoding='utf-8')
        self.preserve_existing_output()
        result = self.run_export()
        self.assertEqual(result.returncode, 1)
        self.assertIn('Invalid MINERVA JSON', result.stderr)
        self.assertEqual(self.output.read_text(encoding='utf-8'), 'existing output\n')

    def test_filters_keep_existing_numbering_and_relative_output(self):
        self.prepare([
            annotation('excluded', 'VLOG Style'),
            annotation('also-excluded', 'STEM', 'Perception'),
            annotation('selected', 'STEM', 'Reasoning'),
        ], ['selected'])
        result = self.run_export('--split', 'STEM', '--question_type', 'Reasoning',
                                 '--output', 'nested/custom.jsonl')
        self.assertEqual(result.returncode, 0, result.stderr)
        items = self.read_rows(self.output.parent / 'nested' / 'custom.jsonl')
        self.assertEqual([item['question_id'] for item in items], ['selected-1'])
        self.assertFalse(self.output.exists())

    def test_absolute_output_and_default_filtered_filenames(self):
        self.prepare([annotation('selected', 'Sports and Board Games')], ['selected'])
        absolute_output = self.root / 'custom results' / 'chosen.jsonl'
        result = self.run_export('--output', str(absolute_output))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.read_rows(absolute_output)), 1)
        result = self.run_export('--split', 'Sports and Board Games')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.output.parent / 'sports_board_games_data.jsonl').is_file())
        result = self.run_export('--question_type', 'Reasoning')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.output.parent / 'reasoning_data.jsonl').is_file())


if __name__ == '__main__':
    unittest.main()
