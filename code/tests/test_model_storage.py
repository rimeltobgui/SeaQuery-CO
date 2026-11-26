"""Opt-in storage tests with mocked downloads, no large weights or GPU."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from seaquery.inference.loaders import resolve_checkpoint, MODEL_IDS, MODEL_REVISIONS
from seaquery.inference.model_storage import require_code_ocean
from seaquery.inference.run import parser


class ModelStorageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'models'

    def download(self, repo, **kwargs):
        self.assertEqual(repo, MODEL_IDS['llava'])
        self.assertEqual(kwargs['revision'], MODEL_REVISIONS['llava'])
        dest = Path(kwargs['local_dir'])
        dest.mkdir(parents=True, exist_ok=True)
        for name in kwargs['allow_patterns']:
            (dest / name).write_text('{}' if name.endswith('.json') else 'test weights')
        return str(dest)

    def test_saves_one_model_and_reuses_without_network(self):
        files = ['config.json', 'model.safetensors', 'pytorch_model.bin', 'pytorch_model.bin.index.json',
                 'tokenizer.json', 'README.md', 'training/model.safetensors']
        with patch('seaquery.inference.model_storage.require_code_ocean'), \
             patch('huggingface_hub.list_repo_files', return_value=files), \
             patch('huggingface_hub.snapshot_download', side_effect=self.download) as download:
            path, opts, source = resolve_checkpoint('llava', self.root, save_models_to_data=True)
            self.assertEqual(Path(path), self.root / 'llava')
            self.assertTrue(opts['local_files_only'])
            self.assertEqual(source['manifest']['revision'], MODEL_REVISIONS['llava'])
            self.assertFalse((Path(path) / 'pytorch_model.bin').exists())
            self.assertFalse((Path(path) / 'README.md').exists())
            self.assertFalse((self.root / 'internvl').exists())
            resolve_checkpoint('llava', self.root, offline=True, save_models_to_data=True)
            self.assertEqual(download.call_count, 1)
        with patch('huggingface_hub.snapshot_download', side_effect=AssertionError('network')):
            self.assertEqual(resolve_checkpoint('llava', self.root, offline=True)[0], path)

    def test_interruption_retains_staging_and_retry_finishes(self):
        with patch('seaquery.inference.model_storage.require_code_ocean'), \
             patch('huggingface_hub.list_repo_files', return_value=['config.json', 'model.safetensors']), \
             patch('huggingface_hub.snapshot_download', side_effect=RuntimeError('interrupted')):
            with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                resolve_checkpoint('llava', self.root, save_models_to_data=True)
        self.assertFalse((self.root / 'llava').exists())
        self.assertTrue((self.root / '.llava.download').is_dir())
        with patch('seaquery.inference.model_storage.require_code_ocean'), \
             patch('huggingface_hub.list_repo_files', return_value=['config.json', 'model.safetensors']), \
             patch('huggingface_hub.snapshot_download', side_effect=self.download):
            resolve_checkpoint('llava', self.root, save_models_to_data=True)
        self.assertFalse((self.root / '.llava.download').exists())
        self.assertTrue((self.root / 'llava' / 'seaquery_source.json').is_file())

    def test_no_download_in_offline_mode(self):
        with patch('seaquery.inference.model_storage.require_code_ocean'), \
             patch('huggingface_hub.snapshot_download') as download:
            with self.assertRaises(FileNotFoundError):
                resolve_checkpoint('llava', self.root, offline=True, save_models_to_data=True)
            download.assert_not_called()

    def test_local_machine_rejects_flag_and_default_is_off(self):
        with self.assertRaisesRegex(ValueError, 'Code Ocean'):
            require_code_ocean(self.root)
        args = parser().parse_args(['coralvqa', '--model', 'llava'])
        self.assertFalse(args.code_ocean_save_models)
        args = parser().parse_args(['coralvqa', '--model', 'llava', '--code-ocean-save-models'])
        self.assertTrue(args.code_ocean_save_models)


if __name__ == '__main__':
    unittest.main()
