import importlib
import sys
import unittest
from types import SimpleNamespace


def import_train_module():
    original_argv = sys.argv
    try:
        sys.argv = ['train.py']
        return importlib.import_module('train')
    finally:
        sys.argv = original_argv


class TrainAutoTestCommandTests(unittest.TestCase):
    def test_mamba_branch_is_forwarded_to_final_test(self):
        train = import_train_module()
        train.opt = SimpleNamespace(
            model_name='SP_KAN',
            dataset_name='SIRST3',
            dataset_dir='datasets',
            best_checkpoint_path='/tmp/checkpoint.pth.tar',
            run_dir='/tmp/run',
            threshold=0.5,
            auto_test_save_img=False,
            cross_view=False,
            cross_view_topk=0.2,
            mamba_branch=True,
        )

        command = train.build_final_test_command('/tmp/test.py')

        self.assertIn('--mamba_branch', command)
        self.assertIn('--no-save_img', command)

    def test_cross_view_settings_are_forwarded_to_final_test(self):
        train = import_train_module()
        train.opt = SimpleNamespace(
            model_name='SP_KAN',
            dataset_name='SIRST3',
            dataset_dir='datasets',
            best_checkpoint_path='/tmp/checkpoint.pth.tar',
            run_dir='/tmp/run',
            threshold=0.5,
            auto_test_save_img=True,
            cross_view=True,
            cross_view_topk=0.15,
            mamba_branch=False,
        )

        command = train.build_final_test_command('/tmp/test.py')

        self.assertIn('--cross_view', command)
        self.assertIn('--cross_view_topk', command)
        self.assertIn('0.15', command)
        self.assertIn('--save_img', command)
        self.assertNotIn('--mamba_branch', command)


if __name__ == '__main__':
    unittest.main()
