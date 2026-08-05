import ast
import pathlib
import unittest

from util.data_split import split_train_validation


class DataSplitTests(unittest.TestCase):
    def test_train_and_validation_are_disjoint(self):
        names = [f"sample-{i}" for i in range(20)]
        train, validation = split_train_validation(names, val_ratio=0.1, seed=7)
        self.assertTrue(set(train).isdisjoint(validation))
        self.assertEqual(set(train) | set(validation), set(names))

    def test_split_is_reproducible_and_does_not_mutate_input(self):
        names = [f"sample-{i}" for i in range(20)]
        original = names.copy()
        first = split_train_validation(names, val_ratio=0.1, seed=123)
        second = split_train_validation(names, val_ratio=0.1, seed=123)
        self.assertEqual(first, second)
        self.assertEqual(names, original)
        self.assertNotEqual(first, split_train_validation(names, val_ratio=0.1, seed=124))

    def test_training_updates_only_train_loader(self):
        source = pathlib.Path(__file__).parents[1].joinpath("train.py").read_text()
        tree = ast.parse(source)
        optimizer_steps = [node for node in ast.walk(tree)
                           if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                           and node.func.attr == "step" and isinstance(node.func.value, ast.Name)
                           and node.func.value.id == "optimizer"]
        self.assertEqual(len(optimizer_steps), 1)
        self.assertIn("for img, gt_mask in train_loader", source)
        self.assertIn("for idx_iter, (img, gt_mask, target_size, org_size, _) in enumerate(validation_loader)", source)

    def test_test_loader_reads_official_test_list(self):
        dataset = pathlib.Path(__file__).parents[1] / "datasets" / "SIRST3"
        if not dataset.exists():
            self.skipTest("dataset archive is not unpacked")
        train_list = (dataset / "img_idx" / "train_SIRST3.txt").read_text().splitlines()
        test_list = (dataset / "img_idx" / "test_SIRST3.txt").read_text().splitlines()
        self.assertTrue(test_list)
        self.assertTrue(set(train_list).isdisjoint(test_list))
        # This guards the contract used by test.py without loading image data.
        loader_source = pathlib.Path(__file__).parents[1].joinpath("util/dataset_resize.py").read_text()
        self.assertIn("test_\' + test_dataset_name + \'.txt", loader_source)


if __name__ == "__main__":
    unittest.main()
