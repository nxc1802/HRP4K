import time
import unittest
from hrp4k.infra.timing import Timer
from hrp4k.infra.hashing import canonical_json, sha256_dict, experiment_id


class TestInfra(unittest.TestCase):
    def test_timer(self):
        with Timer() as timer:
            time.sleep(0.01)
        self.assertGreater(timer.elapsed_ms, 5.0)

    def test_canonical_json_sorting(self):
        d1 = {"b": 2, "a": 1}
        d2 = {"a": 1, "b": 2}
        self.assertEqual(canonical_json(d1), canonical_json(d2))
        self.assertEqual(sha256_dict(d1), sha256_dict(d2))

    def test_experiment_id_format(self):
        exp_id = experiment_id({"model": "yolo11m", "seed": 42})
        self.assertEqual(len(exp_id), 12)
        self.assertTrue(exp_id.isalnum())


if __name__ == "__main__":
    unittest.main()
