import unittest

from src.detection.nli_detector import NLIDetector


class NLILabelMappingTest(unittest.TestCase):
    def test_uses_checkpoint_semantic_labels(self):
        mapping = NLIDetector._resolve_label_mapping(
            {0: "contradiction", 1: "entailment", 2: "neutral"}
        )
        self.assertEqual(mapping["CONTRADICTION"], 0)
        self.assertEqual(mapping["ENTAILMENT"], 1)
        self.assertEqual(mapping["NEUTRAL"], 2)

    def test_does_not_assume_neutral_is_entailment(self):
        mapping = NLIDetector._resolve_label_mapping(
            {0: "CONTRADICTION", 1: "NEUTRAL", 2: "ENTAILMENT"}
        )
        self.assertEqual(mapping["NEUTRAL"], 1)
        self.assertEqual(mapping["ENTAILMENT"], 2)

    def test_generic_labels_use_documented_default(self):
        mapping = NLIDetector._resolve_label_mapping(
            {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"}
        )
        self.assertEqual(
            mapping,
            {"CONTRADICTION": 0, "ENTAILMENT": 1, "NEUTRAL": 2},
        )


if __name__ == "__main__":
    unittest.main()
