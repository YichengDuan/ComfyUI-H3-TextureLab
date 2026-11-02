import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('texture_planning', Path(__file__).resolve().parents[1] / 'planning.py')
planning = importlib.util.module_from_spec(spec)
spec.loader.exec_module(planning)

class WindowPlanTests(unittest.TestCase):
    def test_full_short_and_padded_timelines(self):
        for total in [5, 22, 56, 57, 90, 158, 480, 960]:
            for size, context in [(39, 5), (56, 22), (73, 39), (90, 56)]:
                with self.subTest(total=total, size=size, context=context):
                    entries = planning.window_plan(total, size, context, 41729)
                    retained = 0
                    for index, entry in enumerate(entries):
                        self.assertEqual(entry['index'], index)
                        self.assertEqual(entry['source_start'] + entry['discarded_overlap'], retained)
                        self.assertGreater(entry['source_end_exclusive'], retained)
                        self.assertLessEqual(entry['source_end_exclusive'] - entry['source_start'], size)
                        retained = entry['source_end_exclusive']
                    self.assertEqual(retained, total)

    def test_known_four_window_delivery(self):
        entries = planning.window_plan(158, 56, 22, 7)
        self.assertEqual([e['source_end_exclusive'] for e in entries], [56, 90, 124, 158])
        self.assertEqual([e['discarded_overlap'] for e in entries], [0, 22, 22, 22])

    def test_non_advancing_plan_rejected(self):
        with self.assertRaises(ValueError):
            planning.window_plan(960, 56, 56, 7)

if __name__ == '__main__':
    unittest.main()
