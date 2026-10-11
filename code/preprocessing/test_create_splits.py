"""Run: python -m unittest discover -s code/preprocessing -p 'test_*.py'."""
import unittest
import pandas as pd
from create_splits import make_splits


def fixture():
    return pd.DataFrame([dict(cell_id=f'{kind}_{i:03}', cell_type=kind, donor=f'd{i % 4}')
                         for kind, n in [('common', 100), ('boundary', 30), ('rare', 29)]
                         for i in range(n)])


class SplitTests(unittest.TestCase):
    def test_named_cell_id_index_from_anndata(self):
        cells = fixture()
        indexed = cells.set_index('cell_id', drop=False)
        expected = make_splits(cells)
        actual = make_splits(indexed)
        for left, right in zip(expected, actual):
            pd.testing.assert_frame_equal(left, right)

    def test_partition_rounding_and_reproducibility(self):
        cells = fixture()
        manifest, excluded, counts, _ = make_splits(cells)
        self.assertEqual(set(excluded.cell_type), {'rare'})
        self.assertEqual(counts.loc['common', ['train', 'validation', 'test']].tolist(), [70, 15, 15])
        self.assertEqual(counts.loc['boundary', ['train', 'validation', 'test']].tolist(), [21, 5, 4])
        self.assertEqual(set(manifest.cell_id) | set(excluded.cell_id), set(cells.cell_id))
        self.assertFalse(set(manifest.cell_id) & set(excluded.cell_id))
        self.assertTrue(manifest.cell_id.is_unique)
        pd.testing.assert_frame_equal(manifest, make_splits(cells.sample(frac=1, random_state=7))[0])

    def test_exclusions_before_threshold(self):
        manifest, excluded, _, _ = make_splits(fixture(), ['boundary_000'])
        self.assertEqual(set(manifest.cell_type), {'common'})
        self.assertEqual(len(excluded), 59)
        self.assertEqual(excluded.set_index('cell_id').loc['boundary_000', 'reason'], 'scgpt_incompatible')

    def test_invalid_metadata_and_exclusions(self):
        for ids in (['unknown'], ['common_000', 'common_000']):
            with self.assertRaises(ValueError):
                make_splits(fixture(), ids)
        cells = fixture()
        for column, value in [('cell_id', 'duplicate'), ('cell_type', ''), ('donor', None)]:
            bad = cells.copy()
            bad.loc[:1, column] = value
            with self.assertRaises(ValueError):
                make_splits(bad)
        with self.assertRaises(ValueError):
            make_splits(cells[cells.cell_type == 'rare'])


if __name__ == '__main__':
    unittest.main()
