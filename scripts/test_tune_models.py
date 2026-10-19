import unittest
import pandas as pd
from tune_models import rank_dictionaries


class RankingTests(unittest.TestCase):
    def test_score_then_size_then_activity(self):
        names={f'd{i}':f'c{i}' for i in range(4)}
        best=pd.DataFrame(dict(feature_set=list(names),val_macro_f1=[.8,.9,.9,.9],C=[1.,1.,1.,1.]))
        grid=pd.DataFrame(dict(candidate=list(names.values()),atoms=[8,64,32,32],validation_mean_active_atoms=[1.,2.,6.,4.]))
        ranked=rank_dictionaries(best,grid,names)
        self.assertEqual(ranked.candidate.tolist(),['c3','c2','c1','c0'])

    def test_missing_candidate_rejected(self):
        best=pd.DataFrame(dict(feature_set=['a'],val_macro_f1=[.9],C=[1.]))
        grid=pd.DataFrame(dict(candidate=['A'],atoms=[32],validation_mean_active_atoms=[4.]))
        with self.assertRaisesRegex(ValueError,'Missing candidate'):
            rank_dictionaries(best,grid,{'a':'A','b':'B'})

if __name__=='__main__': unittest.main()
