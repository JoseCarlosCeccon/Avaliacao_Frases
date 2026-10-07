import unittest
import numpy as np
import trabalho4 as t


class SearchTests(unittest.TestCase):
    def test_rrf_uses_one_based_ranks_and_ignores_zero_lexical(self):
        np.testing.assert_allclose(t.rrf(np.array([2., 0.]), np.array([0., 1.]), 1), [1/2+1/3, 1/2])

    def test_weight_endpoints_and_constant_normalization(self):
        a, b = np.array([10., 20.]), np.array([.9, .2])
        np.testing.assert_allclose(t.weighted(a, b, 1), [0, 1])
        np.testing.assert_allclose(t.weighted(a, b, 0), [1, 0])
        np.testing.assert_allclose(t.normalize(np.ones(3)), [0, 0, 0])

    def test_bm25_finds_accented_terms_and_unknown_query_is_zero(self):
        index = t.BM25(['Treino de judô', 'Praticar boxe', 'Uma corrida'])
        self.assertEqual(int(np.argmax(index.score('judo'))), 0)
        np.testing.assert_allclose(index.score('astronomia'), [0, 0, 0])

    def test_precision_recall_and_ndcg_use_judgments(self):
        result = t.metrics(['a', 'b'], {'a': 2, 'b': 0, 'c': 1})
        self.assertEqual(result['precision'], .5)
        self.assertEqual(result['recall'], .5)
        self.assertEqual(result['mrr'], 1.)
        self.assertAlmostEqual(result['ndcg'], 3/(3+1/np.log2(3)))


if __name__ == '__main__':
    unittest.main()
