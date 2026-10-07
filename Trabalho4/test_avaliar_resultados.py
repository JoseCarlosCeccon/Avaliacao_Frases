import csv
import json
import tempfile
import unittest
from pathlib import Path
from avaliar_resultados import evaluate
from trabalho4 import write_csv


class EvaluationTests(unittest.TestCase):
    def fixture(self, root):
        source = root/'source'
        source.mkdir()
        write_csv(source/'rankings.csv', ['consulta_id', 'metodo', 'posicao', 'id'], [
            {'consulta_id': 1, 'metodo': 'teste', 'posicao': 1, 'id': 'a'},
            {'consulta_id': 1, 'metodo': 'teste', 'posicao': 2, 'id': 'b'}])
        write_csv(source/'metricas.csv', ['consulta_id', 'metodo', 'segundos'], [
            {'consulta_id': 1, 'metodo': 'teste', 'segundos': .125}])
        (source/'execucao.json').write_text(json.dumps({'top': 2}), encoding='utf-8')
        qrels = root/'qrels.csv'
        write_csv(qrels, ['consulta_id', 'id', 'relevancia'], [
            {'consulta_id': 1, 'id': 'a', 'relevancia': 2},
            {'consulta_id': 1, 'id': 'b', 'relevancia': 0}])
        return source, qrels

    def test_existing_rankings_get_metrics_without_changing_times(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, qrels = self.fixture(root)
            summary = evaluate(source, root/'output', qrels)
            self.assertEqual(summary[0]['precision'], .5)
            self.assertEqual(summary[0]['recall'], 1.)
            self.assertEqual(summary[0]['ndcg'], 1.)
            self.assertEqual(summary[0]['segundos'], .125)
            self.assertEqual((source/'rankings.csv').read_bytes(), (root/'output/rankings.csv').read_bytes())

    def test_incomplete_pool_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, qrels = self.fixture(root)
            write_csv(qrels, ['consulta_id', 'id', 'relevancia'], [
                {'consulta_id': 1, 'id': 'a', 'relevancia': 2}])
            with self.assertRaisesRegex(ValueError, 'Resultado não julgado'):
                evaluate(source, root/'output', qrels)
