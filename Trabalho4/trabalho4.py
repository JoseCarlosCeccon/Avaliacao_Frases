"""Trabalho 4: BM25, embeddings densos, RRF e média ponderada.

Busca exata por cosseno com NumPy (10.000 vetores), sem servidor externo.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
import unicodedata
import numpy as np

os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
MODELS = {'minilm': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
          'bge': 'BAAI/bge-m3'}
QUERIES = [
    'Como melhorar o equilíbrio durante os chutes?',
    'Como evitar lesões durante o treino de artes marciais?',
    'Qual a importância da respiração no combate?',
    'Como desenvolver defesa e esquiva no boxe?',
    'Como melhorar as quedas e projeções no judô?',
    'Como trabalhar flexibilidade para chutes altos?',
    'Como controlar a ansiedade antes de uma luta?',
    'Como fortalecer a disciplina com treinos regulares?',
    'Como manter a distância certa do adversário?',
    'Como preparar o corpo antes de treinar?']


def tokenize(text):
    text = ''.join(c for c in unicodedata.normalize('NFD', text.lower())
                   if unicodedata.category(c) != 'Mn')
    return re.findall(r'[a-z0-9]+', text)


class BM25:
    """BM25 com k1=1.5, b=0.75 e IDF positiva; sem stemming/stopwords."""
    def __init__(self, texts):
        self.counts = [Counter(tokenize(text)) for text in texts]
        self.lengths = np.array([sum(c.values()) for c in self.counts])
        self.avg = max(float(self.lengths.mean()), 1.)
        self.df = Counter(term for c in self.counts for term in c)

    def score(self, query):
        scores = np.zeros(len(self.counts))
        for term in set(tokenize(query)):
            df = self.df.get(term, 0)
            if not df:
                continue
            tf = np.array([c.get(term, 0) for c in self.counts], dtype=float)
            idf = math.log(1 + (len(scores) - df + .5) / (df + .5))
            scores += idf * tf * 2.5 / (tf + 1.5 * (.25 + .75 * self.lengths / self.avg))
        return scores


def order(scores):
    # Desempates seguem a ordem original do CSV.
    return np.argsort(-scores, kind='stable')


def normalize(scores):
    lo, hi = float(np.min(scores)), float(np.max(scores))
    return (scores - lo) / (hi - lo) if hi > lo else np.zeros_like(scores)


def weighted(lexical, semantic, alpha):
    if not 0 <= alpha <= 1:
        raise ValueError('O peso lexical deve estar entre 0 e 1.')
    return alpha * normalize(lexical) + (1 - alpha) * normalize(semantic)


def rrf(lexical, semantic, k):
    if k <= 0:
        raise ValueError('k deve ser positivo.')
    scores = np.zeros(len(lexical))
    for values, only_positive in [(lexical, True), (semantic, False)]:
        ranked = order(values)
        if only_positive:
            ranked = ranked[values[ranked] > 0]
        scores[ranked] += 1 / (k + np.arange(1, len(ranked) + 1))
    return scores


def metrics(ids, judgments):
    grades = [int(judgments.get(i, 0)) for i in ids]
    relevant = sum(int(g) > 0 for g in judgments.values())
    hits = sum(g > 0 for g in grades)
    dcg = sum((2**g - 1) / math.log2(p + 2) for p, g in enumerate(grades))
    ideal = sorted((int(g) for g in judgments.values()), reverse=True)[:len(ids)]
    idcg = sum((2**g - 1) / math.log2(p + 2) for p, g in enumerate(ideal))
    return {'precision': hits / len(ids) if ids else 0.,
            'recall': hits / relevant if relevant else 0.,
            'mrr': next((1 / p for p, g in enumerate(grades, 1) if g > 0), 0.),
            'ndcg': dcg / idcg if idcg else 0.}


def load_dataset(path, limit):
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if not {'id', 'frase'} <= set(reader.fieldnames or []):
            raise ValueError('O CSV deve ter as colunas id e frase.')
        rows = list(reader)
    if len(rows) < limit:
        raise ValueError(f'CSV contém {len(rows)} registros; são necessários {limit}.')
    selected = rows[:limit]
    if any(not r['id'].strip() or not r['frase'].strip() for r in selected):
        raise ValueError('Há IDs ou frases vazios.')
    if len({r['id'] for r in selected}) != limit:
        raise ValueError('Há IDs repetidos.')
    if len({r['frase'] for r in selected}) != limit:
        raise ValueError('Há frases repetidas.')
    return selected, len(rows)


def write_csv(path, fields, records):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def embeddings(texts, model_name, cache, batch_size, device):
    from sentence_transformers import SentenceTransformer
    import torch
    key = hashlib.sha256(json.dumps([model_name, texts], ensure_ascii=False).encode()).hexdigest()
    path = cache / f'{key}.npy'
    started = time.perf_counter()
    model = SentenceTransformer(model_name, device=device)
    cached = path.exists()
    if cached:
        vectors = np.load(path, allow_pickle=False)
    else:
        vectors = model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                               convert_to_numpy=True, show_progress_bar=True)
        np.save(path, vectors)
    if vectors.ndim != 2 or len(vectors) != len(texts) or not np.isfinite(vectors).all():
        raise ValueError(f'Cache inválido: {path}. Remova o arquivo e execute novamente.')
    return model, vectors, {'cache_reutilizado': cached,
                            'preparacao_segundos': time.perf_counter() - started,
                            'dimensao': int(vectors.shape[1]),
                            'device': str(model.device), 'torch': torch.__version__}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('dataset_10000.csv'))
    parser.add_argument('--output-dir', type=Path, default=Path('resultados'))
    parser.add_argument('--limit', type=int, default=10000)
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--lexical-only', action='store_true')
    parser.add_argument('--models', nargs='+', choices=list(MODELS), default=list(MODELS))
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--top', type=int, default=10)
    parser.add_argument('--weights', nargs='+', type=float, default=[0, .25, .5, .75, 1])
    parser.add_argument('--ks', nargs='+', type=int, default=[10, 30, 60, 100])
    parser.add_argument('--query', action='append', help='Pode repetir para consultas personalizadas.')
    parser.add_argument('--qrels', type=Path, help='CSV: consulta_id,id,relevancia (0,1,2).')
    args = parser.parse_args()
    if args.limit < 1 or args.top < 1 or args.batch_size < 1:
        parser.error('limit, top e batch-size devem ser positivos.')
    if any(not 0 <= w <= 1 for w in args.weights) or any(k <= 0 for k in args.ks):
        parser.error('Pesos devem estar entre 0 e 1; valores de k devem ser positivos.')
    rows, total = load_dataset(args.input, args.limit)
    print(f'Entrada validada: {len(rows)} frases únicas (CSV: {total} registros).', flush=True)
    if args.validate:
        return
    queries = args.query or QUERIES
    judgments = {}
    if args.qrels:
        with args.qrels.open(encoding='utf-8-sig', newline='') as f:
            for r in csv.DictReader(f):
                if r['relevancia'].strip():
                    grade = int(r['relevancia'])
                    if grade not in (0, 1, 2) or r['id'] not in {row['id'] for row in rows}:
                        raise ValueError('Julgamento inválido: ID inexistente ou relevância fora de 0,1,2.')
                    if not 1 <= int(r['consulta_id']) <= len(queries):
                        raise ValueError('Consulta inexistente no arquivo de julgamentos.')
                    judgments.setdefault(r['consulta_id'], {})[r['id']] = grade
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache = Path('cache_embeddings')
    cache.mkdir(exist_ok=True)
    texts = [r['frase'] for r in rows]
    index = BM25(texts)
    lexical = []
    results, summary, model_info = [], [], {}

    def record(qid, method, scores, elapsed, positive_only=False):
        ranked = order(scores)
        if positive_only:
            ranked = ranked[scores[ranked] > 0]
        ranked = ranked[:args.top]
        for rank, i in enumerate(ranked, 1):
            results.append({'consulta_id': qid, 'consulta': queries[qid-1], 'metodo': method,
                            'posicao': rank, 'id': rows[i]['id'], 'score': float(scores[i]), 'frase': texts[i]})
        entry = {'consulta_id': qid, 'metodo': method, 'segundos': elapsed}
        if str(qid) in judgments:
            # Lista vazia conta como precision=0; vagas vazias também são penalizadas.
            ids = [rows[i]['id'] for i in ranked]
            entry.update(metrics(ids + ['__sem_resultado__'] * (min(args.top, len(rows))-len(ids)), judgments[str(qid)]))
            entry['julgados_no_top'] = sum(i in judgments[str(qid)] for i in ids)
        summary.append(entry)

    for qid, query in enumerate(queries, 1):
        start = time.perf_counter()
        scores = index.score(query)
        lexical.append(scores)
        record(qid, 'BM25', scores, time.perf_counter()-start, True)
    if not args.lexical_only:
        for label in args.models:
            print(f'Preparando {MODELS[label]}...', flush=True)
            model, vectors, model_info[label] = embeddings(texts, MODELS[label], cache, args.batch_size, args.device)
            for qid, query in enumerate(queries, 1):
                start = time.perf_counter()
                qvector = model.encode(query, normalize_embeddings=True, convert_to_numpy=True)
                scores = vectors @ qvector
                elapsed = time.perf_counter()-start
                record(qid, label, scores, elapsed)
                for k in args.ks:
                    start = time.perf_counter()
                    fused = rrf(lexical[qid-1], scores, k)
                    record(qid, f'{label}_RRF_k{k}', fused, time.perf_counter()-start)
                for alpha in args.weights:
                    start = time.perf_counter()
                    fused = weighted(lexical[qid-1], scores, alpha)
                    record(qid, f'{label}_ponderada_lexical{alpha:g}', fused, time.perf_counter()-start)
            del model, vectors
            gc.collect()
            if args.device == 'cuda':
                import torch
                torch.cuda.empty_cache()
    fields = ['consulta_id', 'consulta', 'metodo', 'posicao', 'id', 'score', 'frase']
    write_csv(args.output_dir/'rankings.csv', fields, results)
    write_csv(args.output_dir/'metricas.csv', ['consulta_id', 'metodo', 'segundos', 'precision', 'recall', 'mrr', 'ndcg', 'julgados_no_top'], summary)
    pool = sorted({(r['consulta_id'], r['id']) for r in results})
    by_id = {r['id']: r['frase'] for r in rows}
    template = args.output_dir/'avaliacao_preencher.csv'
    if not template.exists():
        write_csv(template, ['consulta_id', 'consulta', 'id', 'frase', 'relevancia'],
                  [{'consulta_id': q, 'consulta': queries[q-1], 'id': i, 'frase': by_id[i], 'relevancia': ''} for q, i in pool])
    metadata = {'registros': len(rows), 'input_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
                'modelos': model_info, 'queries': queries, 'weights': args.weights, 'ks': args.ks,
                'top': args.top, 'numpy': np.__version__,
                'nota_tempos': 'Tempo por método: cálculo dos scores, sem ordenação/exportação. Fusão exclui cálculo lexical e embedding.',
                'nota_avaliacao': 'Métricas usam apenas julgamentos fornecidos; não julgados contam como zero. Recall é relativo ao conjunto julgado.'}
    (args.output_dir/'execucao.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Resultados salvos em {args.output_dir.resolve()}', flush=True)


if __name__ == '__main__':
    main()
