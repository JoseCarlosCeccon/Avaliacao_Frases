"""Avalia rankings já gerados, sem recalcular embeddings ou alterar tempos originais."""
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from trabalho4 import metrics, write_csv


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def evaluate(source, destination, qrels):
    source, destination, qrels = map(Path, (source, destination, qrels))
    rankings = read_csv(source/'rankings.csv')
    judgments = defaultdict(dict)
    for row in read_csv(qrels):
        grade = int(row['relevancia'])
        if grade not in (0, 1, 2):
            raise ValueError('Relevância deve ser 0, 1 ou 2.')
        key = row['consulta_id'], row['id']
        if row['id'] in judgments[row['consulta_id']]:
            raise ValueError(f'Julgamento duplicado: {key}')
        judgments[row['consulta_id']][row['id']] = grade
    grouped = defaultdict(list)
    for row in rankings:
        if row['id'] not in judgments[row['consulta_id']]:
            raise ValueError(f"Resultado não julgado: {row['consulta_id']}/{row['id']}")
        grouped[row['consulta_id'], row['metodo']].append(row)
    metadata = json.loads((source/'execucao.json').read_text(encoding='utf-8'))
    timed = read_csv(source/'metricas.csv')
    for row in timed:
        ranked = sorted(grouped[row['consulta_id'], row['metodo']], key=lambda r: int(r['posicao']))
        ids = [r['id'] for r in ranked]
        row.update(metrics(ids + ['__sem_resultado__'] * (metadata['top']-len(ids)), judgments[row['consulta_id']]))
        row['julgados_no_top'] = len(ids)
    averages = []
    for method in sorted({r['metodo'] for r in timed}):
        rows = [r for r in timed if r['metodo'] == method]
        averages.append({'metodo': method, 'consultas': len(rows), **{
            field: mean(float(r[field]) for r in rows)
            for field in ['precision', 'recall', 'mrr', 'ndcg', 'segundos']}})
    averages.sort(key=lambda r: (-r['ndcg'], -r['precision'], r['metodo']))
    destination.mkdir(exist_ok=True)
    write_csv(destination/'metricas.csv', list(timed[0]), timed)
    write_csv(destination/'comparacao_metodos.csv', list(averages[0]), averages)
    # Preservar exatamente os rankings e tempos medidos na execução original.
    import shutil
    shutil.copy2(source/'rankings.csv', destination/'rankings.csv')
    shutil.copy2(qrels, destination/'avaliacao_preenchida.csv')
    metadata['avaliacao'] = {'avaliador': 'IA (Codex), sem validação humana',
                            'julgamentos': sum(map(len, judgments.values())),
                            'cobertura_rankings': '100%',
                            'fonte_tempos': str(source/'metricas.csv'),
                            'recall': 'Relativo ao pool julgado, não ao corpus inteiro.'}
    (destination/'execucao.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    return averages


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='resultados_completos')
    parser.add_argument('--output-dir', default='resultados_avaliados')
    parser.add_argument('--qrels', default='resultados_completos/avaliacao_preencher.csv')
    args = parser.parse_args()
    result = evaluate(args.source, args.output_dir, args.qrels)
    for row in result:
        print(f"{row['metodo']:32} P@10={row['precision']:.3f} nDCG@10={row['ndcg']:.3f}")
