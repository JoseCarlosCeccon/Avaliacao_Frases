"""Materializa julgamentos feitos por leitura da consulta e da frase, sem usar rankings.

0: não responde; 1: tema pertinente, mas genérico/incompleto;
2: orientação/explicação concreta diretamente ligada à consulta.
As listas são decisões explícitas da IA sobre as 348 frases do pool.
"""
import csv
import json
import shutil
from pathlib import Path
from collections import Counter
from trabalho4 import write_csv

PARTIAL = {
    1: '1309 1813 2065 2101 532 4025 4144 6364 6710 6950 7214 7549 7560 7801 7804 7873 7944 7969 8130 8324 8433 9304 9809',
    2: '3500 7045 7136 7415 7741 7821 8150 8441 8620 8904 8981 8986 9004 9232 9652 9903',
    3: '1012 1715 2030 2241 2281 2344 3168 3461 3541 3647 3669 3703 4137 4434 4874 5312 5356 5828 5883 6488 6595',
    4: '1126 1297 1329 1412 1461 1602 2074 2200 2401 2941 3504 3806 6027 630 6696 7850 7972 9022 9544',
    5: '1057 1136 1370 1610 2161 2373 2518 264 2903 629 7292 7806 7977 8033 8429 8661 8689 9552 9658',
    6: '1009 1024 1451 1579 1735 1742 184 2191 259 2716 2836 2945 2968 329 429 763 7882 7887 859 883 9735',
    7: '1115 1357 1484 1544 2111 2490 2948 3339 4080 4576 5069 5254 5623 6187 6474 872 912',
    8: '1272 1536 1591 1623 1796 1799 2569 2646 2654 2730 3001 4407 4487 4634 4834 4844 4945 5045 5235 5327 5505 5698 5995 6079 642 67 68 77 8534 8693',
    9: '3025 3044 3414 3674 3680 3855 3861 3979 4126 4186 4203 4237 4310 4531 4539 4690 4723 4772 4783 4830 4955 5086 5332 5847 6117 6243 6380 6408 6469 6598 6969 7778',
    10: '26 3828 3951 4187 4277 4377 4512 5013 510 5184 5900 6234 743 78 911',
}
DIRECT = {
    1: '9427',
    2: '9229 9390 9658',
    3: '391 6247 7621 8459 8710',
    4: '7941 8639 9259',
    5: '9666 9725',
    6: '9336',
    7: '',
    8: '6612',
    9: '',
    10: '7490 7646 7647 8403 8737 9279 9494 9543 9746',
}
CRITERIA = {
    1: 'Equilíbrio/postura em golpes ou deslocamentos: parcial. Respiração relacionada explicitamente a preservar equilíbrio após chute: direta. Equilíbrio só em luta agarrada, distância, lesão e pausas: insuficiente.',
    2: 'Controle de força para evitar ferimentos e treino de quedas seguras: direto. Postura segura, sinalização de lesão e verificação de proteções: parcial. Evitar confrontos fora do treino não responde à prevenção de lesão durante o treino.',
    3: 'Benefício explícito da respiração para percepção corporal, manter técnica sob cansaço ou enfrentar um desafio de combate: direto. Apenas incentivo genérico a treinar respiração: parcial. Técnica/movimentação sem respiração: zero.',
    4: 'Esquiva protegendo queixo, leitura de sinais e preservação de opções defensivas no boxe: direto. Defesa genérica, reação, coordenação, velocidade ou prática corretiva: parcial. Flexibilidade/concentração sem defesa, rotas de fuga e recuperação: zero.',
    5: 'Posicionamento e ajuste do tempo da projeção de quadril no judô: direto. Coordenação, transições, precisão, técnica, leitura de ritmo, correções e quedas seguras: parcial. História, fuga ou capacidades genéricas pouco ligadas à execução de quedas: zero.',
    6: 'Mobilidade suficiente para chute alto sem forçar amplitude: direto. Flexibilidade genérica, postura do chute alto e mobilidade dos chutes: parcial. Pegada, defesa de queda, distância e equilíbrio sem flexibilidade: zero.',
    7: 'Controle emocional, autocontrole, confiança, foco, paciência e respiração: parcialmente relacionados. Nenhuma frase apresenta orientação explícita para ansiedade antes da luta. Força, distância tática, roupas, confronto externo e técnica corporal: zero.',
    8: 'Treinar disciplina em cada sessão e progredir: direto. Disciplina genérica e rotina regular: parcial. Não agressão sem referência à disciplina/regularidade: zero.',
    9: 'Treinar/ajustar distância e ler adversário: parcial, sem explicar concretamente como manter a distância adequada. Trabalhar outras habilidades em diferentes distâncias e manter base contra adversário pesado: zero. Nenhuma instrução específica suficiente para grau 2.',
    10: 'Aquecimento gradual antes de treino prepara articulações: direto. Condicionamento/preparação física/consciência corporal pertinentes: parcial. Preparar competição, equipamento, estados mentais ou técnica isolada não explica preparar o corpo antes de treinar.',
}


if __name__ == '__main__':
    path = Path('resultados_completos/avaliacao_preencher.csv')
    rows = list(csv.DictReader(path.open(encoding='utf-8-sig', newline='')))
    if any(r['relevancia'].strip() for r in rows):
        raise ValueError('Há avaliações existentes: não serão sobrescritas.')
    expected = {(int(r['consulta_id']), r['id']) for r in rows}
    for mapping in [PARTIAL, DIRECT]:
        assert all((q, i) in expected for q, ids in mapping.items() for i in ids.split())
    assert all(not set(PARTIAL[q].split()) & set(DIRECT[q].split()) for q in PARTIAL)
    backup = path.with_name('avaliacao_original_sem_rotulos.csv')
    if not backup.exists():
        shutil.copy2(path, backup)
    for row in rows:
        q, i = int(row['consulta_id']), row['id']
        row['relevancia'] = 2 if i in DIRECT[q].split() else 1 if i in PARTIAL[q].split() else 0
    write_csv(path, list(rows[0]), rows)
    audit = {'avaliador': 'IA (Codex); sem revisão humana',
             'procedimento': 'Leitura dos 348 pares consulta/frase; decisão sem consultar método, score ou posição.',
             'escala': {'0': 'Não responde', '1': 'Relacionado, mas incompleto ou genérico', '2': 'Resposta concreta ligada ao pedido'},
             'criterios_por_consulta': CRITERIA,
             'contagens': {str(q): dict(Counter(r['relevancia'] for r in rows if int(r['consulta_id']) == q)) for q in CRITERIA}}
    Path('resultados_completos/criterios_avaliacao_IA.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'{len(rows)} pares avaliados. Original sem rótulos preservado.')
