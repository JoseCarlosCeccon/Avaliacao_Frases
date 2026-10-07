# Trabalho 4 — busca lexical e semântica

Exigência atual informada pelo aluno: 10.000 frases. `dataset_10000.csv` contém as primeiras 10.000 frases do CSV original, que permanece preservado. O Trabalho 3 foi usado como referência para o fluxo Python/Sentence Transformers. Nesta adaptação, a busca vetorial usa cosseno exato com NumPy em vez de ChromaDB: todas as 10.000 frases são comparadas, evitando aproximações na comparação das fusões.

## Comandos no PowerShell

Use Python 3.11 ou 3.12. Abra o terminal nesta pasta e execute:

```powershell
Set-Location 'Trabalho4'
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe trabalho4.py --validate
.\.venv\Scripts\python.exe trabalho4.py --lexical-only --output-dir resultados_lexical
.\.venv\Scripts\python.exe trabalho4.py --models minilm --output-dir resultados_minilm
.\.venv\Scripts\python.exe trabalho4.py --output-dir resultados_completos
```

Não é necessário ativar o ambiente. A última execução compara os dois modelos. A primeira execução de cada modelo exige internet para baixar seus arquivos. BGE-M3 é maior e pode levar bastante tempo em CPU. O lote padrão é 8; se faltar memória, acrescente `--batch-size 1`. O padrão é CPU; `--device cuda` requer PyTorch com CUDA e GPU compatível já configurados.

Para uma consulta própria:

```powershell
.\.venv\Scripts\python.exe trabalho4.py --query 'Como melhorar o equilíbrio nos chutes?' --output-dir consulta_personalizada
```

## Métodos e parâmetros

- BM25: tokens minúsculos sem acentos, k1=1,5 e b=0,75. Mantém stopwords e não usa stemming. Retorna apenas documentos com score positivo.
- MiniLM: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
- BGE-M3: `BAAI/bge-m3`, apenas embeddings densos. O componente lexical é o BM25 separado.
- Embeddings normalizados em L2; produto escalar equivale ao cosseno.
- RRF: soma de `1/(k + posição)`, posições começando em 1. Testa k=10,30,60,100. Usa ranking semântico completo e apenas matches lexicais positivos; uma consulta sem match lexical não recebe contribuição lexical arbitrária. Empates seguem a ordem do CSV.
- Média ponderada: `alpha * BM25_normalizado + (1-alpha) * cosseno_normalizado`. Min-max aplicado separadamente a todos os scores de cada consulta; scores constantes viram zero. Testa alpha=0;0,25;0,5;0,75;1. Alpha é o peso lexical. Os extremos reproduzem as ordenações das fontes quando há variação nos scores.

Os embeddings são reutilizados em `cache_embeddings`, com chave que inclui texto, ordem e modelo. Para regenerar após atualizar as dependências/modelos, remova os arquivos `.npy` dessa pasta. MiniLM e BGE são carregados um de cada vez.

## Resultados e avaliação

As mesmas dez consultas sobre artes marciais são usadas em todos os métodos. Cada execução gera:

- `rankings.csv`: consulta, método, posição, ID, score e frase; abra no Excel para comparação.
- `metricas.csv`: tempos por consulta/método e métricas de relevância quando houver julgamentos.
- `execucao.json`: configurações, hash da entrada, informações dos modelos e observações de medição.
- `avaliacao_preencher.csv`: união dos documentos do top 10, sem relevância inventada. Não sobrescreve uma avaliação existente; use nova pasta se mudar consultas/configurações.

Preencha `relevancia` com 0 (irrelevante), 1 (parcialmente relevante) ou 2 (responde diretamente). Avalie o conteúdo sem favorecer um método. Mantenha IDs e consultas. Depois execute:

```powershell
.\.venv\Scripts\python.exe trabalho4.py --output-dir resultados_avaliados --qrels 'resultados_completos\avaliacao_preencher.csv'
```

Gera Precision@10, Recall@10, MRR@10 e nDCG@10 por consulta/método. Relevância maior que zero conta como relevante; nDCG usa ganho `2^relevancia-1`. Campos não preenchidos não são julgamentos. Documentos não julgados contam como zero: complete todo o pool antes de comparar. Recall é relativo aos relevantes julgados, não ao corpus inteiro. O pool pode deixar relevantes fora dos resultados avaliados; declare essa limitação na apresentação. Escolher o melhor peso e avaliá-lo nas mesmas consultas é exploratório; use outras consultas para validar generalização.

Tempos lexicais incluem BM25; tempos semânticos incluem embedding da consulta e comparação vetorial. Tempos de fusão incluem apenas combinação dos scores, não os custos anteriores. Não compare tempo da fusão isolada com tempo total de busca. Preparação dos modelos/corpus é registrada separadamente e distingue cache reutilizado.

## O que analisar na apresentação

A avaliação por IA foi concluída em 04/10/2026. Consulte `resultados_avaliados/analise_resultados.md` para conclusões e limitações e `resultados_avaliados/comparacao_metodos.csv` para médias. Foram julgados os 348 pares do pool; os rankings e tempos da execução original foram preservados. Os critérios estão em `resultados_completos/criterios_avaliacao_IA.json`. O arquivo sem notas foi preservado em `resultados_completos/avaliacao_original_sem_rotulos.csv`. A avaliação não foi revisada por humanos.

Para recalcular métricas a partir de rankings existentes, sem carregar modelos:

```powershell
.\.venv\Scripts\python.exe avaliar_resultados.py --qrels 'resultados_avaliados\avaliacao_preenchida.csv' --output-dir resultados_revisados
```

Mostre exemplos em que palavras iguais favorecem BM25 e paráfrases favorecem embeddings; observe mudanças de posição entre modelos. Compare k pequeno/grande e pesos lexicais baixos/altos nas mesmas consultas. Use as métricas e exemplos reais para sustentar conclusões. Os resultados neurais da execução realizada estão em `resultados_completos` e a avaliação em `resultados_avaliados`.

## Verificação

```powershell
.\.venv\Scripts\python.exe -m unittest discover -v
```

Referências: https://www.sbert.net/docs/package_reference/sentence_transformer/model.html e https://huggingface.co/BAAI/bge-m3.
