# Trabalho 4 — análise dos resultados

Avaliação realizada em 04/10/2026, sobre a execução registrada em 02/10/2026.

O BGE-M3 isolado obteve o melhor resultado médio nas dez consultas: Precision@10 de 0,850 e nDCG@10 de 0,680. As fusões testadas não superaram esse modelo isolado. No MiniLM, 25% de peso lexical elevou Precision@10 de 0,670 para 0,720, mas reduziu nDCG@10 de 0,534 para 0,501. Portanto, recuperar mais frases relacionadas e ordenar melhor respostas diretas não são necessariamente o mesmo resultado.

## Método de avaliação

O corpus contém 10.000 frases distintas sobre artes marciais. Foram utilizadas as mesmas dez consultas, top 10 e parâmetros para todos os métodos. A união dos resultados contém 348 pares distintos de consulta e frase. Todos foram lidos e classificados pela IA (Codex), sem consultar o método, score ou posição para decidir a nota. Não houve revisão humana ou anotação de todo o corpus.

Escala: 0 = não responde; 1 = relacionado, mas genérico/incompleto; 2 = orientação ou explicação concreta diretamente ligada à consulta. Há 110 notas zero, 213 notas um e 25 notas dois. Os critérios de cada consulta estão em `resultados_completos/criterios_avaliacao_IA.json` e as notas individuais em `avaliacao_preenchida.csv`.

Precision e Recall consideram notas 1 e 2 como relevantes. nDCG distingue as notas com ganho `2^nota-1`. Todas as 2.100 posições dos rankings estão cobertas pelos julgamentos. As médias abaixo são médias por consulta. Recall é relativo aos relevantes encontrados no pool de 348 pares, e não aos relevantes existentes em todas as 10.000 frases.

## Métodos isolados

| Método | Precision@10 | Recall@10 no pool | MRR@10 | nDCG@10 | Tempo médio por consulta |
|---|---:|---:|---:|---:|---:|
| BM25 | 0,590 | 0,235 | 0,675 | 0,413 | 12,4 ms |
| MiniLM | 0,670 | 0,283 | 0,925 | 0,534 | 21,6 ms |
| BGE-M3 | 0,850 | 0,364 | 0,950 | 0,680 | 238,2 ms |

Em média, o BGE recuperou 8,5 frases relacionadas por consulta, contra 6,7 do MiniLM e 5,9 do BM25. Isso não significa 85% de respostas completas: apenas 17% dos resultados do BGE receberam nota 2; MiniLM teve 7% e BM25 teve 1%. Grande parte do corpus é formada por frases genéricas ou por variações de estruturas repetidas.

BGE teve mais qualidade média e maior custo observado. A preparação inicial registrada foi de aproximadamente 2,56 minutos para MiniLM e 87,43 minutos para BGE, incluindo carregamento/download quando necessário e geração dos embeddings. Esses tempos foram medidos em CPU. Não são tempos puros de inferência nem garantias para outra máquina. Os vetores possuem 384 e 1.024 dimensões, respectivamente, e são reutilizados nas próximas execuções.

## Média ponderada e efeito dos pesos

Alpha é o peso do BM25; o peso semântico é `1-alpha`. A normalização min-max é feita separadamente sobre todos os scores da consulta.

| Peso lexical | MiniLM P@10 | MiniLM nDCG@10 | BGE P@10 | BGE nDCG@10 |
|---:|---:|---:|---:|---:|
| 0% | 0,670 | 0,534 | 0,850 | 0,680 |
| 25% | 0,720 | 0,501 | 0,770 | 0,549 |
| 50% | 0,650 | 0,462 | 0,630 | 0,439 |
| 75% | 0,630 | 0,444 | 0,630 | 0,433 |
| 100% | 0,590 | 0,413 | 0,590 | 0,413 |

No BGE, qualquer peso lexical positivo testado piorou ambos os indicadores em relação à busca semântica isolada. Entre as fusões com contribuições das duas fontes, 25% lexical foi a melhor configuração nos dois modelos por esses indicadores. No MiniLM, essa configuração aumentou a quantidade de resultados relacionados, mas perdeu qualidade de ordenação medida por nDCG. Os extremos 0% e 100% reproduzem, neste experimento, o semântico e o lexical isolados.

## RRF e efeito de k

RRF soma `1/(k+posição)` para cada fonte, com posições começando em 1. Nesta implementação, são usados todos os documentos semânticos e os documentos com score lexical positivo. Não há janela de candidatos limitada ao top 10 antes da fusão.

| k | MiniLM P@10 | MiniLM nDCG@10 | BGE P@10 | BGE nDCG@10 |
|---:|---:|---:|---:|---:|
| 10 | 0,670 | 0,492 | 0,740 | 0,525 |
| 30 | 0,660 | 0,467 | 0,740 | 0,505 |
| 60 | 0,710 | 0,480 | 0,740 | 0,504 |
| 100 | 0,690 | 0,473 | 0,730 | 0,499 |

k menor dá maior diferença relativa entre as primeiras posições. Com k maior, as contribuições dos primeiros colocados ficam mais próximas. O melhor nDCG de RRF foi obtido com k=10 nos dois modelos. No MiniLM, a melhor Precision ocorreu com k=60, evidenciando novamente o conflito entre quantidade de relevantes e ordenação. No BGE, variar k mudou pouco a Precision; nenhum k testado recuperou a qualidade do modelo isolado.

RRF não utiliza a magnitude do score. Um resultado lexical bem colocado pode receber forte contribuição mesmo quando é apenas uma coincidência de palavras. A média ponderada preserva diferenças entre scores normalizados, mas pode ser sensível à escala e a valores extremos de cada consulta.

## Precision@10 por consulta

| Consulta | BM25 | MiniLM | BGE |
|---|---:|---:|---:|
| 1. Equilíbrio nos chutes | 0,60 | 0,50 | 0,80 |
| 2. Prevenir lesões no treino | 0,00 | 0,30 | 1,00 |
| 3. Respiração no combate | 0,90 | 1,00 | 1,00 |
| 4. Defesa e esquiva no boxe | 0,80 | 0,90 | 0,80 |
| 5. Quedas e projeções no judô | 0,60 | 0,60 | 0,80 |
| 6. Flexibilidade para chutes altos | 1,00 | 0,30 | 0,80 |
| 7. Ansiedade antes da luta | 0,00 | 0,70 | 0,50 |
| 8. Disciplina com treinos regulares | 0,90 | 1,00 | 1,00 |
| 9. Distância do adversário | 0,90 | 0,70 | 0,80 |
| 10. Preparar corpo antes do treino | 0,20 | 0,70 | 1,00 |

O melhor modelo médio não venceu todas as consultas. BM25 foi melhor em quantidade de relacionados na consulta de flexibilidade, embora predominem respostas genéricas. MiniLM foi melhor em ansiedade. Em ansiedade e distância, nenhuma frase do pool foi considerada resposta concreta de grau 2; bons valores de Precision nesses casos medem pertinência parcial.

## Exemplos para explicar na apresentação

**Lesões (consulta 2).** O primeiro resultado do BM25, ID 7151, diz: “Fora do treino de capoeira, evitar o confronto costuma ser a decisão mais segura.” Recebeu zero porque trata de evitar confronto fora do treino, não prevenir lesões durante ele. MiniLM colocou primeiro o ID 9229: “Em karatê, controlar a força favorece aprendizado sem ferimentos evitáveis.” Recebeu dois pela relação concreta entre controle e prevenção. BGE teve 10/10 frases pelo menos parcialmente pertinentes, mas seu primeiro resultado, ID 8981, apenas orienta sinalizar uma lesão e recebeu um.

**Preparação corporal (consulta 10).** BGE colocou primeiro o ID 7490: “Antes do treino de cortes e deslocamentos, um aquecimento gradual prepara articulações e atenção.” Recebeu dois. O primeiro do BM25, ID 7653, fala em evolução conjunta de intensidade e técnica para combate de solo, e recebeu zero: coincide com preparação/combate, sem explicar preparar o corpo antes do treino. BGE obteve nDCG@10 de 0,990 nessa consulta.

**Defesa no boxe (consulta 4).** BM25 e BGE colocaram primeiro o ID 8639: “A defesa em boxe inclui a esquiva com proteção do queixo.” A frase recebeu dois por indicar ação concreta. É um exemplo em que a coincidência lexical é útil e alinhada à intenção da consulta.

**Flexibilidade (consulta 6).** O primeiro do MiniLM, ID 7214, fala em bloqueio de chute com a canela e correção de postura, e recebeu zero para trabalhar flexibilidade em chutes altos. BM25 recuperou dez frases parcialmente relacionadas. Já o ID 9336, no pool, diz que mobilidade suficiente permite praticar chute alto sem forçar amplitudes desnecessárias e recebeu dois. Essa distinção ajuda a explicar por que Precision e nDCG podem contar histórias diferentes.

## Conclusões e limites

1. BGE-M3 foi o melhor método médio de relevância neste conjunto, com maior custo computacional observado.
2. A união das fontes não assegurou melhoria. Um ranking lexical fraco em determinadas consultas pode reduzir a qualidade semântica quando recebe peso na fusão.
3. Para MiniLM, 25% lexical melhorou Precision, enquanto o semântico isolado manteve o melhor nDCG médio entre suas configurações.
4. RRF k=10 teve o melhor nDCG entre os k testados; o melhor k em Precision variou por modelo.
5. A escolha de pesos e k deve declarar qual métrica se deseja otimizar e precisa ser validada com novas consultas.

Os rótulos foram atribuídos por IA e são subjetivos. As métricas não equivalem a uma avaliação humana independente. O pool foi construído pelos próprios métodos testados; frases relevantes fora dele não entram no denominador de Recall ou no ranking ideal do nDCG. Há apenas dez consultas, nenhuma separação treino/teste para selecionar parâmetros e um corpus sintético com muita repetição estrutural. Estes resultados são exploratórios e não permitem afirmar superioridade geral dos modelos.

## Arquivos e reprodução

`comparacao_metodos.csv` contém as médias de todas as 21 configurações. `metricas.csv` contém 210 linhas, uma por consulta/configuração. `rankings.csv` preserva as 2.100 posições da execução original. `avaliacao_preenchida.csv` contém os 348 julgamentos. Os tempos da execução original foram preservados; embeddings não foram recalculados para avaliar.

Para recalcular as métricas depois de revisar os rótulos:

```powershell
.\.venv\Scripts\python.exe avaliar_resultados.py --qrels 'resultados_avaliados\avaliacao_preenchida.csv' --output-dir resultados_revisados
```

Esse comando reutiliza os rankings originais e rejeita avaliações que deixem qualquer resultado sem julgamento.
