# Trabalho 3 - Busca Semantica com ChromaDB

## Objetivo

Produzir todos os artefatos solicitados no enunciado do Trabalho 3: um programa Python que compare os modelos `all-MiniLM-L6-v2` e `all-mpnet-base-v2` em buscas semanticas com ChromaDB, dois arquivos de saida com cinco frases por pergunta e um relatorio que analise os resultados e explique as diferencas entre os modelos.

## Contexto verificado

- O enunciado fornece 10 perguntas em portugues e exige cinco frases associadas a cada pergunta para cada modelo.
- O PDF afirma que `trabalho3-input.txt` possui 10.000 frases unicas.
- O arquivo efetivamente fornecido possui 1.000 linhas nao vazias e 1.000 frases unicas em UTF-8. A solucao processara o arquivo real integralmente e registrara essa divergencia no relatorio, sem fabricar dados ausentes.
- A pasta nao e um repositorio Git; portanto, os documentos de projeto nao podem ser versionados por commit neste ambiente.

## Abordagem selecionada

Cada modelo sera usado por meio do `SentenceTransformerEmbeddingFunction` do ChromaDB. Para garantir uma comparacao justa, as mesmas frases, identificadores, perguntas, quantidade de resultados e metrica de distancia serao usadas nas duas colecoes. Cada colecao sera construida em memoria durante a execucao; isso evita que resultados antigos contaminem novas execucoes e mantem o trabalho simples de reproduzir.

A busca sera vetorial direta, sem expansao de consulta, palavras-chave auxiliares ou reranqueamento. Assim, qualquer diferenca observada decorre principalmente dos embeddings dos dois modelos, e nao de etapas adicionais.

## Estrutura dos artefatos

Todos os arquivos finais ficarao centralizados em `entrega-trabalho3/`, pronta para compactacao e envio ao AVA:

Os tres artefatos executaveis/resultantes obrigatorios sao o codigo `trabalho3.py` e os dois arquivos `.txt` efetivamente gerados por esse codigo, um para cada modelo. O relatorio analisara exatamente esses resultados.

- `entrega-trabalho3/trabalho3.py`: entrada de linha de comando, leitura e validacao do corpus, criacao das colecoes ChromaDB, consultas, formatacao das saidas e calculo das metricas comparativas.
- `entrega-trabalho3/trabalho3-input.txt`: copia do arquivo de entrada necessario para reproduzir a execucao.
- `entrega-trabalho3/requirements.txt`: dependencias e versoes minimas compativeis.
- `entrega-trabalho3/README.md`: instalacao, execucao, estrutura e observacoes de reprodutibilidade.
- `entrega-trabalho3/trabalho3-output-all-MiniLM-L6-v2.txt`: 10 blocos de perguntas, cada um com cinco frases e seus valores de distancia.
- `entrega-trabalho3/trabalho3-output-all-mpnet-base-v2.txt`: mesma estrutura para o segundo modelo.
- `entrega-trabalho3/trabalho3-metricas.json`: dados estruturados usados na analise, incluindo distancias, sobreposicao entre os top-5 e indicadores de relevancia.
- `entrega-trabalho3/relatorio-trabalho3.md`: versao editavel do relatorio.
- `entrega-trabalho3/relatorio-trabalho3.pdf`: relatorio final diagramado.

Os testes e documentos internos de projeto ficarao fora de `entrega-trabalho3/`, para nao poluir o pacote a ser enviado.

## Fluxo de dados

1. Ler `trabalho3-input.txt` como UTF-8, remover apenas linhas vazias e preservar a ordem original.
2. Rejeitar arquivo vazio, gerar identificadores posicionais unicos e informar as contagens real e unica de frases.
3. Para cada modelo, criar uma colecao ChromaDB separada com funcao de embedding propria.
4. Inserir as 1.000 frases em lotes, usando identificadores estaveis derivados da posicao original.
5. Consultar as 10 perguntas com `n_results=5`, incluindo documentos e distancias.
6. Escrever um arquivo de texto deterministico para cada modelo.
7. Calcular comparacoes por pergunta: distancia media, sobreposicao entre top-5 e avaliacao de relevancia baseada em conjuntos de frases tematicas identificadas no corpus.
8. Produzir o relatorio com metodologia, resultados, limitacoes e conclusao sobre eventual vencedor.

## Formato de saida

Cada arquivo de resultados tera cabecalho com modelo, quantidade de frases e parametros de consulta. Em seguida, cada pergunta aparecera numerada e sera seguida por cinco linhas ordenadas do resultado mais proximo ao menos proximo. Cada linha contera posicao, distancia numerica e frase original. A codificacao sera UTF-8.

## Metodo de avaliacao

A avaliacao combinara tres evidencias:

1. **Inspecao de relevancia**: para cada pergunta, verificar quantas das cinco frases recuperadas tratam diretamente do tema solicitado.
2. **Reciprocal Rank e Recall@5**: usar como referencia as frases inequivocamente tematicas presentes no corpus, catalogadas por categorias semanticas. O catalogo sera derivado do proprio arquivo e documentado, sem alterar a busca.
3. **Sobreposicao dos rankings**: medir quantas frases os dois modelos compartilham em cada top-5 e comparar a ordem.

Distancias de modelos diferentes nao serao comparadas como se estivessem na mesma escala. Elas servirao apenas para analisar separacao e ordenacao dentro de cada modelo. O vencedor, se houver, sera definido principalmente por Recall@5, MRR e relevancia qualitativa. Empate ou vantagem pequena sera relatado com a devida cautela.

## Tratamento de erros

- Arquivo ausente: mensagem clara e codigo de saida diferente de zero.
- Arquivo vazio: erro antes de carregar modelos.
- Linhas duplicadas: aceitas como documentos distintos somente se o arquivo real as contiver, preservando IDs posicionais; a contagem de frases unicas sera registrada.
- Dependencias ou download de modelo indisponiveis: mensagem que identifica o componente ausente e mantem os artefatos ja produzidos.
- Colecao com menos de cinco documentos: consultar o minimo entre cinco e o tamanho do corpus.
- Saida parcial: gravar primeiro em arquivo temporario e substituir o destino apenas ao concluir.

## Testes

Os testes automatizados usarao uma colecao falsa pequena ou uma funcao de embedding deterministica injetada, evitando rede e modelos pesados. Eles cobrirao: UTF-8 e linhas vazias; corpus vazio; IDs estaveis; exatamente cinco resultados quando possivel; ordenacao; inclusao de todas as perguntas; nomes dos arquivos; e serializacao das metricas. A verificacao de integracao executara o programa real com os dois modelos e validara 50 resultados por arquivo.

## Relatorio

O relatorio tera: objetivo, ambiente e dados, metodologia, resultados por pergunta, metricas agregadas, comparacao dos modelos, explicacao tecnica das diferencas, limitacoes e conclusao. Tambem destacara que o corpus recebido possui 1.000 frases, embora o enunciado mencione 10.000.

## Criterios de aceitacao

- O programa usa ChromaDB e ambos os modelos especificados, sem substituir nenhum deles.
- Todas as 1.000 frases fornecidas sao indexadas para cada modelo.
- Cada arquivo de saida contem as 10 perguntas e cinco frases por pergunta, totalizando 50 resultados.
- Os arquivos podem ser regenerados por um unico comando documentado.
- Os resultados usados no relatorio correspondem aos arquivos entregues.
- Os testes automatizados passam sem depender de acesso a rede.
- O PDF final e renderizado e inspecionado sem cortes, sobreposicoes ou caracteres corrompidos.
- Todos os artefatos de envio ficam reunidos em `entrega-trabalho3/`, sem depender de arquivos localizados fora dela.
