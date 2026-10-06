# Trabalho 3 - Busca semântica com ChromaDB

**Disciplina:** Tópicos Especiais em Banco de Dados  
**Professor:** Leonardo Fiório Soares  
**Data da execução:** 25 de setembro de 2026

## Resumo

Este trabalho comparou os modelos de embeddings `all-MiniLM-L6-v2` e `all-mpnet-base-v2` na recuperação de frases em português usando ChromaDB. Cada uma das dez perguntas foi usada para recuperar as cinco frases mais próximas entre as 1.000 frases únicas do arquivo recebido. O `all-mpnet-base-v2` obteve os melhores indicadores agregados: Recall@5 médio de 0,0833 e MRR de 0,20, contra 0,0450 e 0,07 do `all-MiniLM-L6-v2`. Portanto, o MPNet foi o vencedor relativo deste experimento. A vantagem, porém, deve ser interpretada com cautela: ambos os modelos são identificados como modelos em inglês e produziram muitos resultados irrelevantes para consultas e documentos em português.

## 1. Objetivo

O objetivo foi implementar uma busca vetorial com ChromaDB e comparar os modelos exigidos pelo enunciado. Para cada modelo, o sistema deveria associar cinco frases do corpus a cada uma das dez perguntas, gerar um arquivo de saída e analisar as diferenças.

## 2. Base de dados

O PDF da atividade informa que `trabalho3-input.txt` contém 10.000 frases únicas. A verificação do arquivo efetivamente entregue encontrou 1.000 linhas não vazias e 1.000 frases únicas, codificadas em UTF-8. Nenhuma frase foi criada ou duplicada para completar a quantidade indicada no enunciado: todas as 1.000 frases reais foram indexadas em ambos os experimentos.

O corpus mistura frases diretamente relacionadas às dez áreas investigadas com grande quantidade de notícias genéricas sobre produtos tecnológicos. Essa composição torna o teste desafiador, pois uma consulta deve distinguir conteúdo temático de frases que apenas contêm palavras genéricas como tecnologia, empresa, software ou produto.

## 3. Metodologia

O programa foi desenvolvido em Python 3.12.10 com ChromaDB 1.5.9, sentence-transformers 5.7.0 e PyTorch 2.10.0 em CPU. Para cada modelo foi criada uma coleção ChromaDB independente e efêmera. As mesmas 1.000 frases foram inseridas com identificadores posicionais estáveis e a coleção foi configurada com distância cosseno.

As dez perguntas foram consultadas sem tradução, expansão, palavras-chave auxiliares ou reranqueamento. Para cada pergunta foram solicitados cinco documentos. Essa decisão garante que a comparação reflita os embeddings dos modelos, sem interferência de outro método.

Além da inspeção qualitativa, foi criado um catálogo de referência com frases inequivocamente ligadas a cada tema. O catálogo foi usado somente na avaliação, nunca na busca. Foram calculadas três medidas:

- **Recall@5:** proporção das frases de referência do tema recuperadas entre as cinco primeiras.
- **MRR (Mean Reciprocal Rank):** média do inverso da posição da primeira frase de referência recuperada; vale 1 quando a primeira posição é relevante e 0 quando nenhuma referência aparece.
- **Sobreposição:** quantidade de frases iguais presentes nos dois conjuntos top-5 para cada pergunta.

As distâncias absolutas não foram comparadas entre modelos porque cada espaço vetorial possui distribuição própria. Elas servem para ordenar resultados dentro do mesmo modelo.

## 4. Resultados quantitativos

| Modelo | Dimensão do embedding | Recall@5 médio | MRR | Perguntas com referência no top-5 |
|---|---:|---:|---:|---:|
| all-MiniLM-L6-v2 | 384 | 0,0450 | 0,0700 | 2 de 10 |
| all-mpnet-base-v2 | 768 | 0,0833 | 0,2000 | 3 de 10 |

O detalhamento por pergunta é apresentado abaixo. A coluna `R` mostra o número de frases do catálogo de referência recuperadas no top-5; `Total` é a quantidade de referências disponíveis para o tema.

| Nº | Tema resumido | MiniLM R/Total | MPNet R/Total | Sobreposição top-5 |
|---:|---|---:|---:|---:|
| 1 | IA e mercado de trabalho | 0/5 | 0/5 | 1 |
| 2 | Riscos de segurança | 0/6 | 0/6 | 0 |
| 3 | Computação em nuvem | 0/4 | 1/4 | 0 |
| 4 | Big data e decisões | 1/4 | 0/4 | 1 |
| 5 | Automação industrial | 0/5 | 0/5 | 1 |
| 6 | Tecnologia na saúde | 0/6 | 0/6 | 0 |
| 7 | Sustentabilidade | 0/6 | 2/6 | 0 |
| 8 | Trabalho remoto | 1/5 | 0/5 | 0 |
| 9 | Proteção por cibersegurança | 0/4 | 0/4 | 0 |
| 10 | Startups e novos negócios | 0/4 | 1/4 | 1 |

Somando as dez perguntas, apenas quatro documentos apareceram no top-5 de ambos os modelos para a mesma consulta. A baixa sobreposição demonstra que as arquiteturas produziram espaços vetoriais bastante diferentes neste corpus em português.

## 5. Análise qualitativa

O `all-MiniLM-L6-v2` encontrou uma frase diretamente relacionada a big data na segunda posição da pergunta 4: "Interpretar volumes gigantescos de informação reduz a incerteza nas escolhas estratégicas de uma organização." Também recuperou uma frase sobre flexibilidade fora do escritório na quinta posição da pergunta 8. Entretanto, falhou em perguntas com referências claras sobre saúde, automação industrial e cibersegurança.

O `all-mpnet-base-v2` recuperou uma frase de armazenamento remoto na segunda posição da pergunta sobre nuvem. Seu melhor resultado ocorreu em sustentabilidade: painéis solares e turbinas eólicas apareceram nas posições 2 e 3. Para startups, colocou na primeira posição a frase sobre ideias iniciadas em garagens que se transformam em empresas. Ainda assim, também apresentou erros fortes, como cinco frases sobre teclado mecânico para a pergunta de saúde e resultados sobre carregadores e aplicativos de corrida para cibersegurança.

Há um padrão importante: os dois modelos frequentemente se apoiaram em coincidências lexicais ou associações frágeis. Termos como "trabalho", "empresas", "tecnologia" e "processos" aproximaram frases que compartilhavam vocabulário, mas não respondiam ao sentido completo da pergunta.

## 6. Por que os modelos diferiram

Segundo os cartões oficiais, o MiniLM gera vetores de 384 dimensões, enquanto o MPNet gera vetores de 768 dimensões. A documentação do Sentence Transformers descreve o `all-mpnet-base-v2` como o modelo de maior qualidade entre os modelos originais `all-*`, enquanto o `all-MiniLM-L6-v2` é aproximadamente cinco vezes mais rápido e ainda mantém boa qualidade. Isso cria uma troca esperada: MiniLM privilegia eficiência; MPNet usa uma representação maior e tende a preservar mais nuances semânticas.

Os dois modelos foram ajustados com mais de um bilhão de pares de sentenças usando objetivo contrastivo, mas seus cartões os identificam como modelos em inglês. O experimento foi inteiramente em português. A documentação do Sentence Transformers lista modelos multilíngues específicos, inclusive variantes MiniLM e MPNet treinadas para mais de 50 idiomas, incluindo português. Como inferência, o desalinhamento entre idioma de treinamento e idioma do corpus é a explicação mais forte para o baixo desempenho absoluto observado.

O corpus também contribuiu para os erros. Ele contém muitas frases quase duplicadas sobre produtos, preços, avaliações e defeitos. Essas repetições temáticas formam regiões densas no espaço vetorial e podem ocupar o top-5 mesmo quando existe uma frase menor, mas semanticamente correta, em outra região.

## 7. Houve um vencedor?

Sim, mas apenas em sentido relativo: o `all-mpnet-base-v2` venceu este experimento. Ele quase dobrou o Recall@5 médio, de 0,0450 para 0,0833, elevou o MRR de 0,07 para 0,20 e colocou referências corretas no top-5 de três perguntas, contra duas do MiniLM. A vantagem também está alinhada à documentação oficial, que posiciona MPNet como a opção de maior qualidade e MiniLM como a opção mais rápida.

Não seria correto concluir que o MPNet apresentou resultado satisfatório. Ele não recuperou nenhuma referência em sete das dez perguntas. O resultado adequado é: **MPNet foi o melhor dos dois modelos exigidos, porém ambos tiveram baixa adequação ao corpus em português.**

## 8. Limitações

- O arquivo recebido possui 1.000 frases, embora o enunciado mencione 10.000.
- O catálogo de referência inclui somente frases claramente temáticas; frases parcialmente relacionadas podem não contar na métrica.
- O experimento compara qualidade de recuperação, não tempo de execução nem consumo de memória.
- Os modelos exigidos são indicados para inglês, enquanto consultas e corpus estão em português.
- O top-5 é pequeno e torna cada erro de ordenação mais significativo.

## 9. Conclusão

O trabalho demonstrou o fluxo completo de busca vetorial: criação de embeddings, indexação no ChromaDB, consulta por similaridade cosseno e comparação de rankings. O `all-mpnet-base-v2` foi o vencedor relativo, sobretudo pelos acertos em nuvem, sustentabilidade e startups. O `all-MiniLM-L6-v2` conseguiu acertos pontuais em big data e trabalho remoto, mas teve métrica agregada inferior.

A principal conclusão é que escolher um modelo apenas pela reputação geral não basta. A adequação ao idioma e ao tipo de consulta é decisiva. Em uma continuação fora das restrições do enunciado, seria recomendável repetir o teste com `paraphrase-multilingual-MiniLM-L12-v2` ou `paraphrase-multilingual-mpnet-base-v2`, usando o mesmo corpus e as mesmas métricas.

## Referências

1. Sentence Transformers. **Pretrained Models**. Disponível em: https://sbert.net/docs/sentence_transformer/pretrained_models.html. Acesso em: 25 set. 2026.
2. Hugging Face. **sentence-transformers/all-MiniLM-L6-v2 - Model Card**. Disponível em: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2. Acesso em: 25 set. 2026.
3. Hugging Face. **sentence-transformers/all-mpnet-base-v2 - Model Card**. Disponível em: https://huggingface.co/sentence-transformers/all-mpnet-base-v2. Acesso em: 25 set. 2026.
