# Trabalho 3 - ChromaDB

Esta pasta contém o código, os dados, os dois arquivos de saída gerados e o relatório da comparação entre `all-MiniLM-L6-v2` e `all-mpnet-base-v2`.

## Arquivos principais

- `trabalho3.py`: código Python completo.
- `trabalho3-input.txt`: corpus fornecido para a atividade.
- `trabalho3-output-all-MiniLM-L6-v2.txt`: saída gerada pelo primeiro modelo.
- `trabalho3-output-all-mpnet-base-v2.txt`: saída gerada pelo segundo modelo.
- `relatorio-trabalho3.pdf`: relatório final.
- `relatorio-trabalho3.md`: versão editável do relatório.
- `trabalho3-metricas.json`: métricas usadas no relatório.
- `requirements.txt`: dependências necessárias.

## Como executar

Requer Python 3.11 ou mais recente e acesso à internet na primeira execução para baixar os modelos.

No PowerShell, dentro desta pasta:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python trabalho3.py
```

O programa lê `trabalho3-input.txt` e substitui os dois arquivos `trabalho3-output-*.txt` e `trabalho3-metricas.json` somente após concluir cada gravação.

Também é possível informar outros caminhos:

```powershell
python trabalho3.py --input caminho\entrada.txt --output-dir caminho\saida
```

## Observação sobre o corpus

O enunciado afirma que o arquivo possui 10.000 frases únicas. O arquivo efetivamente fornecido contém 1.000 linhas não vazias, todas únicas. O programa processa integralmente o arquivo recebido e registra essas contagens nas métricas.

## Reprodutibilidade

Os resultados entregues foram gerados com Python 3.12.10, ChromaDB 1.5.9, sentence-transformers 5.7.0 e PyTorch 2.10.0 em CPU. As coleções usam distância cosseno, as mesmas 10 perguntas e `n_results=5` para ambos os modelos.
