# SDD ledger - plan: docs/superpowers/plans/2026-09-25-trabalho3-chromadb.md

Setup Ruling: a pasta nao e um repositorio Git; executar no diretorio atual sem worktree ou commits - necessario para produzir os artefatos solicitados - custo se errado: ausencia de historico Git, mitigada por testes e arquivos separados.
Pre-flight: Task 1 produz dataclasses, parser, IDs e formatacao consumidos pelas Tasks 2-4; assinaturas consistentes.
Pre-flight: Task 2 produz busca, escrita atomica e orquestracao consumidas pelas Tasks 3-4; `write_metrics` sera resolvida na Task 3 antes da execucao do CLI.
Pre-flight: Task 3 produz resultados e JSON consumidos pela Task 4; nomes de arquivos consistentes com a especificacao.
Pre-flight Ruling: o plano referencia `tests/test_integration_outputs.py` antes de cria-lo - criar o teste na Task 4 e executar validacao estrutural equivalente na Task 3 - custo se errado: defeito estrutural tardio, mitigado pela verificacao final completa.
Task 1: complete (sem commits: pasta nao Git; tests: `python -m pytest tests/test_trabalho3.py -v` -> 4 passed).
Task 2: Ruling: implementar o CLI na Task 3 junto com `write_metrics`, pois o ponto de entrada depende desse contrato ainda inexistente - preserva coesao e evita funcao quebrada intermediaria - custo se errado: Task 2 isoladamente nao e executavel, mitigado antes da integracao real.
Task 2: complete (sem commits: pasta nao Git; tests: `python -m pytest tests/test_trabalho3.py -v` -> 8 passed).
Task 3: complete (sem commits: pasta nao Git; pipeline real executado com 1.000 frases, dois modelos, 100 resultados e JSON de metricas; validacao de integracao -> 4 passed).
Task 4: complete (sem commits: pasta nao Git; relatorio PDF de 5 paginas renderizado e inspecionado; suite completa -> 15 passed antes da revisao).
Final review: self-review (delegacao nao solicitada e pasta sem Git).
Final: fixed insercao unica que poderia exceder o limite do ChromaDB com 10.000 frases - `test_run_model_adds_large_corpus_in_bounded_batches` RED->GREEN.
Final: fixed cabecalho incorreto para corpus com menos de cinco frases - `test_format_output_reports_actual_count_for_short_corpus` RED->GREEN.
Final verification: `python -m pytest -v` -> 17 passed; compilacao Python OK; 100 resultados pertencem ao corpus; hashes do input identicos; PDF A4 com 5 paginas; zero arquivos temporarios.
Final Ruling: preservar este ledger fora da pasta de entrega porque nao existe historico Git que o substitua - custo se errado: arquivo interno adicional fora do pacote, sem impacto na entrega.
