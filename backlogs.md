## Retrieval Backlog

- **Chunk focus controls**  
  Use token-aware chunkers (~150–300 tokens) and enforce tool/section boundaries so each snippet stays coherent; this prevents blended guidance, keeps prompts tighter, and should improve answer attribution.

- **Contextual headers in chunks**  
  Prepend section titles, tool names, and role tags to each chunk’s text so lexical matches become easier and embeddings capture richer semantics, giving the retriever more hooks when users mention specific tools.

- **Rich metadata schema**  
  Persist section (frontline/team/mandator), module, tool number, and page for every chunk; these fields unlock role-aware filtering, better auditing of retrieved evidence, and future UI facets.

- **Hybrid retrieval path**  
  Add a lightweight BM25/keyword index alongside FAISS, then fuse scores or re-rank so exact terms like “frontline negotiator” surface even when embeddings miss them, reducing brittle failures.

- **Role-biased reranker**  
  Once metadata exists, add a reranking layer that boosts chunks matching the inferred role/tool context of the question, ensuring responses cite the most relevant manual persona without discarding diversity.

