from mcts.memory import PGVectorMemory

def retrieval_reasoning(self, query: str, memory: PGVectorMemory):
    cached_results = memory.retrieve_similar(query, top_k=1)

    if cached_results:
        retrieved_docs = cached_results[0][1]
        source = "memory"
    else:
        retrieved_docs = self.retriever.retrieve(query)
        memory.save_retrieval(query, retrieved_docs)
        source = "live"

    print(f"[RetrievalReasoning] Source: {source}")
    reasoning_input = self._construct_prompt_with_retrieval(query, retrieved_docs)
    return self.model.generate(reasoning_input)
