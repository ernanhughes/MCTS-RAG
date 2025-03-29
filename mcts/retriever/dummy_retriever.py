class DummyRetriever:
    def retrieve(self, query):
        return [{"text": f"Simulated document for query: {query}"}]
