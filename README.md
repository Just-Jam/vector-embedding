# vector-embedding
A simple script to vectorize data for language model embeddings

# Use Cases
## 1. Information Retrieval
Vectorizing our data allows our language model (LLM) to perform a similarity search, improving the relevance of search results. The vector embedding is simply a representation of data that LLMs can easily understand and process. The logic of similarity search is rather similar to reading comprehension. When given a 3000-word length text and asked about the actions of the character "John", the fastest method would be to scan through the text, mentally note down sentences mentioning "John", and answer the questions with the sentences in mind. Vector Embedding and similarity search allow our LLM to do the same for any text-based information much larger than what you could normally fit in a prompt, say, the entirety of the Harry Potter book series. The LLM finds parts of the data that are relevant to the user's query and uses that to answer the query.
