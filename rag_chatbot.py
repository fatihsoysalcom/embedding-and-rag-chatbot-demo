import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# --- Mock Data --- 
# In a real scenario, these would be actual documents.
# We'll use simple strings for demonstration.
documents = [
    "Yapay zeka, bilgisayarların insan benzeri zeka sergilemesini sağlayan bir teknolojidir.",
    "Makine öğrenmesi, yapay zekanın bir alt dalıdır ve algoritmaların verilerden öğrenmesini sağlar.",
    "Derin öğrenme, çok katmanlı sinir ağları kullanarak karmaşık örüntüleri tanır.",
    "Doğal dil işleme (NLP), bilgisayarların insan dilini anlamasını ve işlemesini amaçlar.",
    "Büyük dil modelleri (LLM'ler), devasa metin verileri üzerinde eğitilmiş gelişmiş NLP modelleridir."
]

# --- Mock Embedding Function --- 
# In a real scenario, this would use a pre-trained embedding model (e.g., Sentence-BERT).
# For this demo, we'll use a very simplistic (and unrealistic) approach: word counts.
def get_embedding(text):
    words = text.lower().split()
    # Create a vocabulary from all documents
    all_words = set()
    for doc in documents:
        all_words.update(doc.lower().split())
    
    vector = np.zeros(len(all_words))
    word_to_index = {word: i for i, word in enumerate(sorted(list(all_words))))}
    
    for word in words:
        if word in word_to_index:
            vector[word_to_index[word]] += 1
    return vector.reshape(1, -1)

# Pre-compute embeddings for all documents
document_embeddings = [get_embedding(doc) for doc in documents]

# --- RAG Logic --- 
def answer_question_with_rag(question):
    # 1. Embed the question
    question_embedding = get_embedding(question)
    
    # 2. Find relevant documents (Retrieval)
    similarities = []
    for doc_emb in document_embeddings:
        # Calculate cosine similarity between question and document embeddings
        sim = cosine_similarity(question_embedding, doc_emb)[0][0]
        similarities.append(sim)
    
    # Get the index of the most similar document
    most_similar_doc_index = np.argmax(similarities)
    retrieved_document = documents[most_similar_doc_index]
    
    # 3. Generate the answer (Augmented Generation)
    # In a real LLM scenario, we'd pass the question and the retrieved_document
    # to an LLM to generate a context-aware answer.
    # For this demo, we'll simulate by constructing a simple answer.
    
    # Basic keyword matching for demonstration
    if "yapay zeka" in question.lower() and "nedir" in question.lower():
        return f"'{question}' sorunuz için en alakalı bilgi: '{retrieved_document}'. Yapay zeka, bilgisayarların insan benzeri zeka sergilemesini sağlayan bir teknolojidir."
    elif "makine öğrenmesi" in question.lower():
        return f"'{question}' sorunuz için en alakalı bilgi: '{retrieved_document}'. Makine öğrenmesi, yapay zekanın bir alt dalıdır ve algoritmaların verilerden öğrenmesini sağlar."
    else:
        return f"'{question}' sorunuz için en alakalı bilgi: '{retrieved_document}'. Bu belge sorunuzla ilgili temel bilgileri içermektedir."

# --- Example Usage --- 
if __name__ == "__main__":
    print("Sohbet Botu Hazır! Sorularınızı sorun.")
    
    questions = [
        "Yapay zeka nedir?",
        "Makine öğrenmesi hakkında bilgi verir misin?",
        "Derin öğrenme nasıl çalışır?",
        "NLP'nin amacı nedir?"
    ]
    
    for q in questions:
        answer = answer_question_with_rag(q)
        print(f"Soru: {q}")
        print(f"Cevap: {answer}")
        print("---")
