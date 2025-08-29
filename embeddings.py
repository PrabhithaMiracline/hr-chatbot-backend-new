# fixed_embeddings_setup.py
import os
import pandas as pd
import chromadb
from chromadb.config import Settings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# --- Configuration ---
PERSIST_DIRECTORY = r"D:\hr-assistant-chatbot\embeddings"
COLLECTION_NAME = "hr_faqs"
MODEL_PATH = r"D:\hr-assistant-chatbot\hugging_face"
CSV_PATH = r"D:\hr-assistant-chatbot\HR_FAQs_Comprehensive_Dataset (1).csv"

def setup_vector_db():
    """Initialize and populate the vector database"""
    
    # 1. Initialize embedding model
    print("Initializing embedding model...")
    embedding_function = HuggingFaceEmbeddings(model_name=MODEL_PATH)
    print("Embedding model initialized.")
    
    # 2. Initialize ChromaDB with proper settings
    print("Initializing ChromaDB client...")
    chroma_settings = Settings(
        anonymized_telemetry=False,
        is_persistent=True,
        persist_directory=PERSIST_DIRECTORY
    )
    
    # Create directory if it doesn't exist
    os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
    
    # 3. Initialize Chroma vector store
    vector_db = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_function,
        persist_directory=PERSIST_DIRECTORY
    )
    
    # 4. Check if collection already has data
    try:
        existing_count = vector_db._collection.count()
        print(f"Existing embeddings in collection: {existing_count}")
        
        if existing_count > 0:
            print("Collection already has data. Use clear_and_reload=True to reload.")
            return vector_db
    except Exception as e:
        print(f"Error checking existing data: {e}")
    
    # 5. Load and process CSV data
    print(f"Loading CSV data from: {CSV_PATH}")
    try:
        df = pd.read_csv(CSV_PATH)
        print(f"Loaded {len(df)} rows from CSV")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None
    
    # 6. Add data to vector database
    print("Adding data to vector database...")
    texts = []
    metadatas = []
    
    for index, row in df.iterrows():
        # Use both question and answer as searchable text
        text = f"Question: {row['Question']}\nAnswer: {row['Answer']}"
        texts.append(text)
        
        metadata = {
            "question": str(row["Question"]),
            "answer": str(row["Answer"]),
            "category": str(row["Category"]),
            "difficulty": str(row["Difficulty Level"]),
            "keywords": str(row["Keywords"]),
            "source": "hr_faq_csv"
        }
        metadatas.append(metadata)
        
        if (index + 1) % 10 == 0:
            print(f"Processed {index + 1} rows...")
    
    # Add texts in batches for better performance
    batch_size = 50
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_metadatas = metadatas[i:i+batch_size]
        vector_db.add_texts(batch_texts, batch_metadatas)
        print(f"Added batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
    
    # 7. Verify the data was added
    final_count = vector_db._collection.count()
    print(f"Final count of embeddings: {final_count}")
    
    return vector_db

def test_vector_db():
    """Test the vector database with a sample query"""
    print("\n--- Testing Vector Database ---")
    
    embedding_function = HuggingFaceEmbeddings(model_name=MODEL_PATH)
    
    vector_db = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_function,
        persist_directory=PERSIST_DIRECTORY
    )
    
    # Test query
    test_queries = [
        "How can I apply for leave?",
        "What is the dress code policy?",
        "How do I reset my password?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        try:
            results = vector_db.similarity_search(query, k=3)
            print(f"Found {len(results)} results:")
            for i, result in enumerate(results):
                print(f"  {i+1}. {result.page_content[:100]}...")
                print(f"     Metadata: {result.metadata}")
        except Exception as e:
            print(f"Error searching: {e}")

def clear_vector_db():
    """Clear the existing vector database"""
    try:
        chroma_client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
        chroma_client.delete_collection(name=COLLECTION_NAME)
        print(f"Cleared collection '{COLLECTION_NAME}'")
    except Exception as e:
        print(f"Error clearing collection: {e}")

# --- Flask App Integration Helper ---
def get_vector_db_for_flask():
    """Get vector database instance for Flask app"""
    embedding_function = HuggingFaceEmbeddings(model_name=MODEL_PATH)
    
    vector_db = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_function,
        persist_directory=PERSIST_DIRECTORY
    )
    
    # Verify it has data
    count = vector_db._collection.count()
    if count == 0:
        print("WARNING: Vector database is empty! Run setup_vector_db() first.")
    else:
        print(f"Vector database loaded with {count} embeddings")
    
    return vector_db

if __name__ == "__main__":
    # Uncomment the operation you want to perform:
    
    # Setup vector database (run this first)
    vector_db = setup_vector_db()
    
    # Test the vector database
    test_vector_db()
    
    # To clear and reload (uncomment if needed):
    # clear_vector_db()
    # setup_vector_db()
