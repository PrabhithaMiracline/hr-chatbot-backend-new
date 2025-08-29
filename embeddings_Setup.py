import pandas as pd
#import requests
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModel

df=pd.read_csv("D:\hr-assistant-chatbot\HR_FAQs_Comprehensive_Dataset (1).csv")

model_path=""
embedding_function = HuggingFaceEmbeddings(model_name="C:\\Users\\prabhitha.m\\Desktop\\hr-chatbot\\Models_Rebuilt")
embedding_model = HuggingFaceEmbeddings(model_name=model_path)

# requests.packages.urllib3.disable_warnings()
vector_db=Chroma(collection_name="hr_faqs",embedding_function=embedding_function,persist_directory="C:\\Users\\prabhitha.m\\Desktop\\hr-chatbot\\embeddings\\")
vector_db = Chroma(
    persist_directory=r"",
    embedding_function=embedding_function
)


# for index,row in df.iterrows():
#     metadata={
#         "answer":row["Answer"],
#         "category":row["Category"],
#         "difficulty":row["Difficulty Level"],
#         "keywords":row["Keywords"]
#     }
#     vector_db.add_texts([row["Question"]],[metadata])

test_embedding = embedding_function.embed_query("How can I apply for leave?")
print(test_embedding)

for index, row in df.iterrows():
    print(f"Processing row {index}: {row['Question']}")
    metadata = {
        "answer": row["Answer"],
        "category": row["Category"],
        "difficulty": row["Difficulty Level"],
        "keywords": row["Keywords"]
    }
    vector_db.add_texts([row["Question"]], [metadata])



# vector_db.persist()
# print("Vector database saved successfully!")
print(vector_db.persist_directory)

print(vector_db._collection.count())  # Should return the number of embeddings stored


# chatbot_routes.py
# from flask import BluePrint,request,jsonify

# chatbot_bp=BluePrint('chatbot',__name__)

# @chatbot_bp.route('/query',methods=['POST'])

# def handle_query():
#     data=request.json
#     query=data.get('query','')
#     response={"message":f"You asked: {query}.Feature not implemented yet"}
#     return jsonify(response)
 



