import streamlit as st
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
# from DocumentIndexer import DocumentIndexer
from sentence_transformers import CrossEncoder,SentenceTransformer
from Retriever import Retriever
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to load Mistral model and tokenizer
@st.cache(allow_output_mutation=True)
def load_model_and_tokenizer():
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    model.to(device)
    return model, tokenizer

@st.cache(allow_output_mutation=True)
def load_embedding_model():
    model=SentenceTransformer('thenlper/gte-base', device=device)
    return model

@st.cache(allow_output_mutation=True)
def load_cross_encoder():
    model=CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2",device='cpu')
    return model
    
embedding_model=load_embedding_model()
cr_model=load_cross_encoder()


# Initialize DocumentIndexer for document retrieval
retriever = Retriever(embedding_model,cr_model)

# Function to retrieve the answer using the provided query
def get_answer(query, model, tokenizer,embedding_model,cr_model):
    # Retrieve relevant documents
    results = retriever.get_knn_results(query)
    results=retriever.rerank_results(query,results)
    results['document_id']=[i for i in range(0,len(results))]
    # Prepare context for the Mistral model
    context = "Answer the question based on the provided context. Be concise, straightforward and just return the conclusion. If you don't know the answer, just say that 'I don't know', don't try to make up an answer."+ "Question: " + query + ". " + "Context: " + results[0:10][['document_id', 'document_title']].to_json(orient='records')

    # Prepare input for Mistral model
    messages = [{"role": "user", "content": context}]
    encoded_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)

    # Generate response from Mistral model
    generated_ids = model.generate(encoded_inputs, max_new_tokens=1000, do_sample=True, temperature=0.1)
    decoded_response = tokenizer.batch_decode(generated_ids)

    # Extract answer using regex from decoded response
    pattern = r'\[/INST\](.*?)<\/s>'
    match = re.search(pattern, decoded_response[0], re.DOTALL)

    if match:
        extracted_answer = match.group(1).strip()
        return extracted_answer
    else:
        return "Answer not found."

# Streamlit app
def main():
    st.title("Question Answering System")

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    # Input field for search query
    query = st.text_input("Enter your query:", key="query_input")

    # Trigger answer retrieval when user presses Enter
    if query:
        if st.session_state.query_input:
            answer = get_answer(query, model, tokenizer,embedding_model,cr_model)
            st.write("Extracted Answer:", answer)

# Run the Streamlit app
main()
