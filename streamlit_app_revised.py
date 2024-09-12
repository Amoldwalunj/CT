import streamlit as st
import pandas as pd
import boto3
from langchain_community.embeddings import BedrockEmbeddings
import chromadb
from langchain_chroma import Chroma


aws_access_key_id = ''
aws_secret_access_key = ''

# Initialize Bedrock client
@st.cache_resource
def init_bedrock():
    return boto3.client(
        service_name="bedrock-runtime",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name='us-east-1'
    )

bedrock = init_bedrock()

# Initialize Bedrock embeddings
@st.cache_resource
def init_bedrock_embeddings():
    return BedrockEmbeddings(model_id="cohere.embed-english-v3", client=bedrock)

bedrock_embeddings = init_bedrock_embeddings()

# Initialize ChromaDB client
@st.cache_resource
def init_chroma_client():
    return chromadb.PersistentClient(path="./chroma_db")

persistent_client = init_chroma_client()

# Initialize Chroma database
@st.cache_resource
def init_chroma_db():
    return Chroma(
        client=persistent_client,
        collection_name="candidate_skills",
        embedding_function=bedrock_embeddings,
    )

db = init_chroma_db()

@st.cache_data
def load_data():
    return pd.read_excel("./Ranking algo/datasets/candidate_data_bullhorn1.xlsx")

@st.cache_data
def preprocess_data(df):
    if 'skillSet' not in df.columns:
        raise ValueError("'skillSet' column not found in the dataframe")
    
    filtered_df = df[df['skillSet'].notnull()].copy()
    filtered_df['skills'] = filtered_df['skillSet'].str.replace('[', '', regex=False) \
                                                   .str.replace(']', '', regex=False) \
                                                   .str.replace('[^\w\s]', '') \
                                                   .str.lower()
    return filtered_df

def get_top_candidates(query, df, top_percentage=2):
    k = max(int(len(df) * top_percentage / 100), 1)
    similar_docs = db.similarity_search_with_score(query, k=k)
    
    candidate_ids = [doc[0].metadata['id'] for doc in similar_docs]
    similarity_scores = [1 - doc[1] for doc in similar_docs]
    
    top_candidates = df[df['id'].isin(candidate_ids)].copy()
    score_dict = dict(zip(candidate_ids, similarity_scores))
    top_candidates['Similarity Score'] = top_candidates['id'].map(score_dict)
    top_candidates = top_candidates.sort_values('Similarity Score', ascending=False)
    
    return top_candidates

# Streamlit app
st.title("Top Candidates Selection App")

st.write("""
This app takes a Job Description (JD) as input and shows the top 2% candidates based on skill matching.
""")

# Load and preprocess data
df = load_data()
filtered_df = preprocess_data(df)

job_description = st.text_area("Enter Job Description:", height=250)

if st.button("Find Top Candidates"):
    if job_description:
        st.write("Processing...")

        # Get top 2% candidates
        top_candidates_df = get_top_candidates(job_description, filtered_df, top_percentage=2)
        
        # Display results
        st.write(f"Top {len(top_candidates_df)} Candidates (2% of total {len(filtered_df)}):")
        st.dataframe(top_candidates_df[['id', 'firstName', 'lastName', 'skills', 'Similarity Score']])

        # Detailed view of top candidates
        st.write("Detailed View of Top Candidates:")
        for _, candidate in top_candidates_df.iterrows():
            with st.expander(f"{candidate['firstName']} {candidate['lastName']} - Similarity Score: {candidate['Similarity Score']:.4f}"):
                st.write(f"**ID:** {candidate['id']}")
                st.write(f"**Skills:** {candidate['skills']}")
                if 'title' in candidate:
                    st.write(f"**Title:** {candidate['title']}")
                if 'educationDegree' in candidate:
                    st.write(f"**Education:** {candidate['educationDegree']}")
    else:
        st.warning("Please enter a job description.")

# Add a section to display some stats about the dataset
st.sidebar.header("Dataset Statistics")
st.sidebar.write(f"Total Candidates: {len(df)}")
st.sidebar.write(f"Candidates with Skills: {len(filtered_df)}")