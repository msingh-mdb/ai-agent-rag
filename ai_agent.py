import streamlit as st
import boto3
from langchain_aws import ChatBedrock, BedrockEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from botocore.exceptions import ClientError, NoCredentialsError

# --- Page Configuration ---
st.set_page_config(page_title="AWS Bedrock & MongoDB Agent", page_icon="🤖", layout="wide")

st.title("🤖 AI Agent: Bedrock (Titan V2) + MongoDB")

# --- Session State Initialization ---
if "is_connected" not in st.session_state:
    st.session_state.is_connected = False
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Helper: Response Cleaner ---
def clean_agent_output(output_data):
    """
    Parses the raw output from the agent. 
    If it's a list of blocks (Anthropic format), extracts the text.
    If it's a string, returns it as is.
    """
    if isinstance(output_data, list):
        # Join all blocks of type 'text'
        text_parts = [
            block.get("text", "") 
            for block in output_data 
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "".join(text_parts).strip()
    return str(output_data).strip()

# --- Sidebar: Configuration & Credentials ---
with st.sidebar:
    st.header("🔐 Credentials & Config")
    
    with st.expander("AWS Configuration", expanded=True):
        aws_access_key = st.text_input("AWS Access Key ID", type="password")
        aws_secret_key = st.text_input("AWS Secret Access Key", type="password")
        aws_session_token = st.text_input("AWS Session Token", type="password", help="Required for temporary credentials")
        aws_region = st.text_input("AWS Region", value="us-east-1")

    with st.expander("MongoDB Configuration", expanded=True):
        mongo_conn_str = st.text_input("MongoDB Connection String", type="password", help="mongodb+srv://...")
        db_name = st.text_input("Database Name", value="my_db")
        collection_name = st.text_input("Collection Name", value="my_collection")
        index_name = st.text_input("Vector Index Name", value="vector_index")
    
    with st.expander("Field Mappings", expanded=True):
        st.caption("Map your MongoDB document schema here.")
        vector_field = st.text_input("Vector Field Path", value="embedding")
        text_field = st.text_input("Text Content Field", value="text")
        metadata_fields_str = st.text_input("Additional Fields to Retrieve", value="title, source")

    # --- Connection Test Logic ---
    if st.button("Test & Initialize Connections", type="primary"):
        st.session_state.is_connected = False
        
        if not (aws_access_key and aws_secret_key and mongo_conn_str):
            st.error("❌ Missing required credentials.")
        else:
            errors = []
            
            # Test AWS
            try:
                boto_session = boto3.Session(
                    aws_access_key_id=aws_access_key,
                    aws_secret_access_key=aws_secret_key,
                    aws_session_token=aws_session_token if aws_session_token else None,
                    region_name=aws_region
                )
                bedrock = boto_session.client("bedrock")
                bedrock.list_foundation_models(byProvider="amazon")
                st.success("✅ AWS Bedrock: Connected")
            except (ClientError, NoCredentialsError) as e:
                errors.append(f"AWS Error: {e}")
                st.error("❌ AWS Bedrock: Authentication Failed")

            # Test MongoDB
            try:
                mongo_client = MongoClient(mongo_conn_str, serverSelectionTimeoutMS=5000)
                mongo_client.admin.command('ping')
                st.success("✅ MongoDB: Connected")
            except Exception as e:
                errors.append(f"Mongo Error: {e}")
                st.error("❌ MongoDB: Connection Failed")

            # Finalize
            if not errors:
                st.session_state.is_connected = True
                st.session_state.aws_creds = {
                    "access": aws_access_key,
                    "secret": aws_secret_key,
                    "token": aws_session_token,
                    "region": aws_region
                }
                meta_fields_list = [x.strip() for x in metadata_fields_str.split(",") if x.strip()]
                st.session_state.mongo_config = {
                    "conn": mongo_conn_str,
                    "db": db_name,
                    "coll": collection_name,
                    "idx": index_name,
                    "vector_key": vector_field,
                    "text_key": text_field,
                    "meta_fields": meta_fields_list
                }
                st.rerun()

# --- Stop if not connected ---
if not st.session_state.is_connected:
    st.info("👈 Please enter your credentials and field mappings in the sidebar to start.")
    st.stop()

# --- Backend Setup (Cached) ---

@st.cache_resource
def setup_resources(aws_creds, mongo_config):
    session = boto3.Session(
        aws_access_key_id=aws_creds["access"],
        aws_secret_access_key=aws_creds["secret"],
        aws_session_token=aws_creds["token"] if aws_creds["token"] else None,
        region_name=aws_creds["region"]
    )
    bedrock_runtime = session.client("bedrock-runtime")

    embeddings = BedrockEmbeddings(
        client=bedrock_runtime,
        model_id="amazon.titan-embed-text-v2:0"
    )

    client = MongoClient(mongo_config["conn"])
    collection = client[mongo_config["db"]][mongo_config["coll"]]
    
    vectorstore = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embeddings,
        index_name=mongo_config["idx"],
        relevance_score_fn="cosine",
        text_key=mongo_config["text_key"],
        embedding_key=mongo_config["vector_key"]
    )

    return session, vectorstore

@st.cache_resource
def create_agent(_session, _vectorstore, meta_fields):
    bedrock_runtime = _session.client("bedrock-runtime")

    llm = ChatBedrock(
        client=bedrock_runtime,
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        model_kwargs={"temperature": 0.0}
    )

    @tool
    def perform_vector_search(user_query: str) -> str:
        """Performs a vector search. Returns results or 'NO_RESULTS_FOUND'."""
        try:
            results = _vectorstore.similarity_search(user_query, k=3)
            
            if not results:
                return "NO_RESULTS_FOUND"
            
            formatted_results = []
            for doc in results:
                doc_str = f"Content: {doc.page_content}"
                for field in meta_fields:
                    if field in doc.metadata:
                        doc_str += f"\n{field.capitalize()}: {doc.metadata[field]}"
                formatted_results.append(doc_str)

            return "\n\n---\n\n".join(formatted_results)
        except Exception as e:
            return f"Error during search: {str(e)}"

    tools = [perform_vector_search]

    # --- UPDATED PROMPT ---
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant.
        
        Instructions:
        1. You MUST use the 'perform_vector_search' tool for every user query to check the knowledge base.
        2. If the tool returns "NO_RESULTS_FOUND", you must respond with exactly: "No results found." and nothing else.
        3. If the tool returns content, your response must have two distinct parts:
           
           **Part 1: The Answer**
           Answer the user's question clearly using the provided context.

           **Part 2: Why this is a good match**
           Explain briefly why the retrieved documents were selected and relevant. Mention specific keywords, concepts, or themes from the context that matched the user's intent.
        """),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- Main Application Logic ---

try:
    mongo_conf = st.session_state.mongo_config
    session, vector_db = setup_resources(st.session_state.aws_creds, mongo_conf)
    agent_executor = create_agent(session, vector_db, mongo_conf["meta_fields"])

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about your data..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                try:
                    response = agent_executor.invoke({"input": prompt})
                    raw_output = response["output"]
                    
                    final_text = clean_agent_output(raw_output)
                    
                    st.markdown(final_text)
                    st.session_state.messages.append({"role": "assistant", "content": final_text})
                except Exception as e:
                    st.error(f"An error occurred: {e}")

except Exception as e:
    st.error(f"Application Error: {e}")
    if st.button("Reset Configuration"):
        st.session_state.is_connected = False
        st.rerun()
