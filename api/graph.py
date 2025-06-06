from langgraph.graph import StateGraph
from langchain.schema import BaseMessage
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable
from sentence_transformers import SentenceTransformer
import faiss

class MatchState(dict): pass

def job_input_step(state: MatchState) -> MatchState:
    state["job_desc"] = state["input"]
    return state

def resume_search_step(state: MatchState) -> MatchState:
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    resumes = [
        "Data scientist with expertise in Python, ML, and AI.",
        "Software engineer with experience in cloud computing and backend.",
        "Machine learning engineer specializing in NLP and computer vision."
    ]
    embeddings = embedder.encode(resumes)
    q_emb = embedder.encode([state["job_desc"]])
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    distances, indices = index.search(q_emb, 3)
    matches = [(resumes[i], float(distances[0][j])) for j, i in enumerate(indices[0])]
    state["matches"] = matches
    return state

def llm_summary_step(state: MatchState) -> MatchState:
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
    prompt = ChatPromptTemplate.from_template(
        "Given the job description and top 3 resume matches, summarize which candidate is best suited:\n\nJob: {job_desc}\n\nResumes: {matches}"
    )
    chain: Runnable = prompt | llm
    response = chain.invoke(state)
    state["summary"] = response.content
    return state

def build_graph():
    graph = StateGraph(MatchState)
    graph.add_node("input", job_input_step)
    graph.add_node("match", resume_search_step)
    graph.add_node("llm_summary", llm_summary_step)
    graph.set_entry_point("input")
    graph.add_edge("input", "match")
    graph.add_edge("match", "llm_summary")
    graph.set_finish_point("llm_summary")
    return graph.compile()
