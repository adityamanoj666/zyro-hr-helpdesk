
import os
import glob
import streamlit as st

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import (
    RunnableParallel,
    RunnableLambda,
    RunnablePassthrough
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from pydantic import Field


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="Zyro Dynamics HR Help Desk",
    page_icon="💼"
)

st.title("💼 Zyro Dynamics HR Help Desk")
st.caption("Ask questions about Zyro Dynamics HR policies.")


# ============================================================
# CONFIGURATION
# ============================================================

CORPUS_DIR = "hr_corpus"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 3


# ============================================================
# LOAD PDF DOCUMENTS
# ============================================================

@st.cache_resource
def build_rag_system():

    pdf_files = sorted(
        glob.glob(os.path.join(CORPUS_DIR, "*.pdf"))
    )

    documents = []

    for pdf_path in pdf_files:

        reader = PdfReader(pdf_path)

        for page_number, page in enumerate(reader.pages):

            text = page.extract_text() or ""

            if text.strip():

                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": pdf_path,
                            "page": page_number + 1
                        }
                    )
                )


    # ========================================================
    # CHUNKING
    # ========================================================

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)


    # ========================================================
    # EMBEDDINGS
    # ========================================================

    embed_model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    embeddings = embed_model.encode(
        [chunk.page_content for chunk in chunks],
        show_progress_bar=False
    ).astype("float32")


    # ========================================================
    # FAISS
    # ========================================================

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)


    # ========================================================
    # RETRIEVER
    # ========================================================

    class FAISSRetriever(BaseRetriever):

        index: object = Field(exclude=True)
        chunks: list = Field(exclude=True)
        embed_model: object = Field(exclude=True)
        k: int = 3

        def _get_relevant_documents(
            self,
            query,
            *,
            run_manager=None
        ):

            query_embedding = self.embed_model.encode(
                [query]
            ).astype("float32")

            distances, indices = self.index.search(
                query_embedding,
                self.k
            )

            return [
                self.chunks[idx]
                for idx in indices[0]
            ]


    retriever = FAISSRetriever(
        index=index,
        chunks=chunks,
        embed_model=embed_model,
        k=TOP_K
    )


    # ========================================================
    # GROQ LLM
    # ========================================================

    groq_api_key = st.secrets["GROQ_API_KEY"]

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=groq_api_key,
        temperature=0
    )


    # ========================================================
    # SCOPE CLASSIFIER
    # ========================================================

    scope_prompt = ChatPromptTemplate.from_template("""
You are a scope classifier for the Zyro Dynamics Pvt. Ltd. HR Help Desk.

Supported:
- Leave
- Compensation/payroll
- Benefits/insurance
- Performance
- Work from home
- Code of conduct
- IT/data security
- POSH
- Onboarding/separation
- Travel/expense

OUT OF SCOPE:
- Job applications/recruitment
- ESOP vesting
- Company revenue/financial performance
- Comparisons with external products/companies
- Other-company policies
- Unrelated topics

User question:
{question}

Return exactly:
IN_SCOPE
or
OUT_OF_SCOPE
""")


    scope_chain = scope_prompt | llm


    # ========================================================
    # GROUNDING PROMPT
    # ========================================================

    prompt = ChatPromptTemplate.from_template("""
You are an HR Help Desk assistant for Zyro Dynamics Pvt. Ltd.

Answer the user's question using ONLY the information provided in the context.

Rules:

1. Do not use outside knowledge.

2. Do not invent facts, deadlines, eligibility rules,
   exceptions, or policy details.

3. Answer only what the user asked.
   Do not add unrelated information from the context.

4. If the policy provides a table or list containing
   the relevant information, provide the relevant entries
   even if the user's personal situation cannot be determined.

5. If the user's personal information is required to determine
   their specific answer, explain what information is missing
   after stating the applicable policy rules.

6. If the context does not contain enough information to answer
   the question, clearly say that the information is not available
   in the provided HR policies.

7. Be concise and directly answer the question.

Context:
{context}

Question:
{question}

Answer:
""")


    # ========================================================
    # FORMAT RETRIEVED DOCUMENTS
    # ========================================================

    def format_docs(docs):

        return "\n\n---\n\n".join(
            doc.page_content
            for doc in docs
        )


    # ========================================================
    # LCEL RAG CHAIN
    # ========================================================

    rag_chain = (
        RunnableParallel(
            docs=retriever,
            question=RunnablePassthrough()
        )

        | RunnableLambda(
            lambda x: {
                "answer": llm.invoke(
                    prompt.invoke({
                        "context": format_docs(x["docs"]),
                        "question": x["question"]
                    })
                ),
                "docs": x["docs"]
            }
        )
    )


    # ========================================================
    # FINAL ASK_BOT FUNCTION
    # ========================================================

    def ask_bot(question):

        # Scope check
        scope_result = scope_chain.invoke(
            {"question": question}
        )

        classification = (
            scope_result.content.strip()
        )


        if classification == "OUT_OF_SCOPE":

            return {
                "answer":
                    "I'm sorry, but this question is outside "
                    "the scope of the Zyro Dynamics HR Help Desk.",
                "sources": []
            }


        # RAG
        response = rag_chain.invoke(question)

        answer = response["answer"].content
        docs = response["docs"]


        # Sources
        sources = []

        for doc in docs:

            file_name = os.path.basename(
                doc.metadata["source"]
            )

            page = doc.metadata.get(
                "page",
                "Unknown"
            )

            source = {
                "file": file_name,
                "page": page
            }

            if source not in sources:
                sources.append(source)


        return {
            "answer": answer,
            "sources": sources
        }


    return ask_bot, len(pdf_files), len(chunks)


# ============================================================
# INITIALIZE SYSTEM
# ============================================================

try:

    ask_bot, document_count, chunk_count = build_rag_system()

    st.caption(
        f"📚 {document_count} policy documents • "
        f"🔹 {chunk_count} indexed chunks"
    )

except Exception as e:

    st.error(
        "Unable to initialize the HR Help Desk."
    )

    st.exception(e)

    st.stop()


# ============================================================
# CHAT INTERFACE
# ============================================================

question = st.chat_input(
    "Ask an HR policy question..."
)


if question:

    # User message
    with st.chat_message("user"):
        st.write(question)


    # Assistant response
    with st.chat_message("assistant"):

        with st.spinner("Searching HR policies..."):

            result = ask_bot(question)


        st.write(result["answer"])


        # Sources
        if result["sources"]:

            with st.expander("📚 Sources"):

                for source in result["sources"]:

                    st.write(
                        f"- {source['file']} — "
                        f"Page {source['page']}"
                    )
