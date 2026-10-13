
import streamlit as st
import ollama
import chromadb
from pypdf import PdfReader

st.set_page_config(
    page_title="AI PDF Chatbot",
    page_icon="📄",
    layout="centered"
)

st.title("📄 AI PDF Question-Answering Chatbot")
st.write("Upload a PDF and ask questions about its content.")

# 1. Initialize session state
if "chroma_client" not in st.session_state:
    st.session_state.chroma_client = chromadb.Client()

if "collection" not in st.session_state:
    st.session_state.collection = None

if "pdf_ready" not in st.session_state:
    st.session_state.pdf_ready = False

if "processed_file" not in st.session_state:
    st.session_state.processed_file = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 2. Upload PDF
uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)

if uploaded_file is not None:
    file_id = f"{uploaded_file.name}_{uploaded_file.size}"

    if st.session_state.processed_file != file_id:
        st.session_state.pdf_ready = False
        st.session_state.collection = None

    if st.button("Process PDF"):
        try:
            with st.spinner("Reading and processing your PDF..."):

                reader = PdfReader(uploaded_file)
                chunks = []

                # 3. Extract text and create overlapping chunks
                for page_number, page in enumerate(
                    reader.pages, start=1
                ):
                    page_text = page.extract_text() or ""
                    words = page_text.split()

                    chunk_size = 180
                    overlap = 30
                    step = chunk_size - overlap

                    for i in range(0, len(words), step):
                        chunk_text = " ".join(
                            words[i:i + chunk_size]
                        )

                        if chunk_text.strip():
                            chunks.append({
                                "text": chunk_text,
                                "page": page_number
                            })

                if not chunks:
                    st.error(
                        "No readable text found. "
                        "This PDF may contain scanned images."
                    )
                    st.stop()

                # 4. Create a ChromaDB collection
                client = st.session_state.chroma_client
                collection_name = (
                    "pdf_" + str(abs(hash(file_id)))
                )

                try:
                    client.delete_collection(
                        name=collection_name
                    )
                except Exception:
                    pass

                collection = client.create_collection(
                    name=collection_name
                )

                progress = st.progress(0)

                # 5. Generate and store embeddings
                for i, chunk in enumerate(chunks):
                    embedding_response = ollama.embeddings(
                        model="nomic-embed-text",
                        prompt=chunk["text"]
                    )

                    embedding = embedding_response["embedding"]

                    collection.add(
                        ids=[str(i)],
                        documents=[chunk["text"]],
                        embeddings=[embedding],
                        metadatas=[{
                            "page": chunk["page"]
                        }]
                    )

                    progress.progress(
                        (i + 1) / len(chunks)
                    )

                st.session_state.collection = collection
                st.session_state.pdf_ready = True
                st.session_state.processed_file = file_id

                # Clear history when processing a PDF
                st.session_state.chat_history = []

                st.success(
                    f"PDF processed successfully! "
                    f"Created {len(chunks)} text chunks."
                )

        except Exception as e:
            st.error(f"Error processing PDF: {e}")

# 6. Ask questions about the PDF
if st.session_state.pdf_ready:
    st.divider()
    st.subheader("💬 Ask Questions About Your PDF")

    # Display previous conversations
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["question"])

        with st.chat_message("assistant"):
            st.write(chat["answer"])

    question = st.text_input(
        "Enter your question:",
        key="question_input"
    )

    if st.button("Get Answer"):
        if not question.strip():
            st.warning("Please enter a question.")

        else:
            try:
                with st.spinner(
                    "Searching the PDF and generating an answer..."
                ):
                    collection = st.session_state.collection

                    if collection is None:
                        st.error(
                            "Please upload and process the PDF again."
                        )
                        st.stop()

                    # 7. Convert the question into an embedding
                    query_response = ollama.embeddings(
                        model="nomic-embed-text",
                        prompt=question
                    )

                    query_embedding = query_response["embedding"]

                    # 8. Retrieve relevant PDF chunks
                    result_count = min(
                        3, collection.count()
                    )

                    results = collection.query(
                        query_embeddings=[query_embedding],
                        n_results=result_count
                    )

                    passages = results["documents"][0]
                    metadatas = results["metadatas"][0]

                    # 9. Build the PDF context
                    context_parts = []

                    for passage, metadata in zip(
                        passages, metadatas
                    ):
                        page_number = metadata["page"]

                        context_parts.append(
                            f"Page {page_number}:\n{passage}"
                        )

                    context = "\n\n".join(context_parts)

                    # 10. Prepare the conversation for Ollama
                    messages = [
                        {
                            "role": "system",
                            "content": (
                                "You answer questions about PDF documents. "
                                "Use the provided PDF context for factual "
                                "answers. Use previous conversation messages "
                                "to understand follow-up questions. "
                                "Do not invent facts. If the PDF does not "
                                "provide enough information, say so."
                            )
                        }
                    ]

                    # Add previous questions and answers
                    for chat in st.session_state.chat_history:
                        messages.append({
                            "role": "user",
                            "content": chat["question"]
                        })

                        messages.append({
                            "role": "assistant",
                            "content": chat["answer"]
                        })

                    # Add current question and retrieved context
                    messages.append({
                        "role": "user",
                        "content": (
                            f"PDF context:\n{context}\n\n"
                            f"Question: {question}"
                        )
                    })

                    # 11. Generate the answer
                    response = ollama.chat(
                        model="llama3.2:3b",
                        messages=messages
                    )

                    answer = response["message"]["content"]

                    # 12. Save this conversation
                    st.session_state.chat_history.append({
                        "question": question,
                        "answer": answer
                    })

                    # 13. Display the answer
                    st.subheader("✅ Answer")
                    st.write(answer)

                    # 14. Display source page numbers
                    source_pages = sorted({
                        metadata["page"]
                        for metadata in metadatas
                    })

                    st.subheader("📚 Sources")
                    st.write(
                        "Retrieved PDF page(s): "
                        + ", ".join(
                            str(page) for page in source_pages
                        )
                    )

                    # 15. Display retrieved passages
                    st.subheader("📄 Retrieved PDF Passages")

                    for passage, metadata in zip(
                        passages, metadatas
                    ):
                        with st.expander(
                            f"View source — Page {metadata['page']}"
                        ):
                            st.write(passage)

            except Exception as e:
                st.error(f"Error generating answer: {e}")

st.divider()
st.caption(
    "Built with Python, Streamlit, ChromaDB, PyPDF and Ollama."
)
