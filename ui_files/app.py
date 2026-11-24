# ============================================================
# FreightRates RAG Assistant - Streamlit chat frontend
# ============================================================
import io
import os

import pandas as pd
import requests
import streamlit as st

# Backend address: container name on the shared Docker network (overridable via env var)
RAG_URL = os.getenv("RAG_URL", "http://freight-rag:7870")

EXAMPLES = [
    "What is the latest 40' rate from Shanghai to Los Angeles?",
    "Which carrier offers the cheapest 40' quote from Nhava Sheva to New York?",
    "How has the 20' rate from Qingdao to Long Beach changed over the last 13 weeks?",
    "Which US coast is cheapest for importing from China?",
    "Is Tughlakabad a seaport or an inland depot?",
]

st.set_page_config(page_title="Freight RAG Assistant", page_icon="🚢", layout="centered")
st.title("🚢 FreightRates Assistant")
st.caption("Ask about ports, rate history and carrier quotes between the US and India / China. "
           "Answers come only from the FreightRates data (synthetic test data).")

with st.sidebar:
    st.header("Settings")
    k = st.slider("Documents to retrieve (k)", 1, 8, 4)
    show_sources = st.toggle("Show sources", value=True)
    try:
        info = requests.get(f"{RAG_URL}/", timeout=5).json()
        st.success(f"API online · {info['documents']} documents · data through {info.get('data_through')}")
        st.caption(f"LLM: {info['llm'].split('/')[-1]}")
    except Exception:
        st.error(f"API not reachable at {RAG_URL}")
    st.header("Try")
    for ex in EXAMPLES:
        if st.button(ex, use_container_width=True):
            st.session_state["pending"] = ex

tab_chat, tab_batch = st.tabs(["Chat", "Batch questions"])

with tab_chat:
    if "history" not in st.session_state:
        st.session_state["history"] = []
    for turn in st.session_state["history"]:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn.get("sources") and show_sources:
                with st.expander("Sources"):
                    for s in turn["sources"]:
                        st.write(f"• {s['title']}  (score {s['score']})")

    question = st.chat_input("Ask about a lane, port, rate trend or quote…") or st.session_state.pop("pending", None)
    if question:
        st.session_state["history"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching the rate data…"):
                try:
                    resp = requests.post(f"{RAG_URL}/v1/ask", json={"question": question, "k": k}, timeout=300)
                    if resp.status_code == 200:
                        out = resp.json()
                        st.markdown(out["answer"])
                        st.caption(f"{out['seconds']} s")
                        if show_sources:
                            with st.expander("Sources"):
                                for s in out["sources"]:
                                    st.write(f"• {s['title']}  (score {s['score']})")
                        st.session_state["history"].append({"role": "assistant", "content": out["answer"],
                                                            "sources": out["sources"]})
                    else:
                        st.error(f"API error {resp.status_code}: {resp.text}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Could not reach the API at {RAG_URL}: {e}")

with tab_batch:
    st.write("Upload a CSV with a **question** column (up to 50 rows).")
    uploaded = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded is not None:
        batch_df = pd.read_csv(uploaded)
        st.dataframe(batch_df.head())
        if st.button("Answer all questions", type="primary"):
            with st.spinner("Answering…"):
                try:
                    resp = requests.post(f"{RAG_URL}/v1/askbatch", files={"file": uploaded.getvalue()}, timeout=1800)
                    if resp.status_code == 200:
                        res = resp.json()
                        batch_df["answer"] = [res[str(i)].get("answer", res[str(i)].get("error")) for i in range(len(batch_df))]
                        batch_df["sources"] = ["; ".join(res[str(i)].get("sources", [])) for i in range(len(batch_df))]
                        st.dataframe(batch_df)
                        st.download_button("Download answers", batch_df.to_csv(index=False),
                                           file_name="freight_rag_answers.csv", mime="text/csv")
                    else:
                        st.error(f"API error {resp.status_code}: {resp.text}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Could not reach the API at {RAG_URL}: {e}")
