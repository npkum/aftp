import streamlit as st
from datetime import datetime
import pandas as pd
import numpy as np
import faiss
from typing import Dict, Any, Optional
from dataclasses import dataclass

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langgraph.graph import StateGraph, END

# === Setup ===
st.set_page_config(page_title="Hardship Planner", layout="wide")
st.title("💡 Hardship Plan Recommender")

# === Load LLM ===
@st.cache_resource
def load_llm_streaming():
    model_id = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256)

llm_pipe = load_llm_streaming()

# === Embedding + Vector DB ===
encoder = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')
index = faiss.IndexFlatL2(384)
vector_store: Dict[str, Dict[str, Any]] = {}

# === Sample Staff Data ===
case_worker_df = pd.DataFrame([
    {"name": "Priya Singh", "plan": "3-month deferral", "experience_years": 5, "on_leave": False},
    {"name": "Anil Mehta", "plan": "interest reduction", "experience_years": 7, "on_leave": False},
    {"name": "Sara Khan", "plan": "partial payment suspension", "experience_years": 4, "on_leave": True},
    {"name": "David Rao", "plan": "payment extension", "experience_years": 6, "on_leave": False}
])

# === Plans ===
hardship_plans = {
    "3-month deferral": {"description": "Suspend payments for 3 months.", "risk_score": 2},
    "partial payment suspension": {"description": "Pay only 50% for 3 months.", "risk_score": 3},
    "interest reduction": {"description": "Reduce interest by 25% for 6 months.", "risk_score": 1},
    "payment extension": {"description": "Extend term by 12 months.", "risk_score": 4},
}

def check_eligibility(app: Dict) -> list:
    if app["hardship_reason"] == "Job Loss" and app["income"] < app["expenses"]:
        return ["3-month deferral", "partial payment suspension"]
    return ["payment extension", "interest reduction"]

def build_embedding(app: Dict) -> str:
    return f"{app['hardship_reason']} income:{app['income']} expenses:{app['expenses']}"

def store_context(customer_id: str, embedding_text: str, context: Dict[str, Any]):
    vector = encoder.encode([embedding_text])[0]
    index.add(np.array([vector]))
    vector_store[customer_id] = {"vector": vector, "context": context}

def assign_case_worker(plan: str):
    workers = case_worker_df[(case_worker_df.plan == plan) & (~case_worker_df.on_leave)]
    if not workers.empty:
        return workers.sample(1).iloc[0]["name"]
    return "Queue for manual assignment"

def assign_to_human_caseworker(app: Dict[str, Any]) -> str:
    available = case_worker_df[(case_worker_df["experience_years"] >= 5) & (~case_worker_df["on_leave"])]
    if not available.empty:
        return available.sample(1).iloc[0]["name"]
    return "No senior case workers available."

@dataclass
class ApplicationState:
    application: Dict[str, Any]
    selected_plan: Optional[str] = None
    escalated: bool = False
    handled: bool = False

def validate(state: ApplicationState) -> ApplicationState:
    app = state.application
    for field in ["income", "expenses", "hardship_reason"]:
        if field not in app:
            raise ValueError(f"Missing field: {field}")
    return state

def reason_and_rank(state: ApplicationState) -> ApplicationState:
    app = state.application
    eligible = check_eligibility(app)

    query_vector = encoder.encode([build_embedding(app)])[0]
    similarity_threshold = 10.0
    best_context = None

    if len(vector_store) > 0:
        D, I = index.search(np.array([query_vector]), k=1)
        distance = D[0][0]
        if distance <= similarity_threshold:
            best_match_id = list(vector_store.keys())[I[0][0]]
            best_context = vector_store[best_match_id]["context"]

    ranked = sorted(eligible, key=lambda p: hardship_plans[p]["risk_score"])

    if app.get("feedback_hint") == "wants_alternative" and len(ranked) > 1:
        best_plan = ranked[1]
        del app["feedback_hint"]
    else:
        best_plan = ranked[0]

    return ApplicationState(application=app, selected_plan=best_plan, escalated=False, handled=False)

def check_proactive_escalation(app: Dict[str, Any]) -> bool:
    return app["history"]["late_payments"] > 3 or app.get("flagged_abuse", False)

def take_action(state: ApplicationState) -> ApplicationState:
    app = state.application
    plan = state.selected_plan

    if check_proactive_escalation(app):
        assign_to_human_caseworker(app)
        return ApplicationState(application=app, selected_plan=plan, escalated=True, handled=True)

    if plan:
        assign_case_worker(plan)
        store_context(app["customer_id"], build_embedding(app), {"plan": plan, "status": "active"})

    return ApplicationState(application=app, selected_plan=plan, escalated=False, handled=True)

def decide_next_node(state: ApplicationState) -> str:
    app = state.application
    if state.handled:
        return END

    prompt = f"""
Customer Info:
- Income: {app['income']}
- Expenses: {app['expenses']}
- Hardship: {app['hardship_reason']}
- Selected Plan: {state.selected_plan}
- Handled: {state.handled}
- Escalated: {state.escalated}

Options: validate, reason_and_rank, take_action, END
"""

    output = llm_pipe(prompt)[0]["generated_text"].strip().lower()
    if "validate" in output:
        return "validate"
    if "reason" in output:
        return "reason_and_rank"
    if "action" in output:
        return "take_action"
    return "reason_and_rank"

workflow = StateGraph(state_schema=ApplicationState)
workflow.add_node("validate", validate)
workflow.add_node("reason_and_rank", reason_and_rank)
workflow.add_node("take_action", take_action)
workflow.set_entry_point("validate")
workflow.add_conditional_edges("validate", decide_next_node)
workflow.add_conditional_edges("reason_and_rank", decide_next_node)
workflow.add_conditional_edges("take_action", decide_next_node)
compiled_graph = workflow.compile()

def prepopulate_samples():
    samples = [
        {"customer_id": "C100", "text": "Job Loss income:1500 expenses:2200", "context": {"plan": "3-month deferral"}},
        {"customer_id": "C101", "text": "Disaster income:1800 expenses:2100", "context": {"plan": "interest reduction"}},
        {"customer_id": "C102", "text": "Medical Emergency income:2000 expenses:2500", "context": {"plan": "partial payment suspension"}},
    ]
    for s in samples:
        store_context(s["customer_id"], s["text"], s["context"])

prepopulate_samples()

# === Main UI ===
if "final_state" not in st.session_state:
    with st.form("user_input_form"):
        name = st.text_input("Full Name")
        customer_id = st.text_input("Customer ID")
        hardship_reason = st.selectbox("Hardship Reason", ["Job Loss", "Medical Emergency", "Disaster"])
        income = st.number_input("Monthly Income", min_value=0.0)
        expenses = st.number_input("Monthly Expenses", min_value=0.0)
        account_type = st.selectbox("Account Type", ["mortgage", "credit card", "personal loan"])
        late_payments = st.slider("Late Payments (last 12 months)", 0, 12, 0)
        flagged_abuse = st.checkbox("Previously defaulted or rejected plan?")
        submitted = st.form_submit_button("Submit")

    if submitted:
        app = {
            "customer_id": customer_id,
            "name": name,
            "submitted_on": str(datetime.today().date()),
            "hardship_reason": hardship_reason,
            "income": income,
            "expenses": expenses,
            "account_type": account_type,
            "history": {"on_time_payments": 12 - late_payments, "late_payments": late_payments},
            "flagged_abuse": flagged_abuse
        }
        initial_state = ApplicationState(application=app)
        final_state = compiled_graph.invoke(initial_state)
        st.session_state.final_state = final_state
        st.rerun()

# === Feedback Handling ===
elif "final_state" in st.session_state:
    final_state = st.session_state.final_state

    if final_state.selected_plan and not final_state.escalated:
        st.subheader("✅ Recommended Plan")
        st.write(f"**Plan:** {final_state.selected_plan}")
        st.write(f"**Description:** {hardship_plans[final_state.selected_plan]['description']}")

        if st.session_state.get("plan_accepted", False):
            st.success("✅ You have accepted the plan.")
            st.info(f"Reference #: {final_state.application['customer_id'].upper()}-{datetime.now().strftime('%d%m%y%H%M%S')}")
            st.markdown("---")
            st.markdown("🔁 Would you like to submit another hardship application?")
            if st.button("Submit another hardship application"):
                for k in list(st.session_state.keys()):
                    del st.session_state[k]
                st.rerun()
        else:
            st.markdown("#### 💬 What would you like to do next?")
            with st.form("feedback_form"):
                feedback_choice = st.radio(
                    "Choose an option:",
                    ["Accept this plan", "Suggest an alternative", "I can’t afford this"],
                    key="feedback_radio"
                )
                feedback_submit = st.form_submit_button("Submit")

            if feedback_submit:
                current_state = st.session_state.final_state
                if feedback_choice == "Accept this plan":
                    st.session_state.plan_accepted = True
                    st.rerun()
                elif feedback_choice == "Suggest an alternative":
                    current_state.application["feedback_hint"] = "wants_alternative"
                    updated = reason_and_rank(current_state)
                    updated = take_action(updated)
                    st.session_state.final_state = updated
                    st.rerun()
                elif feedback_choice == "I can’t afford this":
                    current_state.application["income"] /= 2
                    updated = reason_and_rank(current_state)
                    updated = take_action(updated)
                    st.session_state.final_state = updated
                    st.rerun()
    else:
        st.warning("Unable to find suitable plan. Try again or contact support.")
