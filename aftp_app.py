# === aftp_app.py ===

import streamlit as st
from datetime import datetime
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, END
from dataclasses import dataclass
import faiss
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from typing import Dict, Any, Optional

# === Page config ===
st.set_page_config(page_title="AFTP - Hardship Planner", layout="centered")

# === Load LLM ===
@st.cache_resource
def load_llm_streaming():
    model_id = "google/flan-t5-base"  # ✅ Using smaller model for faster response
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256)

llm_pipe = load_llm_streaming()

# === Embedding + Vector Store ===
encoder = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')
index = faiss.IndexFlatL2(384)
vector_store: Dict[str, Dict[str, Any]] = {}

# === Case workers and plans ===
case_worker_df = pd.DataFrame([
    {"name": "Priya Singh", "plan": "3-month deferral", "experience_years": 5, "on_leave": False},
    {"name": "Anil Mehta", "plan": "interest reduction", "experience_years": 7, "on_leave": False},
    {"name": "Sara Khan", "plan": "partial payment suspension", "experience_years": 4, "on_leave": True},
    {"name": "David Rao", "plan": "payment extension", "experience_years": 6, "on_leave": False}
])

hardship_plans = {
    "3-month deferral": {"description": "Suspend payments for 3 months.", "risk_score": 2},
    "partial payment suspension": {"description": "Pay only 50% for 3 months.", "risk_score": 3},
    "interest reduction": {"description": "Reduce interest by 25% for 6 months.", "risk_score": 1},
    "payment extension": {"description": "Extend term by 12 months.", "risk_score": 4},
}

# === Utility Functions ===
def check_eligibility(app):
    if app["income"] < app["expenses"]:
        return ["3-month deferral", "partial payment suspension"]
    return ["payment extension", "interest reduction"]

def build_embedding(app):
    return f"{app['hardship_reason']} income:{app['income']} expenses:{app['expenses']}"

def store_context(customer_id, embedding_text, context):
    vector = encoder.encode([embedding_text])[0]
    index.add(np.array([vector]))
    vector_store[customer_id] = {"vector": vector, "context": context}

def assign_case_worker(plan):
    workers = case_worker_df[(case_worker_df.plan == plan) & (~case_worker_df.on_leave)]
    if not workers.empty:
        selected = workers.sample(1).iloc[0]
        st.success(f"Assigned to {selected['name']} ({selected['experience_years']} yrs experience)")
    else:
        st.warning(f"No available case worker for plan: {plan}")

def assign_to_human_caseworker(app):
    available = case_worker_df[(case_worker_df["experience_years"] >= 5) & (~case_worker_df["on_leave"])]
    if not available.empty:
        selected = available.sample(1).iloc[0]
        st.session_state.reviewer_name = selected["name"]
    else:
        st.session_state.reviewer_name = "a senior case worker"

def check_proactive_escalation(app):
    return app["history"]["late_payments"] > 3 or app.get("flagged_abuse", False)

@st.cache_resource
def prepopulate_samples():
    samples = [
        {"customer_id": "C100", "text": "Job Loss income:1500 expenses:2200", "context": {"plan": "3-month deferral"}},
        {"customer_id": "C101", "text": "Disaster income:1800 expenses:2100", "context": {"plan": "interest reduction"}},
        {"customer_id": "C102", "text": "Medical Emergency income:2000 expenses:2500", "context": {"plan": "partial payment suspension"}},
        {"customer_id": "C103", "text": "Job Loss income:1400 expenses:2000", "context": {"plan": "3-month deferral"}},
        {"customer_id": "C104", "text": "Disaster income:2500 expenses:2600", "context": {"plan": "payment extension"}},
    ]
    for sample in samples:
        store_context(sample["customer_id"], sample["text"], sample["context"])

prepopulate_samples()

# === LangGraph State ===
@dataclass
class ApplicationState:
    application: Dict[str, Any]
    selected_plan: Optional[str] = None
    escalated: bool = False
    handled: bool = False

def validate(state: ApplicationState) -> ApplicationState:
    return state

def reason_and_rank(state: ApplicationState) -> ApplicationState:
    app = state.application
    eligible = check_eligibility(app)

    if not eligible:
        assign_to_human_caseworker(app)
        return ApplicationState(application=app, escalated=True, handled=True)

    query_vector = encoder.encode([build_embedding(app)])[0]
    best_context = None
    if len(vector_store) > 0:
        D, I = index.search(np.array([query_vector]), k=1)
        if D[0][0] <= 10.0:
            best_match_id = list(vector_store.keys())[I[0][0]]
            best_context = vector_store[best_match_id]["context"]
            st.info(f"Matched similar case: {best_match_id} | Plan: {best_context['plan']}")

    ranked = sorted(eligible, key=lambda p: hardship_plans[p]["risk_score"])

    if app.get("feedback_hint") == "wants_alternative" and len(ranked) > 1:
        best_plan = ranked[1]
    else:
        best_plan = ranked[0]

    return ApplicationState(application=app, selected_plan=best_plan)

def take_action(state: ApplicationState) -> ApplicationState:
    app = state.application
    plan = state.selected_plan

    if check_proactive_escalation(app):
        assign_to_human_caseworker(app)
        return ApplicationState(application=app, selected_plan=plan, escalated=True, handled=True)

    if plan:
        st.success(f"Plan assigned: {plan}")
        assign_case_worker(plan)
        store_context(app["customer_id"], build_embedding(app), {"plan": plan, "status": "active"})
        return ApplicationState(application=app, selected_plan=plan, handled=True)

    return ApplicationState(application=app, escalated=True, handled=True)

def decide_next_node(state: ApplicationState) -> str:
    if state.handled:
        return END

    app = state.application
    prompt = f"""
Customer Info:
- Reason: {app['hardship_reason']}
- Income: {app['income']}
- Expenses: {app['expenses']}
- Late Payments: {app['history']['late_payments']}
- Flagged Abuse: {app.get('flagged_abuse', False)}
- Selected Plan: {state.selected_plan}
- Escalated: {state.escalated}
- Handled: {state.handled}

What is the next step?
Options: validate, reason_and_rank, take_action, END
"""
    llm_output = llm_pipe(prompt)[0]["generated_text"].strip().lower()
    if "validate" in llm_output:
        return "validate"
    elif "reason" in llm_output:
        return "reason_and_rank"
    elif "action" in llm_output and state.selected_plan:
        return "take_action"
    elif "end" in llm_output:
        return END
    return "reason_and_rank"

# === LangGraph Compilation ===
workflow = StateGraph(state_schema=ApplicationState)
workflow.add_node("validate", validate)
workflow.add_node("reason_and_rank", reason_and_rank)
workflow.add_node("take_action", take_action)
workflow.set_entry_point("validate")
workflow.add_conditional_edges("validate", decide_next_node)
workflow.add_conditional_edges("reason_and_rank", decide_next_node)
workflow.add_conditional_edges("take_action", decide_next_node)
compiled_graph = workflow.compile()

# === Streamlit UI ===
st.title("🤖 AFTP Hardship Planner")
if "final_state" not in st.session_state:
    st.markdown("Submit your hardship information to receive a suitable plan recommendation.")

# === Form Input Section ===
# === Streamlit UI ===
st.title("🤖 AFTP Hardship Planner")

# Only show top-level instructions if not processing or done
if "final_state" not in st.session_state and not st.session_state.get("processing"):
    st.markdown("Submit your hardship information to receive a suitable plan recommendation.")

# === Step 1: Show input form only if not already processing or done ===
if "final_state" not in st.session_state and not st.session_state.get("processing"):
    with st.form("input_form"):
        name = st.text_input("Full Name")
        customer_id = st.text_input("Customer ID")
        hardship_reason = st.selectbox("Hardship Reason", ["Job Loss", "Medical Emergency", "Disaster"])
        income = st.number_input("Monthly Income", min_value=0.0)
        expenses = st.number_input("Monthly Expenses", min_value=0.0)
        account_type = st.selectbox("Account Type", ["mortgage", "credit card", "personal loan"])
        late_payments = st.slider("Late Payments in Last 12 Months", 0, 12, 0)
        flagged_abuse = st.checkbox("Previously defaulted or rejected?")
        submitted = st.form_submit_button("Submit Application")

    if submitted:
        st.session_state.processing = True
        st.session_state.application_data = {
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
        st.rerun()

# === Step 2: If processing, run the graph ===
if st.session_state.get("processing") and "final_state" not in st.session_state:
    with st.spinner("⚙️ AFTP is processing your request..."):
        initial_state = ApplicationState(application=st.session_state.application_data)
        final_state_dict = compiled_graph.invoke(initial_state)
        st.session_state.final_state = ApplicationState(**final_state_dict)
        st.session_state.processing = False
        st.rerun()

# === Results + Feedback ===
if "final_state" in st.session_state:
    final_state = st.session_state.final_state

    if final_state.selected_plan and not final_state.escalated:
        st.subheader("✅ Recommended Plan")
        st.write(f"**Plan:** {final_state.selected_plan}")
        st.write(f"**Description:** {hardship_plans[final_state.selected_plan]['description']}")

        if st.session_state.get("plan_accepted", False):
            st.success("✅ You have accepted the plan.")
            st.info(f"Reference #: {final_state.application['customer_id'].upper()}-{datetime.now().strftime('%d%m%y%H%M%S')}")
            if st.button("🔁 Submit a new hardship application"):
                st.session_state.clear()
                st.rerun()
        else:
            with st.form("feedback_form"):
                feedback_option = st.radio("Would you like to:", ["Accept this plan", "Suggest an alternative", "I can't afford this"])
                feedback_submit = st.form_submit_button("Submit Feedback")

            if feedback_submit:
                if feedback_option == "Accept this plan":
                    st.session_state.plan_accepted = True
                    st.rerun()
                else:
                    if "alternative" in feedback_option.lower():
                        final_state.application["feedback_hint"] = "wants_alternative"
                    elif "afford" in feedback_option.lower():
                        final_state.application["income"] /= 2
                    updated = reason_and_rank(final_state)
                    updated = take_action(updated)
                    st.session_state.final_state = updated
                    st.rerun()

    elif final_state.escalated:
        reviewer = st.session_state.get("reviewer_name", "a senior case worker")
        st.subheader(f"📤 Escalated to human reviewer: **{reviewer}**")
        st.warning("Your case was escalated for human review.")
        if st.button("🔁 Submit a new hardship application"):
            st.session_state.clear()
            st.rerun()
    else:
        st.error("Unable to find a suitable plan.")
        if st.button("🔁 Submit a new hardship application"):
            st.session_state.clear()
            st.rerun()
