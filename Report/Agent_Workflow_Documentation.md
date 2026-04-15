# Agent Workflow Documentation

## Customer Churn Intelligence System — Agentic AI Retention Strategist

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Objective of the Agent](#2-objective-of-the-agent)
3. [High-Level Workflow Explanation](#3-high-level-workflow-explanation)
4. [System Architecture](#4-system-architecture)
5. [Agent Workflow Using LangGraph](#5-agent-workflow-using-langgraph)
6. [Agent State Design](#6-agent-state-design)
7. [Detailed Node Explanations](#7-detailed-node-explanations)
   - 7.1 Risk Node
   - 7.2 Retrieval Node (RAG)
   - 7.3 Planning Node (LLM)
8. [RAG Pipeline Explanation](#8-rag-pipeline-explanation)
9. [Prompt Engineering Strategy](#9-prompt-engineering-strategy)
10. [LLM Output Structure](#10-llm-output-structure)
11. [System Robustness & Error Handling](#11-system-robustness--error-handling)
12. [End-to-End Flow](#12-end-to-end-flow)
13. [Example Input & Output](#13-example-input--output)
14. [Key Design Decisions](#14-key-design-decisions)
15. [Limitations](#15-limitations)
16. [Future Enhancements](#16-future-enhancements)
17. [Conclusion](#17-conclusion)

---

## 1. Introduction

The **Customer Churn Intelligence System** is a production-grade application that unifies machine learning prediction with agentic AI reasoning. Built on the Telco Customer Churn dataset, the system moves beyond simple binary classification to deliver **actionable, source-backed retention strategies** through an autonomous multi-node agent pipeline.

The application comprises two tightly integrated milestones:

| Milestone | Capability | Technology |
|-----------|-----------|------------|
| **Milestone 1** | Churn probability prediction | XGBoost, Scikit-learn, Streamlit |
| **Milestone 2** | Agentic retention strategy generation | LangGraph, FAISS, HuggingFace Embeddings, Groq (LLaMA 3.3 70B) |

This document provides a complete technical walkthrough of the **Milestone 2 agent workflow** — the agentic AI system that transforms a churn prediction into a structured, LLM-generated retention plan grounded in retrieved domain knowledge.

---

## 2. Objective of the Agent

The agent's primary objective is to **convert a numerical churn probability into a human-readable, actionable retention strategy** — autonomously, with no manual intervention.

Specifically, the agent must:

1. **Classify risk** — Interpret the raw churn probability into a categorical risk level (High / Medium / Low) and identify the underlying reasons for churn risk.
2. **Retrieve relevant strategies** — Use Retrieval-Augmented Generation (RAG) to fetch domain-specific retention strategies from a curated knowledge base, ensuring the output is grounded in real-world best practices rather than hallucinated content.
3. **Generate a structured plan** — Invoke a Large Language Model (LLM) to synthesize the risk assessment and retrieved strategies into a coherent, structured JSON output containing a risk summary, specific recommendations, verified sources, and a disclaimer.

**Core design principle:** The agent must never fabricate strategies. Every recommendation must trace back to a retrieved source document, enforced through explicit prompt constraints.

---

## 3. High-Level Workflow Explanation

The system executes as a **linear, three-stage pipeline** orchestrated by LangGraph:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        HIGH-LEVEL WORKFLOW                                  │
│                                                                             │
│   User Input (19 features)                                                  │
│       │                                                                     │
│       ▼                                                                     │
│   ┌──────────────────┐                                                      │
│   │  ML PREDICTION   │  XGBoost model → churn_prob (0.0 – 1.0)             │
│   │  (Milestone 1)   │  Threshold τ = 0.4 → binary label                   │
│   └────────┬─────────┘                                                      │
│            │ churn_prob, tenure, monthly_charges                             │
│            ▼                                                                │
│   ┌──────────────────────────────────────────────────────────────────┐      │
│   │                  AGENTIC AI PIPELINE (Milestone 2)               │      │
│   │                                                                  │      │
│   │   ┌────────────┐     ┌─────────────────┐     ┌──────────────┐   │      │
│   │   │ RISK NODE  │────▶│ RETRIEVAL NODE  │────▶│ PLANNING NODE│   │      │
│   │   │            │     │ (RAG + FAISS)   │     │ (LLM + Groq) │   │      │
│   │   └────────────┘     └─────────────────┘     └──────┬───────┘   │      │
│   │                                                      │           │      │
│   └──────────────────────────────────────────────────────┼───────────┘      │
│                                                          │                  │
│                                                          ▼                  │
│                                                   ┌─────────────┐           │
│                                                   │  STRUCTURED │           │
│                                                   │  JSON OUTPUT│           │
│                                                   └─────────────┘           │
│                                                          │                  │
│                                                          ▼                  │
│                                                   Streamlit UI              │
│                                                   (Risk Summary,            │
│                                                    Recommendations,         │
│                                                    Sources, Disclaimer)     │
└─────────────────────────────────────────────────────────────────────────────┘
```

The pipeline is **deterministic in structure** (same three nodes always execute in the same order) but **adaptive in content** (the risk level, retrieved strategies, and LLM output vary based on each customer's profile).

---

## 4. System Architecture

### 4.1 Layered Architecture Overview

The system is organized into six logical layers, each with a distinct responsibility:

```
┌─────────────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                            │
│          Streamlit UI + Custom CSS (Inter, Playfair Display)     │
├─────────────────────────────────────────────────────────────────┤
│                    INPUT PROCESSING LAYER                        │
│      19-feature customer profile form + input validation         │
├─────────────────────────────────────────────────────────────────┤
│                    ML INFERENCE LAYER                            │
│    LabelEncoders → Feature Ordering → StandardScaler → XGBoost  │
├─────────────────────────────────────────────────────────────────┤
│                    AGENTIC AI LAYER                              │
│          LangGraph StateGraph (Risk → Retrieval → Planning)      │
├──────────────────────────┬──────────────────────────────────────┤
│     RAG KNOWLEDGE LAYER  │         LLM REASONING LAYER          │
│  retention_knowledge.json│     Groq API (LLaMA 3.3 70B)         │
│  HuggingFace Embeddings  │     Prompt Engineering + JSON Parse  │
│  FAISS Vector Store       │                                      │
├──────────────────────────┴──────────────────────────────────────┤
│                     OUTPUT LAYER                                 │
│   Churn Prediction + AI Retention Strategy + Confidence Score    │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Technology Stack

| Layer | Technology | Version / Model | Purpose |
|-------|-----------|----------------|---------|
| Frontend | Streamlit | ≥ 1.32.0 | Interactive web UI |
| Styling | Custom CSS | Google Fonts (Inter, Playfair Display) | Premium visual design |
| ML Model | XGBoost | ≥ 1.7.0 | Binary churn classification |
| Preprocessing | Scikit-learn | ≥ 1.2.0 | LabelEncoder, StandardScaler |
| Serialization | Joblib | ≥ 1.2.0 | Model and artifact persistence |
| Agent Framework | LangGraph | StateGraph | Multi-node workflow orchestration |
| Vector Store | FAISS | In-memory | Similarity-based document retrieval |
| Embeddings | HuggingFace | all-MiniLM-L6-v2 (384-dim) | Text-to-vector encoding |
| LLM Provider | Groq API | Cloud-hosted | Fast LLM inference |
| LLM Model | LLaMA 3.3 | 70B Versatile | Reasoning and structured generation |
| Knowledge Base | JSON | 9 curated entries | Domain-specific retention strategies |
| Environment | python-dotenv | — | Secure API key management |

### 4.3 File Structure

```
project/
├── app.py                          # Main application (750 lines)
├── retention_knowledge.json        # RAG knowledge base (9 strategies)
├── model_test.py                   # Model validation script
├── requirements.txt                # Python dependencies
├── .env                            # API keys (GROQ_API_KEY)
├── .streamlit/                     # Streamlit config
├── notebook_&_otherpkl/
│   ├── final_churn_model.pkl       # Trained XGBoost model
│   ├── scaler.pkl                  # StandardScaler (3 numeric columns)
│   ├── threshold.pkl               # Optimized threshold (0.4)
│   ├── encoders.pkl                # LabelEncoders (16 categorical cols)
│   ├── feature_order.pkl           # Column ordering from training
│   ├── CUSTOMER_CHURN_PREDICTION_*.ipynb  # Training notebook
│   └── milestone2_rag_testing.ipynb       # RAG testing notebook
├── Raw_Dataset/                    # Original Telco churn CSV
├── EDA Insights/                   # Exploratory data analysis
├── Report/                         # Project documentation
└── images/                         # UI assets
```

---

## 5. Agent Workflow Using LangGraph

### 5.1 Why LangGraph?

LangGraph was chosen over alternatives (LangChain Agents, raw function chaining) for the following reasons:

- **Explicit state management** — The `AgentState` TypedDict provides type-safe, transparent data flow between nodes. Every field is declared, and every transition is visible.
- **Deterministic execution** — Unlike reactive agents that decide their next action at runtime, this pipeline always executes `risk → retrieval → planning` in the same order. LangGraph's `add_edge` enforces this guarantee.
- **Debuggability** — Each node receives and returns the full state, making it straightforward to inspect intermediate values at any stage.
- **Extensibility** — Adding new nodes (e.g., a "validation node" or "feedback node") requires only `add_node` + `add_edge` — no refactoring.

### 5.2 Graph Construction

```python
# Build the LangGraph pipeline
builder = StateGraph(AgentState)

# Register the three processing nodes
builder.add_node("risk", risk_node)
builder.add_node("retrieval", retrieval_node)
builder.add_node("planning", planning_node)

# Define the execution order
builder.set_entry_point("risk")
builder.add_edge("risk", "retrieval")
builder.add_edge("retrieval", "planning")

# Compile into an executable graph
graph = builder.compile()
```

### 5.3 Agent Flow Diagram

```
                    ┌──────────────────────┐
                    │      ENTRY POINT     │
                    │   graph.invoke({     │
                    │     churn_prob,       │
                    │     tenure,           │
                    │     monthly           │
                    │   })                  │
                    └──────────┬───────────┘
                               │
                               ▼
                ┌──────────────────────────┐
                │     NODE 1: risk_node    │
                │                          │
                │  Input:                  │
                │   • churn_prob           │
                │   • tenure              │
                │   • monthly             │
                │                          │
                │  Logic:                  │
                │   • prob > 0.7 → High   │
                │   • prob > 0.4 → Medium │
                │   • else     → Low      │
                │   • Check tenure < 6    │
                │   • Check monthly > 80  │
                │                          │
                │  Output (added):         │
                │   • risk_level           │
                │   • reasons[]            │
                └──────────┬───────────────┘
                           │
                           ▼
                ┌──────────────────────────┐
                │  NODE 2: retrieval_node  │
                │                          │
                │  Input:                  │
                │   • reasons[]            │
                │                          │
                │  Logic:                  │
                │   • Join reasons → query │
                │   • FAISS search (k=3)   │
                │   • Extract strategies   │
                │   • Extract sources      │
                │                          │
                │  Output (added):         │
                │   • strategies[]         │
                │   • sources[]            │
                └──────────┬───────────────┘
                           │
                           ▼
                ┌──────────────────────────┐
                │  NODE 3: planning_node   │
                │                          │
                │  Input:                  │
                │   • churn_prob           │
                │   • risk_level           │
                │   • reasons[]            │
                │   • strategies[]         │
                │   • sources[]            │
                │                          │
                │  Logic:                  │
                │   • Build structured     │
                │     prompt               │
                │   • Call Groq API        │
                │     (LLaMA 3.3 70B)     │
                │   • Parse JSON response  │
                │   • Fallback on error    │
                │                          │
                │  Output (added):         │
                │   • final_output (JSON)  │
                └──────────┬───────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │   END STATE  │
                    │              │
                    │ Complete     │
                    │ AgentState   │
                    │ returned     │
                    └──────────────┘
```

---

## 6. Agent State Design

The `AgentState` is a Python `TypedDict` that serves as the **shared data contract** between all agent nodes. It is passed through the pipeline, with each node reading from it and appending new fields.

### 6.1 State Definition

```python
class AgentState(TypedDict):
    # --- Inputs (set before graph.invoke) ---
    churn_prob: float          # ML model output: P(churn) ∈ [0.0, 1.0]
    tenure: int                # Customer tenure in months (0–72)
    monthly: float             # Monthly charges in USD ($18–$120)
    
    # --- Set by risk_node ---
    risk_level: str            # Categorical risk: "High" | "Medium" | "Low"
    reasons: List[str]         # Identified churn reasons (e.g., ["low_tenure", "high_charges"])
    
    # --- Set by retrieval_node ---
    strategies: List[str]      # Retrieved retention strategies from FAISS
    sources: List[str]         # Academic/industry source citations
    
    # --- Set by planning_node ---
    final_output: str          # Structured JSON from LLM (parsed as dict at runtime)
```

### 6.2 Field-by-Field Explanation

| Field | Type | Set By | Description |
|-------|------|--------|-------------|
| `churn_prob` | `float` | Caller | The raw churn probability from the XGBoost model. Range: 0.0 to 1.0. |
| `tenure` | `int` | Caller | Customer tenure in months. Used by the risk node to detect early-lifecycle churn risk (tenure < 6). |
| `monthly` | `float` | Caller | Monthly charges in USD. Used by the risk node to detect cost-driven churn (monthly > $80). |
| `risk_level` | `str` | `risk_node` | Categorical interpretation of `churn_prob`. Thresholds: >0.7 = High, >0.4 = Medium, ≤0.4 = Low. |
| `reasons` | `List[str]` | `risk_node` | List of identified churn risk factors (e.g., `"low_tenure"`, `"high_charges"`, `"general"`). Used as the RAG search query. |
| `strategies` | `List[str]` | `retrieval_node` | Top-3 retention strategies retrieved from the FAISS vector store. Contains the full document text (condition + strategy). |
| `sources` | `List[str]` | `retrieval_node` | Corresponding source citations for each retrieved strategy (e.g., "Harvard Business Review", "McKinsey"). |
| `final_output` | `str` | `planning_node` | The LLM-generated JSON output containing risk_summary, recommendations, sources, and disclaimer. Stored as a dict at runtime. |

### 6.3 State Flow Through Pipeline

```
graph.invoke({                            ← Initial state (3 fields)
    "churn_prob": 0.82,
    "tenure": 4,
    "monthly": 95.0
})

After risk_node:                           ← +2 fields
    risk_level = "High"
    reasons = ["low_tenure", "high_charges"]

After retrieval_node:                      ← +2 fields
    strategies = ["Condition: low_tenure\nStrategy: ...", ...]
    sources = ["HubSpot (Link 11)", "HBS Working Knowledge (Link 2)", ...]

After planning_node:                       ← +1 field
    final_output = {
        "risk_summary": "...",
        "recommendations": ["...", "...", "..."],
        "sources": ["...", "..."],
        "disclaimer": "..."
    }
```

---

## 7. Detailed Node Explanations

### 7.1 Node 1: Risk Assessment (`risk_node`)

**Purpose:** Convert the raw churn probability into a human-interpretable risk level and identify the specific reasons contributing to churn risk.

**Why this node exists:** A raw probability like `0.73` is not actionable for business stakeholders. This node transforms it into a categorical label ("High") and identifies root causes ("low_tenure", "high_charges") that drive the RAG retrieval query.

#### Code Implementation

```python
def risk_node(state: AgentState):
    prob = state["churn_prob"]

    # Classify risk level based on probability thresholds
    if prob > 0.7:
        risk = "High"
    elif prob > 0.4:
        risk = "Medium"
    else:
        risk = "Low"

    # Identify specific churn reasons based on customer attributes
    reasons = []

    if state["tenure"] < 6:
        reasons.append("low_tenure")

    if state["monthly"] > 80:
        reasons.append("high_charges")

    # Default to "general" if no specific triggers are detected
    if not reasons:
        reasons.append("general")

    return {**state, "risk_level": risk, "reasons": reasons}
```

#### Logic Breakdown

| Condition | Risk Level | Rationale |
|-----------|-----------|-----------|
| `prob > 0.7` | **High** | Strong churn signal — immediate intervention needed |
| `0.4 < prob ≤ 0.7` | **Medium** | Moderate risk — proactive engagement recommended |
| `prob ≤ 0.4` | **Low** | Stable customer — maintain standard support |

| Condition | Reason Tag | Rationale |
|-----------|-----------|-----------|
| `tenure < 6 months` | `low_tenure` | Industry data shows highest churn rates occur within the first 6 months |
| `monthly > $80` | `high_charges` | High monthly costs are a top driver of voluntary churn in telecom |
| Neither of the above | `general` | Catch-all for customers without specific risk triggers |

#### Design Decision: Why Rule-Based?

The risk node uses simple rule-based logic rather than a secondary ML model. This is intentional:

- **Transparency** — Business stakeholders can understand exactly why a risk level was assigned.
- **Speed** — No additional model inference overhead.
- **Alignment with RAG** — The reason tags (`low_tenure`, `high_charges`, `general`) directly correspond to the `condition` field in the knowledge base, ensuring high-quality retrieval.

---

### 7.2 Node 2: Strategy Retrieval — RAG (`retrieval_node`)

**Purpose:** Retrieve the most relevant retention strategies from a curated knowledge base using vector similarity search. This grounds the final LLM output in verified domain knowledge.

**Why this node exists:** Without RAG, the LLM would generate strategies from its training data — generic, unverifiable, and potentially hallucinated. By constraining the LLM to retrieved strategies with known sources, the system produces **trustworthy, citation-backed recommendations**.

#### Code Implementation

```python
def retrieval_node(state: AgentState):
    # Convert reason tags into a search query
    query = " ".join(state["reasons"])
    
    # Perform similarity search against the FAISS vector store
    results = vectorstore.similarity_search(query, k=3)
    
    # Extract strategy text and source citations
    strategies = []
    sources = []

    for doc in results:
        strategies.append(doc.page_content)
        sources.append(doc.metadata["source"])

    return {
        **state,
        "strategies": list(set(strategies)),  # Deduplicate
        "sources": list(set(sources))          # Deduplicate
    }
```

#### Retrieval Mechanism

```
reasons = ["low_tenure", "high_charges"]
                │
                ▼
        query = "low_tenure high_charges"
                │
                ▼
    ┌───────────────────────────────────────────┐
    │           FAISS Vector Store               │
    │                                           │
    │  ┌─────────────────────────────────────┐  │
    │  │ Doc 1: "low_tenure → onboarding"    │  │  ← Cosine similarity
    │  │ Doc 2: "low_tenure → early value"   │  │     against query
    │  │ Doc 3: "high_charges → pricing"     │  │     embedding
    │  │ Doc 4: "high_charges → optimize"    │  │
    │  │ Doc 5: "low_engagement → comms"     │  │
    │  │ Doc 6: "low_engagement → proactive" │  │
    │  │ Doc 7: "no_support → success teams" │  │
    │  │ Doc 8: "general → churn prediction" │  │
    │  │ Doc 9: "general → experience-led"   │  │
    │  └─────────────────────────────────────┘  │
    │                                           │
    │  Top 3 results returned (k=3)             │
    └───────────────────────────────────────────┘
                │
                ▼
    strategies = [
        "Condition: low_tenure\nStrategy: Implement strong onboarding...",
        "Condition: low_tenure\nStrategy: Deliver value early...",
        "Condition: high_charges\nStrategy: Offer flexible pricing..."
    ]
    sources = [
        "Customer Onboarding Best Practices – HubSpot (Link 11)",
        "The Value of Keeping the Right Customers – Harvard Business Review (Link 1)",
        "Managing Churn to Maximize Profits – HBS Working Knowledge (Link 2)"
    ]
```

#### Why `k=3`?

- **k=1** would be too narrow — may miss complementary strategies.
- **k=5** would introduce noise — irrelevant strategies could dilute the LLM prompt.
- **k=3** balances coverage and precision for a 9-document knowledge base.

#### Why `list(set(...))`?

Deduplication ensures that if two similar documents are retrieved (e.g., two `low_tenure` strategies), duplicate content does not inflate the prompt or produce redundant recommendations.

---

### 7.3 Node 3: Planning & Recommendation — LLM (`planning_node`)

**Purpose:** Synthesize the risk assessment and retrieved strategies into a coherent, structured, and human-readable retention plan using an LLM.

**Why this node exists:** Raw retrieved documents are not directly presentable to business users. The LLM acts as a **reasoning layer** that interprets the risk context, selects the most relevant strategies, and packages them into a structured JSON output with clear language.

#### Code Implementation

```python
def planning_node(state: AgentState):

    prompt = f"""
You are an AI Customer Retention Strategist.

Customer churn probability: {state['churn_prob']}
Risk level: {state['risk_level']}
Reasons: {state['reasons']}

Retrieved Strategies: {state['strategies']}
Sources: {state['sources']}

IMPORTANT RULES:
- Use ONLY the provided strategies and sources
- Do NOT generate new strategies
- If no relevant strategy, say "No recommendation found"

STRICT OUTPUT FORMAT (JSON ONLY):

{{
  "risk_summary": "short explanation of churn risk",
  "recommendations": ["action 1", "action 2", "action 3"],
  "sources": ["source1", "source2"],
  "disclaimer": "This prediction is probabilistic and may not guarantee actual churn."
}}

ONLY return valid JSON. No extra text.
"""

    # Call Groq API with LLaMA 3.3 70B
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    raw_output = response.choices[0].message.content

    # Attempt JSON parsing with graceful fallback
    try:
        parsed_output = json.loads(raw_output)
    except:
        parsed_output = {
            "risk_summary": "Parsing error",
            "recommendations": [],
            "sources": [],
            "disclaimer": "Model output could not be parsed"
        }

    return {**state, "final_output": parsed_output}
```

#### LLM Call Flow

```
┌─────────────────────┐       ┌──────────────────┐       ┌───────────────────┐
│   planning_node     │       │    Groq API      │       │  LLaMA 3.3 70B   │
│                     │       │                  │       │  Versatile        │
│  Build prompt with: │       │                  │       │                   │
│  • churn_prob       │──────▶│  HTTP POST       │──────▶│  Generate         │
│  • risk_level       │       │  /chat/          │       │  structured       │
│  • reasons[]        │       │  completions     │       │  JSON response    │
│  • strategies[]     │       │                  │       │                   │
│  • sources[]        │       │                  │◀──────│                   │
│  • anti-hallucinate │◀──────│  JSON response   │       │                   │
│    rules            │       │                  │       │                   │
│                     │       │                  │       │                   │
│  json.loads(output) │       └──────────────────┘       └───────────────────┘
│  or fallback dict   │
└─────────────────────┘
```

---

## 8. RAG Pipeline Explanation

### 8.1 What is RAG?

**Retrieval-Augmented Generation (RAG)** is a technique that enhances LLM outputs by first retrieving relevant information from an external knowledge base, then providing that information as context to the LLM. This grounds the LLM's response in verified data rather than relying solely on its parametric knowledge.

### 8.2 Knowledge Base Structure

The knowledge base (`retention_knowledge.json`) contains 9 curated retention strategies organized by churn condition:

```json
{
    "condition": "low_tenure",
    "strategy": "Implement strong onboarding and early engagement programs...",
    "source": "Customer Onboarding Best Practices – HubSpot (Link 11)"
}
```

| Condition | Count | Description |
|-----------|-------|-------------|
| `low_tenure` | 2 | Strategies for early-lifecycle customers (< 6 months) |
| `high_charges` | 2 | Strategies for cost-sensitive customers (> $80/month) |
| `low_engagement` | 2 | Strategies for disengaged customers |
| `no_support` | 1 | Strategies for customers without support access |
| `general` | 2 | Broadly applicable retention strategies |

**Sources include:** Harvard Business Review, McKinsey, Bain & Company, HubSpot, HBS Working Knowledge, MDPI.

### 8.3 Document Conversion

Each JSON entry is converted into a LangChain `Document` object for indexing:

```python
from langchain_core.documents import Document

docs = []
for item in knowledge:
    content = f"Condition: {item['condition']}\nStrategy: {item['strategy']}"
    
    docs.append(
        Document(
            page_content=content,      # Searchable text
            metadata={
                "source": item["source"],       # Citation
                "condition": item["condition"]  # Category tag
            }
        )
    )
```

**Design choice:** The `condition` is embedded in `page_content` alongside the strategy so that the embedding captures both the context and the recommendation. Metadata stores the source citation separately for clean extraction.

### 8.4 Embedding Model

```python
from langchain_community.embeddings import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
```

| Property | Value |
|----------|-------|
| Model | `all-MiniLM-L6-v2` |
| Provider | HuggingFace / Sentence-Transformers |
| Embedding Dimension | 384 |
| Training Objective | Semantic similarity |
| Speed | ~14,000 sentences/second on CPU |
| Why chosen | Lightweight, fast, runs locally without GPU, excellent for short-text similarity |

### 8.5 FAISS Vector Store

```python
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(docs, embedding_model)
```

**FAISS** (Facebook AI Similarity Search) provides:

- **In-memory operation** — No external database needed. The 9-document index fits entirely in RAM.
- **Fast nearest-neighbor search** — Even with brute-force search, 9 documents return results in microseconds.
- **Cosine similarity** — Documents are ranked by how semantically similar they are to the query.

### 8.6 Similarity Search

```python
# Query constructed from churn reasons
query = " ".join(state["reasons"])  # e.g., "low_tenure high_charges"

# Retrieve top 3 most similar documents
results = vectorstore.similarity_search(query, k=3)
```

**How it works:**
1. The query string (e.g., `"low_tenure high_charges"`) is embedded into a 384-dimensional vector using the same `all-MiniLM-L6-v2` model.
2. FAISS computes the cosine similarity between this query vector and all 9 document vectors.
3. The top 3 most similar documents are returned, ranked by relevance.

### 8.7 RAG Pipeline Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                     RAG PIPELINE                                      │
│                                                                       │
│  INDEXING (at app startup):                                          │
│                                                                       │
│  retention_knowledge.json                                            │
│       │                                                               │
│       ▼                                                               │
│  9 JSON entries ──▶ 9 LangChain Documents ──▶ all-MiniLM-L6-v2      │
│                                                       │               │
│                                                       ▼               │
│                                               9 × 384-dim vectors    │
│                                                       │               │
│                                                       ▼               │
│                                               FAISS Index (in-RAM)    │
│                                                                       │
│  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  │
│                                                                       │
│  RETRIEVAL (at prediction time):                                     │
│                                                                       │
│  reasons[] ──▶ "low_tenure high_charges" ──▶ all-MiniLM-L6-v2       │
│                                                       │               │
│                                                       ▼               │
│                                               Query vector (384-dim)  │
│                                                       │               │
│                                                       ▼               │
│                                             FAISS cosine similarity   │
│                                                       │               │
│                                                       ▼               │
│                                             Top 3 documents returned  │
│                                                       │               │
│                                                       ▼               │
│                                          strategies[] + sources[]     │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 9. Prompt Engineering Strategy

### 9.1 Anti-Hallucination Design

The prompt is carefully engineered to prevent the LLM from generating fabricated strategies. Three explicit constraints are enforced:

```
IMPORTANT RULES:
- Use ONLY the provided strategies and sources     ← Constraint 1: Source-bound
- Do NOT generate new strategies                   ← Constraint 2: No fabrication
- If no relevant strategy, say "No recommendation  ← Constraint 3: Graceful absence
  found"
```

| Rule | Purpose |
|------|---------|
| **Source-bound** | Forces the LLM to work exclusively with retrieved content, not its parametric knowledge |
| **No fabrication** | Explicitly prohibits the LLM from inventing new strategies |
| **Graceful absence** | Prevents the LLM from hallucinating when no relevant strategy exists |

### 9.2 Structured Context Injection

The prompt injects the full agent state as structured context:

```
Customer churn probability: {state['churn_prob']}    ← Numeric precision
Risk level: {state['risk_level']}                    ← Categorical context
Reasons: {state['reasons']}                          ← Root cause tags
Retrieved Strategies: {state['strategies']}          ← RAG output (grounding)
Sources: {state['sources']}                          ← Citation data
```

This gives the LLM everything it needs to reason about the customer's situation without requiring it to infer or guess.

### 9.3 Output Format Enforcement

```
STRICT OUTPUT FORMAT (JSON ONLY):

{
  "risk_summary": "short explanation of churn risk",
  "recommendations": ["action 1", "action 2", "action 3"],
  "sources": ["source1", "source2"],
  "disclaimer": "This prediction is probabilistic..."
}

ONLY return valid JSON. No extra text.
```

- **"STRICT"** and **"ONLY"** keywords signal high priority to the LLM.
- The template shows the exact key names and value types expected.
- **"No extra text"** prevents the LLM from wrapping JSON in markdown or adding commentary.

---

## 10. LLM Output Structure

### 10.1 Expected JSON Schema

```json
{
  "risk_summary": "string — 1-2 sentence explanation of why the customer is at risk",
  "recommendations": [
    "string — actionable retention action 1",
    "string — actionable retention action 2",
    "string — actionable retention action 3"
  ],
  "sources": [
    "string — academic/industry citation 1",
    "string — academic/industry citation 2"
  ],
  "disclaimer": "string — probabilistic warning"
}
```

### 10.2 Field Descriptions

| Field | Type | Purpose |
|-------|------|---------|
| `risk_summary` | `string` | A concise, human-readable explanation of the customer's churn risk, synthesized from the probability, risk level, and reasons. |
| `recommendations` | `string[]` | A list of 2–3 specific, actionable retention strategies drawn directly from the retrieved knowledge base. |
| `sources` | `string[]` | The academic and industry sources backing the recommendations, extracted from document metadata. |
| `disclaimer` | `string` | A standard probabilistic disclaimer reminding stakeholders that predictions are not guarantees. |

### 10.3 How Output is Displayed

The JSON is parsed and rendered in the Streamlit UI:

```python
output = result["final_output"]

st.subheader("Risk Summary")
st.write(output["risk_summary"])

st.subheader("Recommendations")
for rec in output["recommendations"]:
    st.write("•", rec)

st.subheader("Sources")
for src in output["sources"]:
    st.write("-", src)

st.subheader("Disclaimer")
st.info(output["disclaimer"])
```

---

## 11. System Robustness & Error Handling

### 11.1 Model Loading Safety

All ML artifacts (model, scaler, threshold, encoders, feature_order) are loaded with try/except blocks. If any file is missing, the app halts gracefully with `st.stop()`:

```python
@st.cache_resource
def load_model():
    try:
        return joblib.load("notebook_&_otherpkl/final_churn_model.pkl")
    except Exception:
        st.error("Error loading model file")
        st.stop()  # Prevents the app from proceeding with a missing model
```

### 11.2 Input Validation

User inputs are validated before model inference:

```python
if monthly < 0 or total_c < 0:
    st.warning("Charges cannot be negative.")
    st.stop()

if tenure < 0:
    st.warning("Tenure cannot be negative.")
    st.stop()
```

### 11.3 LLM Output Parsing — Graceful Fallback

The most critical error handling is in `planning_node`. LLMs can occasionally produce malformed JSON (trailing commas, markdown wrapping, extra text). The system handles this with a structured fallback:

```python
try:
    parsed_output = json.loads(raw_output)
except:
    parsed_output = {
        "risk_summary": "Parsing error",
        "recommendations": [],
        "sources": [],
        "disclaimer": "Model output could not be parsed"
    }
```

**Why a fallback dict instead of raising an exception?**
- The user still receives the ML prediction (churn probability + binary label).
- The UI does not crash — it simply shows "Parsing error" instead of the LLM strategy.
- The fallback dictionary has the same schema as a successful response, so the UI rendering code works without branching.

### 11.4 Caching for Performance

All heavy resources are cached with `@st.cache_resource`, preventing repeated loading on each Streamlit rerun:

```python
@st.cache_resource
def load_model():      ...  # Loaded once, reused across sessions
@st.cache_resource
def load_scaler():     ...
@st.cache_resource
def load_threshold():  ...
@st.cache_resource
def load_encoders():   ...
@st.cache_resource
def load_feature_order(): ...
```

The FAISS index and embedding model are also initialized once at module level, avoiding redundant computation.

### 11.5 API Key Security

The Groq API key is stored in `.env` and loaded via `python-dotenv`, never hardcoded:

```python
from dotenv import load_dotenv
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
```

---

## 12. End-to-End Flow

The complete request lifecycle, from user interaction to displayed result:

### Step-by-Step Execution

| Step | Component | Action | Output |
|------|-----------|--------|--------|
| 1 | **Streamlit UI** | User fills in 19 customer profile fields (gender, tenure, contract, charges, etc.) | Raw input values |
| 2 | **Streamlit UI** | User clicks **"✦ Run Churn Prediction"** button | Triggers prediction pipeline |
| 3 | **Input Validation** | Check for negative charges/tenure | Pass or halt with warning |
| 4 | **LabelEncoders** | Encode 16 categorical features using saved encoders | Numerical encoded values |
| 5 | **Feature Ordering** | Reorder DataFrame columns to match training order (`feature_order.pkl`) | Aligned DataFrame |
| 6 | **StandardScaler** | Scale tenure, MonthlyCharges, TotalCharges using saved scaler | Standardized numerical features |
| 7 | **XGBoost Model** | `model.predict_proba(input_df)[0][1]` | `churn_prob` ∈ [0.0, 1.0] |
| 8 | **graph.invoke()** | Pass `{churn_prob, tenure, monthly}` to LangGraph | Initiates agent pipeline |
| 9 | **risk_node** | Classify risk level + identify churn reasons | `risk_level`, `reasons[]` |
| 10 | **retrieval_node** | FAISS similarity search with reasons as query (k=3) | `strategies[]`, `sources[]` |
| 11 | **planning_node** | Build prompt → Call Groq API (LLaMA 3.3 70B) → Parse JSON | `final_output` (dict) |
| 12 | **Threshold Check** | `prob >= 0.4` → churn / no churn label | Binary classification |
| 13 | **Streamlit UI** | Display: progress bar, confidence score, risk summary, recommendations, sources, disclaimer, result card, probability card | Complete visual output |

### Execution Flow Diagram

```
User fills 19 inputs
        │
        ▼
  Click "Run Churn Prediction"
        │
        ▼
  Input Validation ──── FAIL ──▶ st.warning() + st.stop()
        │
      PASS
        │
        ▼
  Encode (LabelEncoders) ──▶ Reorder (feature_order) ──▶ Scale (StandardScaler)
        │
        ▼
  XGBoost predict_proba()
        │
        ├──▶ churn_prob ──▶ Threshold (0.4) ──▶ "Churn" / "No Churn"
        │
        └──▶ graph.invoke({churn_prob, tenure, monthly})
                    │
                    ▼
             ┌─────────────┐
             │  risk_node   │ ──▶ risk_level + reasons[]
             └──────┬──────┘
                    │
                    ▼
             ┌─────────────┐
             │  retrieval   │ ──▶ strategies[] + sources[]
             │  _node       │
             └──────┬──────┘
                    │
                    ▼
             ┌─────────────┐
             │  planning    │ ──▶ Groq API ──▶ JSON output
             │  _node       │
             └──────┬──────┘
                    │
                    ▼
            final_output (dict)
                    │
                    ▼
        ┌───────────────────────┐
        │    Streamlit UI       │
        │                       │
        │  • Progress bar       │
        │  • Confidence score   │
        │  • Risk Summary       │
        │  • Recommendations    │
        │  • Sources            │
        │  • Disclaimer         │
        │  • Result card        │
        │  • Probability card   │
        └───────────────────────┘
```

---

## 13. Example Input & Output

### 13.1 Example Input

A high-risk customer profile:

| Feature | Value |
|---------|-------|
| Gender | Female |
| Senior Citizen | No |
| Partner | No |
| Dependents | No |
| **Tenure** | **4 months** |
| Phone Service | Yes |
| Multiple Lines | No |
| Internet Service | Fiber optic |
| Online Security | No |
| Online Backup | No |
| Device Protection | No |
| Tech Support | No |
| Streaming TV | Yes |
| Streaming Movies | Yes |
| Contract | Month-to-month |
| Paperless Billing | Yes |
| Payment Method | Electronic check |
| **Monthly Charges** | **$95.00** |
| Total Charges | $380.00 |

### 13.2 ML Prediction Output

```
Churn Probability: 0.82
Threshold: 0.40
Prediction: Customer WILL churn
Confidence Score: 0.64
```

### 13.3 Agent Pipeline Execution

**risk_node output:**
```json
{
  "risk_level": "High",
  "reasons": ["low_tenure", "high_charges"]
}
```

**retrieval_node output:**
```json
{
  "strategies": [
    "Condition: low_tenure\nStrategy: Implement strong onboarding and early engagement programs to help customers realize value quickly",
    "Condition: low_tenure\nStrategy: Deliver value early in the customer lifecycle to prevent early-stage churn",
    "Condition: high_charges\nStrategy: Offer flexible pricing plans or targeted discounts to reduce cost burden"
  ],
  "sources": [
    "Customer Onboarding Best Practices – HubSpot (Link 11)",
    "The Value of Keeping the Right Customers – Harvard Business Review (Link 1)",
    "Managing Churn to Maximize Profits – HBS Working Knowledge (Link 2)"
  ]
}
```

**planning_node output (LLM-generated):**
```json
{
  "risk_summary": "This customer has a high churn probability of 82%. The primary risk factors include a very short tenure of only 4 months and high monthly charges of $95.00, both of which are strong indicators of imminent churn in the telecommunications sector.",
  "recommendations": [
    "Implement a personalized onboarding program to ensure the customer quickly realizes the value of their current plan and services",
    "Deliver early value through complimentary premium features or service upgrades during the first 6 months to build loyalty",
    "Offer a targeted pricing adjustment or loyalty discount to reduce the perceived cost burden of the $95/month plan"
  ],
  "sources": [
    "Customer Onboarding Best Practices – HubSpot (Link 11)",
    "The Value of Keeping the Right Customers – Harvard Business Review (Link 1)",
    "Managing Churn to Maximize Profits – HBS Working Knowledge (Link 2)"
  ],
  "disclaimer": "This prediction is probabilistic and may not guarantee actual churn."
}
```

### 13.4 UI Display

The Streamlit interface renders this as:

- **Progress bar** filled to 82%
- **Confidence Score:** 0.64
- **Risk Summary:** "This customer has a high churn probability of 82%..."
- **Recommendations:**
  - • Implement a personalized onboarding program...
  - • Deliver early value through complimentary premium features...
  - • Offer a targeted pricing adjustment...
- **Sources:** HubSpot, Harvard Business Review, HBS Working Knowledge
- **Disclaimer:** Shown in a Streamlit info box
- **Result Card:** Red gradient — "Customer Likely to Churn"
- **Probability Card:** 82.0% with threshold annotation

---

## 14. Key Design Decisions

### 14.1 Why LangGraph over LangChain Agents?

| Factor | LangChain Agent | LangGraph (chosen) |
|--------|----------------|-------------------|
| Execution flow | Dynamic (LLM decides next step) | Deterministic (edges define flow) |
| Predictability | Low — different runs may take different paths | High — always risk → retrieval → planning |
| Debuggability | Hard — internal reasoning is opaque | Easy — inspect state at each node |
| Latency | Higher — multiple LLM calls for routing | Lower — only one LLM call (planning node) |
| Suitability | Open-ended tasks | Well-defined pipelines |

**Decision:** The retention strategy pipeline has a fixed, known structure. There is no value in letting the LLM decide what to do next — it should always assess risk, retrieve strategies, then generate a plan. LangGraph enforces this.

### 14.2 Why Groq (LLaMA 3.3 70B) over Google Gemini?

| Factor | Gemini | Groq + LLaMA 3.3 70B (chosen) |
|--------|--------|-------------------------------|
| Latency | Moderate | Very fast (Groq's custom hardware) |
| Cost | Pay-per-token | Free tier available |
| JSON compliance | Good | Good — 70B model follows structured output well |
| Availability | Requires Google Cloud setup | Simple API key |

**Decision:** Groq provides extremely fast inference with a generous free tier, making it ideal for a student/portfolio project. LLaMA 3.3 70B Versatile is large enough to follow structured JSON instructions reliably.

### 14.3 Why FAISS over ChromaDB / Pinecone?

| Factor | Pinecone | ChromaDB | FAISS (chosen) |
|--------|----------|----------|---------------|
| Hosting | Cloud | Local | Local |
| Setup | API key + account | pip install | pip install |
| Performance | High | Good | High |
| Scale | Millions of docs | Medium scale | Millions of docs |
| Cost | Paid | Free | Free |
| Persistence | Built-in | Built-in | In-memory (default) |

**Decision:** With only 9 documents in the knowledge base, FAISS's in-memory operation is ideal — zero setup, no external dependencies, and sub-millisecond retrieval. There is no need for a persistent or cloud-hosted vector database at this scale.

### 14.4 Why all-MiniLM-L6-v2?

- **Runs locally** — No API calls needed for embedding generation.
- **384-dimensional output** — Compact but semantically rich.
- **Trained on 1B+ sentence pairs** — Excellent semantic similarity performance.
- **Fast** — Suitable for real-time applications without GPU.

### 14.5 Why Threshold 0.4 Instead of 0.5?

The default classification threshold of 0.5 balances precision and recall equally. This system uses **0.4** to:

- **Prioritize recall** — It is more costly to miss a churning customer (false negative) than to falsely flag a stable customer (false positive).
- **Enable proactive intervention** — A lower threshold catches borderline cases early, giving the business team more time to act.

### 14.6 Why Rule-Based Risk Node?

The risk node could have been an LLM call, but rule-based logic was chosen for:

- **Speed** — No API latency.
- **Transparency** — Stakeholders can see exactly why a risk level was assigned.
- **Consistency** — Same inputs always produce the same risk level (no LLM randomness).
- **Cost** — One fewer API call per prediction.

---

## 15. Limitations

### 15.1 Knowledge Base Scale

The current knowledge base contains only **9 entries** across 5 condition categories. This limits the diversity of recommendations for edge cases or multi-factor churn scenarios.

### 15.2 Reason Detection

Only two customer attributes are checked for reason tagging:
- `tenure < 6` → `low_tenure`
- `monthly > 80` → `high_charges`

Other potential churn indicators (contract type, internet service type, lack of tech support) are not captured as distinct reason tags.

### 15.3 Single LLM Call

The planning node makes a single LLM call without validation. If the LLM produces suboptimal strategies (e.g., repeating the same recommendation twice), there is no self-critique or refinement loop.

### 15.4 No Feedback Loop

The system does not track whether recommended strategies were effective. There is no mechanism to learn from past retention outcomes and improve future recommendations.

### 15.5 In-Memory FAISS

The FAISS index is rebuilt from scratch every time the Streamlit app restarts. For larger knowledge bases, this could become a performance bottleneck.

### 15.6 LLM Dependencies

The system depends on:
- **Groq API availability** — If Groq is down, the planning node fails.
- **Internet connectivity** — Required for LLM inference.
- **API rate limits** — Free-tier limits may affect high-volume usage.

---

## 16. Future Enhancements

### 16.1 Expanded Knowledge Base

- Increase from 9 to 50+ strategies covering more conditions (contract type, service type, payment method, customer demographics).
- Support multi-source knowledge (PDFs, research papers, CRM data).

### 16.2 Advanced Reason Detection

- Use SHAP (SHapley Additive exPlanations) values from the XGBoost model to identify the top contributing features per prediction, replacing hand-coded rules.
- Map SHAP feature importance to condition tags dynamically.

### 16.3 Multi-Agent Architecture

- Add a **Validation Node** that checks the LLM output for quality (e.g., are recommendations distinct? do sources match?).
- Add a **Personalization Node** that tailors recommendations based on customer segment.

### 16.4 Persistent Vector Store

- Save the FAISS index to disk for faster startup.
- Implement incremental indexing for growing knowledge bases.

### 16.5 Feedback Integration

- Track which recommendations were acted upon and their outcomes.
- Use reinforcement learning from human feedback (RLHF) to improve strategy selection.

### 16.6 Multi-Model Support

- Allow switching between LLM providers (Groq, Gemini, OpenAI) based on availability and cost.
- Implement model fallback chains for resilience.

### 16.7 Batch Processing

- Support bulk customer analysis (CSV upload) with parallel agent execution.
- Generate aggregate retention reports for business intelligence teams.

---

## 17. Conclusion

The **Customer Churn Intelligence System** demonstrates the successful integration of classical machine learning with modern agentic AI architecture. By combining:

- **XGBoost** for accurate churn probability estimation,
- **LangGraph** for deterministic, stateful agent orchestration,
- **FAISS + HuggingFace Embeddings** for grounded knowledge retrieval,
- **Groq (LLaMA 3.3 70B)** for structured reasoning and recommendation generation,

the system transforms a simple binary prediction into **actionable, source-backed business intelligence**.

The three-node agent pipeline (Risk → Retrieval → Planning) ensures that every recommendation is:

1. **Contextually relevant** — informed by the customer's specific risk factors.
2. **Evidence-grounded** — sourced from curated domain knowledge, not hallucinated.
3. **Actionable** — formatted as clear, implementable retention strategies.
4. **Transparent** — every recommendation includes its source citation.

This architecture serves as a blueprint for building production-grade AI systems where **trustworthiness, explainability, and practical utility** are non-negotiable requirements.

---

**Document Version:** 1.0  
**Last Updated:** April 2026  
**System:** Customer Churn Intelligence System (Agentic AI Version)  
**Author:** Project Team  
**Stack:** Python · Streamlit · XGBoost · LangGraph · FAISS · Groq · LLaMA 3.3 70B
