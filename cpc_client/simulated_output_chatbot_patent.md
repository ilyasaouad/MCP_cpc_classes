# Simulated Phase 1 & Phase 5 Output

## Patent: LLM-Based Chatbot with Role Interchange

---

## PHASE 1: LLM EXTRACTION OUTPUT

### Technical Object
A computer-implemented query-response method using an LLM-based chatbot system that dynamically interchanges USER and CHATBOT role attributes to generate responses oriented toward the CHATBOT attribute, enabling more concrete and targeted dialog outputs.

### Problem to be Solved
Conventional chatbot systems generate responses oriented exclusively to the USER attribute, limiting the system's ability to produce assistant-like, concrete responses when a chatbot assistant entity triggers a query within the dialog.

### Solution Summary
The system defines three participant types with attributes (SYSTEM/T1, USER/T2, CHATBOT/T3). When the CHATBOT participant sends a message, the system interchanges the USER and CHATBOT attributes so the LLM generates a response oriented to the CHATBOT attribute rather than the USER attribute. This is implemented via API-operated chatbot systems with multi-phase chat history management.

### System Context
**Natural language processing and conversational AI systems**, specifically LLM-based chatbot platforms and automated dialog management systems for query-response interactions in digital communication environments (customer service, e-commerce, technical support).

### Core Technical Function
Generating contextually appropriate responses in a dialog system by **dynamically interchanging role attributes** between user and chatbot entities to optimize response orientation and content specificity.

---

### Claim Analysis

| Claim | Type | Core Function | Key Features |
|-------|------|---------------|--------------|
| 1 | METHOD | Role interchange for chatbot-oriented response generation | LLM-based, API-operated, SYSTEM/USER/CHATBOT attributes, attribute swapping |
| 2 | METHOD | Prompt-driven role assignment | Initial instruction defines role exchange |
| 3 | METHOD | Hidden role exchange | Role swap transparent to user |
| 4 | METHOD | Enhanced concreteness via CHATBOT orientation | Identical prompt produces more concrete response under CHATBOT attribute |
| 5 | METHOD | Multi-suggestion generation | Prompt generates multiple response suggestions in chat history |
| 6 | METHOD | Stateful message transmission | Chat history re-transmitted for each message |
| 7 | METHOD | Multi-modal output | Response as audio/text/image file |
| 8 | METHOD | Multi-phase chat history management | Phases P1/P2/P3 with phase detection and message element supplementation |
| 9 | METHOD | Phase-based dialog flow | P1=request, P2=suggestions, P3=selection |
| 10 | METHOD | Content item detection | Product info, technical questions, comparisons, restart, duration monitoring |
| 11 | METHOD | Analytical phase assignment | Content elements mapped to phases |
| 12 | METHOD | Table-driven message insertion | Predefined message table inserted by prompt |
| 13 | METHOD | ML-based table updating | Machine learning updates message table |
| 14 | METHOD | Chat-history-based learning | Updates from previous chats and purchase decisions |

---

### Classification Strategy
**function-first**

Reasoning: This is a software method for dialog management that can be deployed across multiple unrelated industries (customer service, e-commerce, healthcare, technical support) without modification. The core novelty is in the dialog management mechanism (role interchange), not in a specific hardware apparatus or industry-specific system.

---

### Class Hypotheses (Soft)

| Class | Confidence | Reasoning |
|-------|-----------|-----------|
| G06F16 | 0.85 | Information retrieval and query formulation — core query-response function |
| G06F17 | 0.80 | Digital computing for specific applications — dialog management systems |
| G06N3 | 0.70 | Neural networks — LLM architecture and model-based generation |
| G06F40 | 0.65 | Natural language processing — text generation and understanding |
| H04L29 | 0.50 | API communication and network messaging — client-server chatbot operation |
| G10L15 | 0.40 | Speech/dialog systems — if voice message output is emphasized |

---

### Essential Terms (Section-Aware)

| Term | Importance | Source | Justification |
|------|-----------|--------|---------------|
| LLM | 10 | claims | Core technology, appears in independent claim |
| chatbot system | 10 | claims | Primary system type, independent claim |
| query and response | 10 | claims | Core function, independent claim |
| role interchange | 9 | claims | Key novel mechanism, independent claim |
| attribute | 9 | claims | Critical technical element, independent claim |
| message completion | 9 | claims | Core operation, independent claim |
| dialog | 9 | claims | System context, independent claim |
| chat history | 8 | claims | Important mechanism, Claim 8 |
| phase detection | 8 | claims | Multi-phase management, Claim 8 |
| prompt | 8 | claims | Control mechanism, Claim 2/5 |
| API | 7 | description | Implementation detail |
| GPT | 7 | description | Alternative LLM type, Claim 8 |
| natural language | 7 | description | Domain descriptor |
| conversational AI | 6 | abstract | Broad domain label |

---

### Negative Signals (Soft)

| Term | Confidence | Why Excluded |
|------|-----------|--------------|
| image processing | 0.7 | No image analysis in claims |
| computer vision | 0.7 | No visual recognition |
| fault tolerance | 0.8 | No redundancy or error recovery |
| hardware device | 0.6 | Pure software method |
| mechanical component | 0.9 | No physical parts |
| data compression | 0.5 | Not mentioned |
| encryption | 0.5 | Not mentioned |

### Negative Domains (Soft)

| Domain | Confidence | Why Excluded |
|--------|-----------|--------------|
| computer vision | 0.8 | No image/video processing |
| fault tolerance systems | 0.8 | No error recovery focus |
| mechanical engineering | 0.9 | No physical apparatus |
| signal processing | 0.6 | No audio signal analysis (only playback) |

---

### Per-Claim Preliminary CPC

| Claim | Type | CPC Classes | Reasoning |
|-------|------|-------------|-----------|
| 1 | independent | G06F16/3329, G06F17/30 | Core: query-response method with role interchange |
| 2 | dependent | G06F17/30 | Prompt-driven role assignment (dialog control) |
| 3 | dependent | G06F17/30 | Hidden role exchange (dialog management) |
| 4 | dependent | G06F17/30, G06N3/08 | Response concreteness via LLM orientation |
| 5 | dependent | G06F17/30 | Multi-suggestion generation (dialog control) |
| 6 | dependent | G06F17/30 | Chat history transmission (dialog state) |
| 7 | dependent | G10L15/24, G06F17/30 | Voice/text/image output (speech + dialog) |
| 8 | independent | G06F17/30, G06N3/08 | Multi-phase chat history, GPT/LLM generation |
| 9 | dependent | G06F17/30 | Phase-based dialog flow |
| 10 | dependent | G06F17/30 | Content item detection and mapping |
| 11 | dependent | G06F17/30 | Analytical phase assignment |
| 12 | dependent | G06F17/30 | Table-driven message insertion |
| 13 | dependent | G06N3/08, G06F40/00 | ML-based table updating |
| 14 | dependent | G06N3/08, G06F40/00 | Learning from chat history |

---

## PHASE 5: DYNAMIC DOMAIN VALIDATION (Simulated)

### Invention Domain (from Phase 1)
**Primary Domain:** Natural language processing / conversational AI / dialog management
**Secondary Domain:** Neural networks / machine learning (for LLM architecture)
**Context:** Software method for automated query-response via chatbot

---

### Candidate 1: G06F11/182 — Mutual exchange of output between redundant processing components

**Domain Analysis:**
- Candidate Domain: Fault tolerance / redundant computing systems
- Invention Domain: Dialog management / conversational AI
- **Assessment: COMPLETE MISMATCH**

**Reasoning:**
G06F11/182 is about fault-tolerant computing where redundant components exchange outputs to detect errors. This patent is about dialog systems where participants exchange messages. The word "exchange" appears in both, but the context is entirely different:
- G06F11: Hardware/software redundancy, error detection, fault recovery
- This patent: Natural language dialog, role attributes, response generation

**Decision: FAIL**
**Rejection Reason:** "COMPLETE MISMATCH. G06F11 covers fault tolerance and redundant processing. This invention is about natural language dialog management. The word 'exchange' refers to message passing in a conversation, not hardware redundancy."

---

### Candidate 2: G06F9/543 — User-generated data transfer (clipboards, DDE, OLE)

**Domain Analysis:**
- Candidate Domain: Operating system-level data exchange mechanisms
- Invention Domain: Dialog management / conversational AI
- **Assessment: COMPLETE MISMATCH**

**Reasoning:**
G06F9/543 covers OS-level data transfer like copy-paste (clipboard), Dynamic Data Exchange (DDE), and Object Linking and Embedding (OLE). This patent's "transfer" refers to message transmission in a chat dialog via API. These are completely different technical areas:
- G06F9/543: Local inter-process communication, OS clipboard
- This patent: Network API messaging, chat history transmission

**Decision: FAIL**
**Rejection Reason:** "COMPLETE MISMATCH. G06F9/543 is about OS-level data transfer (clipboard, DDE, OLE). This invention uses API-based message transmission in a chatbot system. The word 'transfer' refers to sending chat messages over a network, not local OS data exchange."

---

### Candidate 3: G06F3/015 — Input arrangements based on nervous system activity (EEG, EMG)

**Domain Analysis:**
- Candidate Domain: Biometric input devices / brain-computer interfaces
- Invention Domain: Dialog management / conversational AI
- **Assessment: COMPLETE MISMATCH**

**Reasoning:**
G06F3/015 covers input devices that detect brain waves (EEG), muscle activity (EMG), or electrodermal response. The word "user" in this patent refers to a dialog participant (T2), not a human providing biometric input. These domains are unrelated:
- G06F3/015: Physiological sensors, biometric input
- This patent: Software entities with USER attribute in a dialog

**Decision: FAIL**
**Rejection Reason:** "COMPLETE MISMATCH. G06F3/015 is about biometric input devices (EEG, EMG). This invention's 'user' is a software attribute in a dialog system, not a human providing physiological input."

---

### Candidate 4: G06F16/3329 — Query formulation with natural language

**Domain Analysis:**
- Candidate Domain: Information retrieval / natural language query processing
- Invention Domain: Dialog management / conversational AI
- **Assessment: MATCH**

**Reasoning:**
G06F16/3329 covers query formulation using natural language, which directly aligns with this patent's core function:
- The patent processes user queries (natural language input)
- It formulates responses based on query context
- It handles natural language dialog (multi-turn conversation)
- The LLM processes and generates natural language

**Function Alignment:** 0.95 (excellent match)
**Context Alignment:** 0.90 (NLP/IR domain matches chatbot domain)
**Visual Bias:** false
**Method/Apparatus:** Matches METHOD claims

**Decision: PASS**
**Confidence: HIGH**

---

### Candidate 5: G06F17/30 — Digital computing for specific applications (dialog management)

**Domain Analysis:**
- Candidate Domain: Digital computing / application-specific data processing
- Invention Domain: Dialog management / conversational AI
- **Assessment: MATCH**

**Reasoning:**
G06F17/30 covers digital computing methods specially adapted for specific applications, including dialog management systems. This is a broad but accurate classification for:
- Computer-implemented query-response methods
- Dialog state management (chat history, phases)
- Response generation and selection
- Multi-turn conversation management

**Function Alignment:** 0.90 (strong match)
**Context Alignment:** 0.85 (computing domain)
**Visual Bias:** false
**Method/Apparatus:** Matches METHOD claims

**Decision: PASS**
**Confidence: HIGH**

---

### Candidate 6: G06N3/08 — Neural networks / computer systems based on specific computational models

**Domain Analysis:**
- Candidate Domain: Neural networks / machine learning models
- Invention Domain: Dialog management using LLM (which is a neural network)
- **Assessment: RELATED**

**Reasoning:**
G06N3/08 covers neural network architectures and computational models. This patent uses an LLM (Large Language Model), which is a type of neural network. However:
- The patent's novelty is in dialog management (role interchange), not in the LLM architecture itself
- The LLM is a tool/component, not the primary invention
- This class is more about the model architecture than its application

**Function Alignment:** 0.60 (partial — LLM is tool, not novel contribution)
**Context Alignment:** 0.70 (AI/ML domain)
**Visual Bias:** false

**Decision: PASS**
**Confidence: MEDIUM**
**Note:** Good secondary class, but not primary.

---

### Candidate 7: H04L29/06 — Communication control for data transmission

**Domain Analysis:**
- Candidate Domain: Network communication / data transmission protocols
- Invention Domain: Dialog management via API
- **Assessment: RELATED**

**Reasoning:**
H04L29/06 covers communication control in digital networks. This patent mentions API operation, which involves network communication. However:
- The patent's focus is on dialog logic (role interchange), not network protocols
- API operation is an implementation detail, not the core invention
- This is a supporting/peripheral aspect

**Function Alignment:** 0.45 (weak — communication is vehicle, not destination)
**Context Alignment:** 0.60 (network domain is adjacent)

**Decision: PASS**
**Confidence: LOW**
**Note:** Relevant for system claims if any, but peripheral for method claims.

---

## PHASE 5 SUMMARY

| Candidate | Symbol | Decision | Confidence | Reason |
|-----------|--------|----------|------------|--------|
| G06F11/182 | Redundant processing | **FAIL** | — | Wrong domain (fault tolerance ≠ chatbot) |
| G06F9/543 | Data transfer (OS) | **FAIL** | — | Wrong domain (OS exchange ≠ API messaging) |
| G06F3/015 | Biometric input | **FAIL** | — | Wrong domain (physiological ≠ dialog) |
| G06F16/3329 | Query formulation | **PASS** | HIGH | Perfect match: NLP query-response |
| G06F17/30 | Dialog management | **PASS** | HIGH | Strong match: application-specific computing |
| G06N3/08 | Neural networks | **PASS** | MEDIUM | Related: LLM is neural network tool |
| H04L29/06 | Communication control | **PASS** | LOW | Related: API is network layer |

**Best Code:** G06F16/3329 (Query formulation with natural language)
**Secondary:** G06F17/30 (Dialog management)

---

## COMPARISON: Before vs After Fixes

### BEFORE (Original Method — Wrong)
```
G06F11/182 — Redundant processing (fault tolerance)
G06F9/543 — OS data transfer (clipboard/DDE)
G06F3/015 — Biometric input (EEG/EMG)
```
Why wrong: Literal keyword matching without domain context. Words like "exchange", "response", "system", "user" were matched to completely unrelated CPC domains.

### AFTER (Fixed Method — Correct)
```
G06F16/3329 — Query formulation with natural language ✓
G06F17/30 — Dialog management systems ✓
G06N3/08 — Neural networks (LLM architecture) ✓
```
Why correct: Dynamic domain validation ensures CPC classes match the actual technical domain (conversational AI / dialog management).

---

## Key Insight

The dynamic blacklist works because:
1. **Phase 1** extracts `system_context`: "Natural language processing and conversational AI systems"
2. **Phase 5** asks: "Does G06F11 (fault tolerance) match 'conversational AI'?" → **NO**
3. **Phase 5** asks: "Does G06F16 (information retrieval) match 'conversational AI'?" → **YES**

No hardcoded rules needed. The LLM understands the domain from the patent text and rejects mismatches.
