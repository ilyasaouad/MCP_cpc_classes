Function-First Classification
Classify by WHAT THE INVENTION DOES, not what it's made of.
Example - Chatbot Patent:
❌ System-first (WRONG):
- "It has a server, database, and UI" → Classify as "computing system" (G06F)
- "It uses neural networks" → Classify as "neural networks" (G06N)
✅ Function-first (CORRECT):
- "It processes natural language conversations" → G06F16/3329 (Natural language processing)
- "It generates automated responses" → G06F40/30 (Dialogue systems)
Why it matters:
A chatbot can be built with:
- Rules → Same function (conversing)
- Neural networks → Same function (conversing)  
- Hybrid → Same function (conversing)
The CPC system cares about the function, not the implementation.
System-First Classification (AVOID)
Classify by WHAT THE INVENTION LOOKS LIKE or IS MADE OF
Common mistakes:
- "It's a mobile app" → Classify by "mobile device" features
- "It uses blockchain" → Classify as blockchain (even if blockchain isn't the inventive part)
- "It's a robot" → Classify as mechanical robot (even if the invention is the AI algorithm)
The "Visual Bias" Problem
Original method had this flaw:
- Saw "exchange" → Assumed "data exchange" (system component)
- Saw "response" → Assumed "system response" (technical term)
- Result: Chatbot patent → G06F11 (Fault tolerance) ❌
Should have seen:
- "Exchange of messages between user and system" → Function: Conversational interaction
- "Generating responses based on context" → Function: Natural language generation
- Result: Chatbot patent → G06F16/33 (NLP) ✅
How to Apply in Your Code:
In prompts.py Phase 1:
Extract the invention's FUNCTIONAL OBJECT:
- What problem does it solve?
- What does it DO for the user?
- What is the end result?
NOT:
- What components does it have?
- What technology does it use?
- What does it look like?
In Phase 5 Validation:
For each candidate code, ask LLM:
"Does this code describe the FUNCTION of the invention?
Or does it only describe a COMPONENT/TECHNOLOGY used?"
Real Example from Your Project:
Patent: "A system for automated customer service using AI"
System-first extraction:
- System: AI chatbot
- Components: Server, database, NLP engine
- Technology: Machine learning
- Result: G06N (AI/ML - too broad) ❌
Function-first extraction:
- Technical Object: Automated customer service
- Function: Receiving customer queries → Generating appropriate responses
- End result: Resolving customer issues without human agent
- Result: G06F16/3329 (Dialog systems) + G06Q30/02 (Customer service) ✅
The Test:
Ask yourself: "If I changed the technology but kept the same function, would I still use this code?"
- Robot arm that picks fruit → A01D (harvesting) ✅
  - Same function if it's mechanical, hydraulic, or AI-powered
- Robot arm with new motor → B25J (robot manipulators) 
  - Here the invention IS the arm mechanism, not the fruit-picking
