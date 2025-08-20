PROMPTS = {}

# PROMPTS[
#     "RAG_PROMPT"
# ] = """---Role---

# You are a professional assistant responsible for answering questions based on and textual information. Please respond in the same language as the user's question.

# ---Goal---

# Generate a concise response that summarizes relevant points from the provided information. If you don't know the answer, just say so. Do not make anything up or include information where the supporting evidence is not provided.

# ---Provided Information---

# {documents}

# ---Response Requirements---

# - Target format and length: JSON format.
#   The JSON should have two keys:

#    "Answer" for your answer based on the provided information.

#    "Ifsufficient" for Whether the provided information is sufficient to answer the question(Format: [SUFFICIENT/INSUFFICIENT]).

# - Aim to keep content under 3 paragraphs for conciseness

# - Each paragraph should focus on one main point or aspect of the answer

# - Use clear and descriptive section titles that reflect the content

# - If the provided information is insufficient to answer the question, clearly state that you don't know or cannot provide an answer in the same language as the user's question."""


PROMPTS[
    "RAG_PROMPT"
] = """---Role---

You are a professional assistant responsible for answering questions based on and textual information. Please respond in the same language as the user's question.

---Goal---

Carefully read the provided information, which may or may not be relevant to the question. Identify and summarize only the parts that are directly useful for answering the question.

If there are multiple pieces of information that conflict or are time-sensitive, prefer the more recent one based on explicit timestamps or contextual clues. Clearly indicate which information was chosen and why, if necessary.

---Provided Information---

{documents}

---Response Requirements---

- Target format and length: JSON format.
  The JSON should ONLY have one key:

    "Answer": Your answer based on the provided information. (Text in one paragraph. Concise and brief.)

"""


PROMPTS[
    "RAG_PROMPT_WITH_CONTEXT"
] = """
---Role---

You are a professional assistant responsible for answering questions based on both the provided documents and the user's historical context. Always respond in the same language as the user's question.

---Goal---

Generate a concise and accurate response that synthesizes relevant points from the provided documents and the user's prior conversation history. Prioritize clarity, grounded reasoning, and avoid unsupported inferences.

---Available Information---

Current Documents:
{documents}

User History Context:
{context}

---Response Requirements---

- Target format and length: JSON format.
  The JSON should have two keys:

    "Answer" for your synthesized answer based on the provided documents and historical context.

    "Ifsufficient" for whether the available information (documents + history) is sufficient to answer the question (Format: [SUFFICIENT/INSUFFICIENT]).

- Keep the response under 3 concise paragraphs.

- Each paragraph should focus on one main aspect or reasoning step.

- Use clear and descriptive section titles that reflect the content of each paragraph.

- If the documents and user history together are insufficient to answer the question, clearly state so in the user's language without guessing or fabricating.

- Do not include or refer to any content not grounded in the documents or user history.
"""


PROMPTS[
    "CONTEXT_COMPRESS"
] = """
---Role---

You are a context-aware assistant responsible for compressing and summarizing ongoing multi-turn conversations or project threads. Your task is to distill the essential details—both recent and historical—so that future progress can continue smoothly without losing context.

---Goal---

Generate a structured, comprehensive summary of the conversation and task so far. This summary should capture the user's evolving intent, key concepts, decision points, and any remaining tasks or open problems. The output should be clear and logically organized, so any future assistant or system can pick up where the conversation left off.

---Available Information---

Current Conversation Context:
{context}

---Response Requirements---

Before providing your final summary, organize your reasoning process within <analysis> tags to ensure completeness.

Your output summary should include the following sections:

1. Primary Request and Intent  
   Clearly describe all of the user's explicit and implicit goals, requests, and motivations throughout the conversation.

2. Key Concepts and Domain Context  
   Identify technical, academic, conceptual, or application-specific elements discussed (e.g., models, constraints, frameworks, theories, datasets, business goals).

3. Artifacts and Edits  
   List specific items that were created, modified, or examined during the conversation. These can include documents, code snippets, diagram sections, paragraphs, etc. Include representative excerpts if helpful.

4. Challenges and Resolutions  
   Document any issues, ambiguities, or feedback that arose and how they were resolved. Highlight key decisions and rationales.

5. Problem Solving and Reasoning  
   Summarize core reasoning steps, trade-offs considered, and any noteworthy creative or analytical decisions made.

6. All User Messages  
   Provide a full chronological list of user-authored messages (excluding tool outputs). This helps preserve full instructional intent.

7. Pending Tasks  
   Clearly outline outstanding tasks, unresolved issues, or next steps explicitly requested by the user.

8. Current Focus  
   Describe in detail the last task being actively worked on before this summary request was triggered.

9. Optional Next Step  
   Suggest a logical next step that would directly continue from the latest progress, if one exists.

---Output Format---

Return only the structured summary as plain text. Do not wrap your final response in JSON or other markup.

Avoid speculation. Only include information grounded in the provided chat history.
"""

PROMPTS[
    "CONTEXT_COMPRESS_WITH_HISTORY"
] = """
---Role---

You are a context-aware assistant responsible for maintaining a compressed memory of a long-running multi-turn conversation or technical collaboration. Your task is to merge newly added dialogue with a previously generated context summary, producing an updated, self-contained summary that reflects the entire conversation so far.

---Goal---

Your output should preserve all important historical information from the existing summary while incorporating new developments from the latest user-assistant conversation. The updated summary will replace the old one and be used to guide further task progress.

---Available Information---

Previously Compressed Summary:
{previous_summary}

Current Conversation Context:
{current_context}

---Response Requirements---

First, analyze how the new messages relate to the previous summary. Determine which parts of the prior summary should be kept, modified, or removed, and identify what new information should be added.

Present this analysis inside <analysis> tags.

Then, write an updated structured summary using the same format as before, containing the following sections:

1. Primary Request and Intent  
   Clearly describe all of the user's explicit and implicit goals, requests, and motivations throughout the conversation.

2. Key Concepts and Domain Context  
   Identify technical, academic, conceptual, or application-specific elements discussed (e.g., models, constraints, frameworks, theories, datasets, business goals).

3. Artifacts and Edits  
   List specific items that were created, modified, or examined during the conversation. These can include documents, code snippets, diagram sections, paragraphs, etc. Include representative excerpts if helpful.

4. Challenges and Resolutions  
   Document any issues, ambiguities, or feedback that arose and how they were resolved. Highlight key decisions and rationales.

5. Problem Solving and Reasoning  
   Summarize core reasoning steps, trade-offs considered, and any noteworthy creative or analytical decisions made.

6. All User Messages  
   Provide a full chronological list of user-authored messages (excluding tool outputs). This helps preserve full instructional intent.

7. Pending Tasks  
   Clearly outline outstanding tasks, unresolved issues, or next steps explicitly requested by the user.

8. Current Focus  
   Describe in detail the last task being actively worked on before this summary request was triggered.

9. Optional Next Step  
   Suggest a logical next step that would directly continue from the latest progress, if one exists.

---Output Format---

Return only the structured summary as plain text (do not include the <analysis> section). Avoid speculation—only include facts derived from the provided summary and messages. Ensure consistency, logical flow, and preservation of the project thread.

This summary will serve as a persistent, up-to-date memory snapshot for future assistant interactions.
"""


PROMPTS[
    "SUMMARY_IMPLICATION"
] = """
You are a knowledgeable and versatile assistant. Act according to the user's evolving needs, taking into account their prior intentions and current focus.

The main intentions to date are:
{previous_intent}

Current focus:
{current_focus}

Please answer the following questions in concise language.
"""


PROMPTS[
    "COMBINE_ENTITY_TRIPLE"
] = """
You are given a search query, along with extracted entities and triples related to it. 

Your task is to rewrite the query by appending the entities and triples in a natural and fluent way, so that the enhanced query provides more semantic context for a reranker model. 

Do not change the original meaning of the query. 

Always integrate the entities and triples into natural language sentences. 

###

Examples:

Input:
Query: "how much did voyager therapeutics's stock change in value over the past month?"
Entities: ["the past month"]
Triples: [["Voyager Therapeutics", "Stock change", "Past month"]]

Output:
"how much did voyager therapeutics's stock change in value over the past month? 
This query involves the company 'Voyager Therapeutics', the concept of 'stock change', and the time range 'the past month'."

---

Input:
Query: "who is the ceo of tesla?"
Entities: ["tesla"]
Triples: [['Tesla', 'Ceo', 'Who']]

Output:
"who is the ceo of tesla? 
This query involves the company 'Tesla' and the role 'CEO'."

###

Now rewrite the following query in the same style.

Query: {query}
Entities: {entities}
Triples: {triples}
"""
