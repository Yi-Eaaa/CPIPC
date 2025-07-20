PROMPTS = {}
PROMPTS[
    "INTENTION_PROMPT"
] = """---Role---

You are a psychologist who is good at summarizing and generalizing. You specialize in discovering the intentions in people's conversations and summarizing them.

---Goal---

Given the history of user queries and retrieved documents from a RAG system, please infer the user's overall search intention. Provide a few keywords that represent the user's intention.

---Instructions---

- Output your result in JSON format.
- The JSON should have two keys:
  - "User Intention Inference" for analyzing and summarizing user intentions.
  - "User Intention Keywords" for extracting the keywords from your analysis and summary.

######################
-Examples-
######################
Example 1:

History:
1.Query: "What are the symptoms of the flu?"
  Document: "The flu is a respiratory illness caused by influenza viruses. Symptoms include fever, cough, sore throat, body aches, and fatigue."
2.Query: "How long does the flu last?"
  Document: "Most people recover from the flu in 3 to 7 days."
################
Output:
{{
    "User Intention Inference": "The user is looking for information about the flu, including its symptoms and duration. Based on these Information, maybe the user have got the flu. So the main query intention is around the information about flu, and how to treat it."
    "User Intention Keywords": ["flu symptoms", "flu duration", "recovery", "treatment means"]
}}
#############################
Example 2:

History:
1.Query: "What is the capital of France?"
  Document: "Paris is the capital of France."
2.Query: "What are some tourist attractions in Paris?"
  Document: "The Eiffel Tower, the Louvre Museum, and the Arc de Triomphe are popular tourist attractions in Paris."
################
Output:
{{
    "User Intention Inference": ["The user is looking for information about the France, including its capital and tourist attractions around its captial. Based on these Information, maybe the user are making a travel plan to France. So the main query intention is around the information related to traveling in France, including its capital and tourist attractions."]
    "User Intention Keywords": ["France", "capital", "Paris", "tourist attractions", "travel plan"]
}}
#############################
Example 3:

History:
1.Query: "What are the benefits of eating apples?"
  Document: "Apples are a good source of fiber and vitamin C. They may also help to protect against heart disease and cancer."
2.Query: "How many calories are in an apple?"
  Document: "A medium-sized apple contains about 95 calories."
################
Output:
{{
    "User Intention Inference": ["The user is looking for information about apples, including their nutritional benefits and calorie content. Based on this information, maybe the user is health-conscious or trying to understand the nutritional value of apples for dietary purposes. So the main query intention is around the nutritional information of apples and their health benefits."]
    "User Intention Keywords": ["apples", "nutrition", "calories", "health benefits"]
}}
#############################

######################
-Real Data-
######################
History:
{history}
################
The `Output` should be human text, not unicode characters. Keep the same language as `History`.
Output:
"""

PROMPTS[
    "RAG_PROMPT"
] = """---Role---

You are a professional assistant responsible for answering questions based on and textual information. Please respond in the same language as the user's question.

---Goal---

Generate a concise response that summarizes relevant points from the provided information. If you don't know the answer, just say so. Do not make anything up or include information where the supporting evidence is not provided.

---Provided Information---

{documents}

---Response Requirements---

- Target format and length: JSON format.
  The JSON should have two keys:
    "Answer" for your answer based on the provided information.
    "Ifsufficient" for Whether the provided information is sufficient to answer the question(Format: [SUFFICIENT/INSUFFICIENT]).
- Aim to keep content under 3 paragraphs for conciseness
- Each paragraph should focus on one main point or aspect of the answer
- Use clear and descriptive section titles that reflect the content
- If the provided information is insufficient to answer the question, clearly state that you don't know or cannot provide an answer in the same language as the user's question."""
