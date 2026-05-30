# Step 1: Prepare responses

# Step 2: Extract answer_tokens

USER_INPUT_TEMPLATE now sends indexed tokens, like (0, "The"), (1, " answer"), etc.
EXAMPLE_MESSAGES now demonstrates returning JSON indices, like [21, 22, 23, 24, 25, 26].
AnswerTokenExtractor.extract_via_llm() now validates indices instead of checking fragile string equality with all(t in tokens for t in extracted).
The function still returns actual token strings for answer_tokens, but those strings are selected directly from the original tokenizer output using the returned indices.
I also added handling for markdown-wrapped JSON replies from the LLM.