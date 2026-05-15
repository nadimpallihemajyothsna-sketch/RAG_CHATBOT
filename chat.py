from transformers import pipeline

def get_answer(context, user_question):

    prompt = f"""
Answer the question based on the context below.

Context:
{context}

Question:
{user_question}
"""

    generator = pipeline(
        task="text-generation",
        model="distilgpt2"
    )

    result = generator(
        prompt,
        max_new_tokens=50
    )

    return result[0]["generated_text"]