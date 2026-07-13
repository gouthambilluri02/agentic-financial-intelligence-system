import ollama


class FinancialAgent:

    def process_question(self, question: str):

        response = ollama.chat(
            model="qwen2.5:7b",
            messages=[
                {
                    "role": "system",
                    "content": """
You are an expert Financial Intelligence AI.

Your responsibilities are:
- Explain financial concepts clearly.
- Analyze company financial statements.
- Compare stocks and companies.
- Explain investment risks.
- Summarize earnings reports.
- Never make up financial facts.
- If information is unavailable, clearly say you don't know.
- Answer professionally and concisely.
"""
                },
                {
                    "role": "user",
                    "content": question
                }
            ]
        )

        return response["message"]["content"]