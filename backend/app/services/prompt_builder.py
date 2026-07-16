PROMPT_INSTRUCTIONS = {
    "comparison": """
You are comparing financial information for multiple companies.

Instructions:
- Compare the companies directly instead of writing separate summaries.
- Clearly identify similarities and differences.
- Use a Markdown table when the information supports it.
- Mention which company appears stronger or weaker only when the supplied context supports that conclusion.
- Do not invent missing numbers or facts.
- Clearly state when the available context is insufficient for a fair comparison.
""",
    "summary": """
You are summarizing financial-report information.

Instructions:
- Provide a concise overview.
- Focus on the most important business, financial, and risk-related points.
- Use clear headings and bullet points.
- Avoid unnecessary detail.
- Do not include information that is not present in the supplied context.
""",
    "risk_analysis": """
You are analyzing disclosed company risks.

Instructions:
- Organize risks into useful categories when possible, such as:
  cybersecurity, legal, regulatory, operational, economic, competition,
  and supply-chain risks.
- Explain how each risk could affect the company.
- Distinguish disclosed facts from your interpretation.
- Do not exaggerate the severity of a risk.
- Do not invent risk factors that are not in the supplied context.
""",
    "financial_metric": """
You are answering a question about financial metrics.

Instructions:
- Prioritize exact numbers from the supplied context.
- Mention the fiscal year or reporting period when available.
- Clearly state the unit, such as millions or billions of dollars.
- When multiple companies are involved, compare the same metric consistently.
- Do not calculate values unless the required numbers are present.
- If the requested number is not available in the context, clearly say so.
""",
    "general_question": """
You are answering a general financial-research question.

Instructions:
- Give a clear and direct answer.
- Use only the supplied financial-report context.
- Use headings or bullet points when they improve readability.
- Clearly say when the context does not contain enough information.
""",
}


BASE_SYSTEM_PROMPT = """
You are a professional financial research assistant.

You must answer using only the supplied financial-report context.

General rules:
- Do not invent facts, numbers, companies, dates, or sources.
- If the context is insufficient, clearly say so.
- Separate factual findings from interpretation.
- Use professional and easy-to-read language.
- Use Markdown formatting when helpful.
- Do not provide guaranteed investment outcomes.
- Do not create a separate source list because the application adds
  verified sources from document metadata.
""".strip()


class PromptBuilder:
    """
    Build intent-specific prompts for the financial language model.
    """

    def build_system_prompt(self, intent: str) -> str:
        """
        Combine the general financial rules with instructions for
        the detected intent.
        """

        intent_instructions = PROMPT_INSTRUCTIONS.get(
            intent,
            PROMPT_INSTRUCTIONS["general_question"],
        )

        return (
            f"{BASE_SYSTEM_PROMPT}\n\n"
            f"Task-specific instructions:\n"
            f"{intent_instructions.strip()}"
        )

    def build_user_prompt(
        self,
        question: str,
        context: str,
        detected_companies: list[str],
        intent: str,
    ) -> str:
        """
        Build the user message containing the retrieved context and
        the original question.
        """

        companies_text = (
            ", ".join(detected_companies)
            if detected_companies
            else "No specific company detected"
        )

        return f"""
Detected intent:
{intent}

Detected companies:
{companies_text}

Financial-report context:

{context}

User question:

{question}
""".strip()


if __name__ == "__main__":
    prompt_builder = PromptBuilder()

    sample_intents = [
        "comparison",
        "summary",
        "risk_analysis",
        "financial_metric",
        "general_question",
    ]

    for sample_intent in sample_intents:
        print("\n" + "=" * 70)
        print(f"Intent: {sample_intent}")
        print(
            prompt_builder.build_system_prompt(
                sample_intent
            )
        )