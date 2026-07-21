from typing import Final


class ResponseTemplates:
    """
    Store reusable response-format templates for LLM-generated answers.

    These templates define only the presentation structure.

    They do not:
    - retrieve financial documents
    - perform calculations
    - analyze risks
    - modify verified values
    - generate final answers directly
    """

    FINANCIAL_METRIC: Final[str] = """
Use exactly this structure when answering a financial-metric question:

## [Metric Name]

### Values

- **Current period:** [verified current value]
- **Previous period:** [verified previous value]

### Formula

[plain-text formula]

### Verified Result

**[verified result]**

### Interpretation

[One or two concise sentences explaining the business meaning.]

Formatting requirements:

- Use only verified values supplied by the application.
- Do not independently recalculate the result.
- Preserve the original units.
- Never use LaTeX.
- Do not add unrelated financial details.
""".strip()

    COMPANY_COMPARISON: Final[str] = """
Use exactly this structure when answering a company-comparison question:

## [Metric Name] Comparison

| Company | Fiscal Year | Value |
|---|---:|---:|
| [Company A] | [Year] | [Verified value] |
| [Company B] | [Year] | [Verified value] |

### Comparison Summary

- **Higher value:** [Company and verified value]
- **Lower value:** [Company and verified value]
- **Absolute difference:** [Verified difference]
- **Percentage difference:** [Verified percentage, when available]

### Conclusion

[Clearly identify which company reported the higher value.]

### Interpretation

[One or two concise sentences explaining what the comparison indicates.]

Formatting requirements:

- Use verified comparison-tool values exactly.
- Do not independently recalculate any value.
- Preserve the original units.
- Mention differing reporting periods when applicable.
- Do not add unrelated company information.
""".strip()

    RISK_ANALYSIS: Final[str] = """
Use exactly this structure when answering a risk-analysis question:

## Key Disclosed Risks

### 1. [Risk Category]

**Description**

[One concise description grounded only in the verified risk evidence.]

**Possible Business Impact**

[One concise explanation of how this risk could affect operations,
financial performance, compliance, reputation, customers, or strategy.]

---

### 2. [Risk Category]

**Description**

[One concise description grounded only in the verified risk evidence.]

**Possible Business Impact**

[One concise business-impact explanation.]

Continue the same structure for additional categories.

Formatting requirements:

- Include no more than 5 risk categories.
- Use only categories supported by verified risk-tool evidence.
- Keep each category concise.
- Use no more than 3 short sentences per category.
- Do not repeat similar risks.
- Do not invent severity rankings.
- Do not include unrelated revenue, products, achievements, or strategy.
- Do not add a general Business Impact section after all categories.
- Do not add a generic conclusion.
- Do not create a source list because verified sources are displayed
  separately by the application.
- End immediately after the final risk category.
""".strip()

    REPORT_SUMMARY: Final[str] = """
Use exactly this structure when summarizing a company or financial report:

## Business Overview

[A concise overview grounded in the supplied report context.]

## Financial Highlights

- [Supported financial fact]
- [Supported financial fact]
- [Supported financial fact]

## Strategic Developments

- [Supported development]
- [Supported development]

## Key Disclosed Risks

- **[Risk category]:** [Concise supported explanation]
- **[Risk category]:** [Concise supported explanation]

## Overall Assessment

[A short and balanced interpretation based only on the supplied context.]

Formatting requirements:

- Include only facts supported by the supplied report context.
- Omit a section when the supplied context does not support it.
- Do not invent values, events, strategies, or risks.
- Do not provide investment advice.
- Keep the summary concise and business-focused.
""".strip()

    GENERAL_QUESTION: Final[str] = """
Use this structure for a general financial question:

## Answer

[Answer the user's exact question directly.]

## Supporting Details

- [Relevant supported detail]
- [Relevant supported detail]

## Interpretation

[Provide a short interpretation only when it adds useful context.]

Formatting requirements:

- Use only the supplied financial-report context.
- Omit headings that are unnecessary for a very short answer.
- Clearly state when the supplied context is insufficient.
- Do not add unrelated analysis.
- Do not create a source list because verified sources are displayed
  separately by the application.
""".strip()

    INTENT_TEMPLATE_MAP: Final[dict[str, str]] = {
        "financial_metric": FINANCIAL_METRIC,
        "comparison": COMPANY_COMPARISON,
        "risk_analysis": RISK_ANALYSIS,
        "summary": REPORT_SUMMARY,
        "general_question": GENERAL_QUESTION,
    }

    @classmethod
    def get_template(
        cls,
        intent: str,
    ) -> str:
        """
        Return the response template associated with an intent.

        Unknown intents fall back to the general-question template.
        """

        normalized_intent = (
            intent.strip()
            if isinstance(intent, str)
            else "general_question"
        )

        return cls.INTENT_TEMPLATE_MAP.get(
            normalized_intent,
            cls.GENERAL_QUESTION,
        )


if __name__ == "__main__":
    intents = [
        "financial_metric",
        "comparison",
        "risk_analysis",
        "summary",
        "general_question",
        "unknown_intent",
    ]

    for intent in intents:
        print("\n" + "=" * 80)
        print(f"INTENT: {intent}\n")
        print(
            ResponseTemplates.get_template(
                intent
            )
        )