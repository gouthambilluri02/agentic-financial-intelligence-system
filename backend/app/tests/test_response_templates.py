"""
Tests for ResponseTemplates.

Covered behavior:

- Template constants
- Intent-to-template mapping
- get_template()
- Unknown and invalid intent fallback behavior
"""

from __future__ import annotations

from typing import Any

import pytest

from backend.app.services.response_templates import ResponseTemplates


class TestTemplateConstants:
    """Tests for the reusable response-template constants."""

    def test_financial_metric_template_is_non_empty_string(self) -> None:
        assert isinstance(ResponseTemplates.FINANCIAL_METRIC, str)
        assert ResponseTemplates.FINANCIAL_METRIC.strip()

    def test_company_comparison_template_is_non_empty_string(
        self,
    ) -> None:
        assert isinstance(ResponseTemplates.COMPANY_COMPARISON, str)
        assert ResponseTemplates.COMPANY_COMPARISON.strip()

    def test_risk_analysis_template_is_non_empty_string(self) -> None:
        assert isinstance(ResponseTemplates.RISK_ANALYSIS, str)
        assert ResponseTemplates.RISK_ANALYSIS.strip()

    def test_report_summary_template_is_non_empty_string(self) -> None:
        assert isinstance(ResponseTemplates.REPORT_SUMMARY, str)
        assert ResponseTemplates.REPORT_SUMMARY.strip()

    def test_general_question_template_is_non_empty_string(
        self,
    ) -> None:
        assert isinstance(ResponseTemplates.GENERAL_QUESTION, str)
        assert ResponseTemplates.GENERAL_QUESTION.strip()

    def test_templates_are_stripped(self) -> None:
        templates = [
            ResponseTemplates.FINANCIAL_METRIC,
            ResponseTemplates.COMPANY_COMPARISON,
            ResponseTemplates.RISK_ANALYSIS,
            ResponseTemplates.REPORT_SUMMARY,
            ResponseTemplates.GENERAL_QUESTION,
        ]

        for template in templates:
            assert template == template.strip()

    def test_templates_are_unique(self) -> None:
        templates = {
            ResponseTemplates.FINANCIAL_METRIC,
            ResponseTemplates.COMPANY_COMPARISON,
            ResponseTemplates.RISK_ANALYSIS,
            ResponseTemplates.REPORT_SUMMARY,
            ResponseTemplates.GENERAL_QUESTION,
        }

        assert len(templates) == 5


class TestFinancialMetricTemplate:
    """Tests for the financial-metric response template."""

    def test_contains_metric_heading(self) -> None:
        assert "## [Metric Name]" in ResponseTemplates.FINANCIAL_METRIC

    def test_contains_values_section(self) -> None:
        assert "### Values" in ResponseTemplates.FINANCIAL_METRIC

    def test_contains_current_and_previous_period_placeholders(
        self,
    ) -> None:
        template = ResponseTemplates.FINANCIAL_METRIC

        assert "**Current period:**" in template
        assert "**Previous period:**" in template

    def test_contains_formula_section(self) -> None:
        assert "### Formula" in ResponseTemplates.FINANCIAL_METRIC
        assert "[plain-text formula]" in ResponseTemplates.FINANCIAL_METRIC

    def test_contains_verified_result_section(self) -> None:
        template = ResponseTemplates.FINANCIAL_METRIC

        assert "### Verified Result" in template
        assert "**[verified result]**" in template

    def test_contains_interpretation_section(self) -> None:
        assert "### Interpretation" in ResponseTemplates.FINANCIAL_METRIC

    def test_requires_verified_values(self) -> None:
        assert (
            "Use only verified values supplied by the application."
            in ResponseTemplates.FINANCIAL_METRIC
        )

    def test_forbids_independent_recalculation(self) -> None:
        assert (
            "Do not independently recalculate the result."
            in ResponseTemplates.FINANCIAL_METRIC
        )

    def test_requires_preserving_units(self) -> None:
        assert (
            "Preserve the original units."
            in ResponseTemplates.FINANCIAL_METRIC
        )

    def test_forbids_latex(self) -> None:
        assert "Never use LaTeX." in ResponseTemplates.FINANCIAL_METRIC

    def test_forbids_unrelated_financial_details(self) -> None:
        assert (
            "Do not add unrelated financial details."
            in ResponseTemplates.FINANCIAL_METRIC
        )


class TestCompanyComparisonTemplate:
    """Tests for the company-comparison response template."""

    def test_contains_comparison_heading(self) -> None:
        assert (
            "## [Metric Name] Comparison"
            in ResponseTemplates.COMPANY_COMPARISON
        )

    def test_contains_comparison_table(self) -> None:
        template = ResponseTemplates.COMPANY_COMPARISON

        assert "| Company | Fiscal Year | Value |" in template
        assert "|---|---:|---:|" in template
        assert "| [Company A] | [Year] | [Verified value] |" in template
        assert "| [Company B] | [Year] | [Verified value] |" in template

    def test_contains_summary_section(self) -> None:
        assert (
            "### Comparison Summary"
            in ResponseTemplates.COMPANY_COMPARISON
        )

    def test_contains_higher_and_lower_value_placeholders(
        self,
    ) -> None:
        template = ResponseTemplates.COMPANY_COMPARISON

        assert "**Higher value:**" in template
        assert "**Lower value:**" in template

    def test_contains_difference_placeholders(self) -> None:
        template = ResponseTemplates.COMPANY_COMPARISON

        assert "**Absolute difference:**" in template
        assert "**Percentage difference:**" in template

    def test_contains_conclusion_section(self) -> None:
        assert "### Conclusion" in ResponseTemplates.COMPANY_COMPARISON

    def test_contains_interpretation_section(self) -> None:
        assert "### Interpretation" in ResponseTemplates.COMPANY_COMPARISON

    def test_requires_verified_tool_values(self) -> None:
        assert (
            "Use verified comparison-tool values exactly."
            in ResponseTemplates.COMPANY_COMPARISON
        )

    def test_forbids_recalculation(self) -> None:
        assert (
            "Do not independently recalculate any value."
            in ResponseTemplates.COMPANY_COMPARISON
        )

    def test_requires_reporting_period_warning(self) -> None:
        assert (
            "Mention differing reporting periods when applicable."
            in ResponseTemplates.COMPANY_COMPARISON
        )

    def test_forbids_unrelated_company_information(self) -> None:
        assert (
            "Do not add unrelated company information."
            in ResponseTemplates.COMPANY_COMPARISON
        )


class TestRiskAnalysisTemplate:
    """Tests for the risk-analysis response template."""

    def test_contains_risk_heading(self) -> None:
        assert "## Key Disclosed Risks" in ResponseTemplates.RISK_ANALYSIS

    def test_contains_numbered_risk_sections(self) -> None:
        template = ResponseTemplates.RISK_ANALYSIS

        assert "### 1." in template
        assert "### 2." in template

    def test_contains_category_placeholder(self) -> None:
        assert "[Risk Category]" in ResponseTemplates.RISK_ANALYSIS

    def test_contains_description_label(self) -> None:
        assert "**Description**" in ResponseTemplates.RISK_ANALYSIS

    def test_contains_business_impact_label(self) -> None:
        assert "**Possible Business Impact**" in ResponseTemplates.RISK_ANALYSIS

    def test_limits_risk_categories(self) -> None:
        assert (
            "Include no more than 5 risk categories."
            in ResponseTemplates.RISK_ANALYSIS
        )

    def test_requires_verified_risk_evidence(self) -> None:
        assert (
            "Use only categories supported by verified risk-tool evidence."
            in ResponseTemplates.RISK_ANALYSIS
        )

    def test_limits_sentences_per_category(self) -> None:
        assert (
            "Use no more than 3 short sentences per category."
            in ResponseTemplates.RISK_ANALYSIS
        )

    def test_forbids_repeated_risks(self) -> None:
        assert (
            "Do not repeat similar risks."
            in ResponseTemplates.RISK_ANALYSIS
        )

    def test_forbids_invented_severity_rankings(self) -> None:
        assert (
            "Do not invent severity rankings."
            in ResponseTemplates.RISK_ANALYSIS
        )

    def test_forbids_unrelated_topics(self) -> None:
        assert (
            "Do not include unrelated revenue, products, achievements, or strategy."
            in ResponseTemplates.RISK_ANALYSIS
        )

    def test_forbids_generic_conclusion(self) -> None:
        assert (
            "Do not add a generic conclusion."
            in ResponseTemplates.RISK_ANALYSIS
        )

    def test_forbids_source_list(self) -> None:
        normalized_template = " ".join(
            ResponseTemplates.RISK_ANALYSIS.split()
        )

        assert (
            "Do not create a source list because verified sources "
            "are displayed separately by the application."
            in normalized_template
        )

    def test_requires_immediate_ending(self) -> None:
        assert (
            "End immediately after the final risk category."
            in ResponseTemplates.RISK_ANALYSIS
        )


class TestReportSummaryTemplate:
    """Tests for the report-summary response template."""

    def test_contains_business_overview_section(self) -> None:
        assert "## Business Overview" in ResponseTemplates.REPORT_SUMMARY

    def test_contains_financial_highlights_section(self) -> None:
        assert "## Financial Highlights" in ResponseTemplates.REPORT_SUMMARY

    def test_contains_strategic_developments_section(self) -> None:
        assert "## Strategic Developments" in ResponseTemplates.REPORT_SUMMARY

    def test_contains_key_risks_section(self) -> None:
        assert "## Key Disclosed Risks" in ResponseTemplates.REPORT_SUMMARY

    def test_contains_overall_assessment_section(self) -> None:
        assert "## Overall Assessment" in ResponseTemplates.REPORT_SUMMARY

    def test_requires_context_supported_facts(self) -> None:
        assert (
            "Include only facts supported by the supplied report context."
            in ResponseTemplates.REPORT_SUMMARY
        )

    def test_allows_unsupported_sections_to_be_omitted(self) -> None:
        assert (
            "Omit a section when the supplied context does not support it."
            in ResponseTemplates.REPORT_SUMMARY
        )

    def test_forbids_invented_content(self) -> None:
        assert (
            "Do not invent values, events, strategies, or risks."
            in ResponseTemplates.REPORT_SUMMARY
        )

    def test_forbids_investment_advice(self) -> None:
        assert (
            "Do not provide investment advice."
            in ResponseTemplates.REPORT_SUMMARY
        )

    def test_requires_concise_business_focus(self) -> None:
        assert (
            "Keep the summary concise and business-focused."
            in ResponseTemplates.REPORT_SUMMARY
        )


class TestGeneralQuestionTemplate:
    """Tests for the general-question response template."""

    def test_contains_answer_section(self) -> None:
        assert "## Answer" in ResponseTemplates.GENERAL_QUESTION

    def test_contains_supporting_details_section(self) -> None:
        assert "## Supporting Details" in ResponseTemplates.GENERAL_QUESTION

    def test_contains_interpretation_section(self) -> None:
        assert "## Interpretation" in ResponseTemplates.GENERAL_QUESTION

    def test_requires_supplied_context_only(self) -> None:
        assert (
            "Use only the supplied financial-report context."
            in ResponseTemplates.GENERAL_QUESTION
        )

    def test_allows_unnecessary_headings_to_be_omitted(self) -> None:
        assert (
            "Omit headings that are unnecessary for a very short answer."
            in ResponseTemplates.GENERAL_QUESTION
        )

    def test_requires_insufficient_context_disclosure(self) -> None:
        assert (
            "Clearly state when the supplied context is insufficient."
            in ResponseTemplates.GENERAL_QUESTION
        )

    def test_forbids_unrelated_analysis(self) -> None:
        assert (
            "Do not add unrelated analysis."
            in ResponseTemplates.GENERAL_QUESTION
        )

    def test_forbids_source_list(self) -> None:
        normalized_template = " ".join(
            ResponseTemplates.GENERAL_QUESTION.split()
        )

        assert (
            "Do not create a source list because verified sources "
            "are displayed separately by the application."
            in normalized_template
        )


class TestIntentTemplateMap:
    """Tests for INTENT_TEMPLATE_MAP."""

    def test_map_is_dictionary(self) -> None:
        assert isinstance(ResponseTemplates.INTENT_TEMPLATE_MAP, dict)

    def test_map_contains_expected_intents(self) -> None:
        assert set(ResponseTemplates.INTENT_TEMPLATE_MAP) == {
            "financial_metric",
            "comparison",
            "risk_analysis",
            "summary",
            "general_question",
        }

    def test_financial_metric_mapping(self) -> None:
        assert (
            ResponseTemplates.INTENT_TEMPLATE_MAP["financial_metric"]
            == ResponseTemplates.FINANCIAL_METRIC
        )

    def test_comparison_mapping(self) -> None:
        assert (
            ResponseTemplates.INTENT_TEMPLATE_MAP["comparison"]
            == ResponseTemplates.COMPANY_COMPARISON
        )

    def test_risk_analysis_mapping(self) -> None:
        assert (
            ResponseTemplates.INTENT_TEMPLATE_MAP["risk_analysis"]
            == ResponseTemplates.RISK_ANALYSIS
        )

    def test_summary_mapping(self) -> None:
        assert (
            ResponseTemplates.INTENT_TEMPLATE_MAP["summary"]
            == ResponseTemplates.REPORT_SUMMARY
        )

    def test_general_question_mapping(self) -> None:
        assert (
            ResponseTemplates.INTENT_TEMPLATE_MAP["general_question"]
            == ResponseTemplates.GENERAL_QUESTION
        )

    def test_map_values_are_non_empty_strings(self) -> None:
        for template in ResponseTemplates.INTENT_TEMPLATE_MAP.values():
            assert isinstance(template, str)
            assert template.strip()


class TestGetTemplate:
    """Tests for ResponseTemplates.get_template()."""

    @pytest.mark.parametrize(
        ("intent", "expected"),
        [
            ("financial_metric", ResponseTemplates.FINANCIAL_METRIC),
            ("comparison", ResponseTemplates.COMPANY_COMPARISON),
            ("risk_analysis", ResponseTemplates.RISK_ANALYSIS),
            ("summary", ResponseTemplates.REPORT_SUMMARY),
            ("general_question", ResponseTemplates.GENERAL_QUESTION),
        ],
    )
    def test_known_intent_returns_expected_template(
        self,
        intent: str,
        expected: str,
    ) -> None:
        result = ResponseTemplates.get_template(intent)

        assert result == expected

    @pytest.mark.parametrize(
        ("intent", "expected"),
        [
            (" financial_metric ", ResponseTemplates.FINANCIAL_METRIC),
            ("\tcomparison\t", ResponseTemplates.COMPANY_COMPARISON),
            ("\nrisk_analysis\n", ResponseTemplates.RISK_ANALYSIS),
            ("  summary  ", ResponseTemplates.REPORT_SUMMARY),
            (
                " general_question ",
                ResponseTemplates.GENERAL_QUESTION,
            ),
        ],
    )
    def test_known_intent_is_trimmed(
        self,
        intent: str,
        expected: str,
    ) -> None:
        result = ResponseTemplates.get_template(intent)

        assert result == expected

    @pytest.mark.parametrize(
        "intent",
        [
            "unknown",
            "financial",
            "metric",
            "compare",
            "risk",
            "SUMMARY",
            "Financial_Metric",
        ],
    )
    def test_unknown_string_intent_returns_general_template(
        self,
        intent: str,
    ) -> None:
        result = ResponseTemplates.get_template(intent)

        assert result == ResponseTemplates.GENERAL_QUESTION

    @pytest.mark.parametrize(
        "intent",
        [
            None,
            123,
            4.5,
            True,
            [],
            {},
            (),
            object(),
        ],
    )
    def test_non_string_intent_returns_general_template(
        self,
        intent: Any,
    ) -> None:
        result = ResponseTemplates.get_template(intent)  # type: ignore[arg-type]

        assert result == ResponseTemplates.GENERAL_QUESTION

    @pytest.mark.parametrize(
        "intent",
        [
            "",
            " ",
            "   ",
            "\n",
            "\t",
            " \n\t ",
        ],
    )
    def test_blank_intent_returns_general_template(
        self,
        intent: str,
    ) -> None:
        result = ResponseTemplates.get_template(intent)

        assert result == ResponseTemplates.GENERAL_QUESTION

    def test_result_is_exact_map_value_for_known_intent(self) -> None:
        for intent, expected in (
            ResponseTemplates.INTENT_TEMPLATE_MAP.items()
        ):
            result = ResponseTemplates.get_template(intent)

            assert result is expected

    def test_method_is_available_on_class(self) -> None:
        result = ResponseTemplates.get_template("comparison")

        assert result == ResponseTemplates.COMPANY_COMPARISON

    def test_method_is_available_on_instance(self) -> None:
        instance = ResponseTemplates()

        result = instance.get_template("summary")

        assert result == ResponseTemplates.REPORT_SUMMARY
