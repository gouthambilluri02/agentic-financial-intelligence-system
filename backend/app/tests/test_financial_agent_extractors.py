"""
Tests for FinancialAgent extractor helper methods.

Covered methods:

- _extract_calculation_result
- _extract_comparison_result
- _extract_calculated_comparison_result
- _extract_calculated_comparison_reasoning
- _extract_risk_result
"""

from __future__ import annotations

from typing import Any

import pytest

from backend.app.agents.financial_agent import FinancialAgent


# ---------------------------------------------------------------------------
# Calculation result extractor
# ---------------------------------------------------------------------------


class TestExtractCalculationResult:
    """Tests for _extract_calculation_result()."""

    @pytest.mark.parametrize(
        "execution_result",
        [
            {},
            {"tool_outputs": None},
            {"tool_outputs": "invalid"},
            {"tool_outputs": 123},
            {"tool_outputs": {}},
            {"tool_outputs": ()},
        ],
    )
    def test_invalid_tool_outputs_returns_none(
        self,
        execution_result: dict[str, Any],
    ) -> None:
        result = FinancialAgent._extract_calculation_result(
            execution_result
        )

        assert result is None

    def test_empty_tool_outputs_returns_none(self) -> None:
        execution_result = {
            "tool_outputs": [],
        }

        result = FinancialAgent._extract_calculation_result(
            execution_result
        )

        assert result is None

    @pytest.mark.parametrize(
        "invalid_step",
        [
            None,
            "invalid",
            100,
            4.5,
            [],
            (),
        ],
    )
    def test_invalid_step_is_ignored(
        self,
        invalid_step: Any,
    ) -> None:
        execution_result = {
            "tool_outputs": [
                invalid_step,
            ],
        }

        result = FinancialAgent._extract_calculation_result(
            execution_result
        )

        assert result is None

    def test_wrong_tool_name_is_ignored(self) -> None:
        execution_result = {
            "tool_outputs": [
                {
                    "tool": "company_comparison",
                    "output": {
                        "calculation": {
                            "value": 10,
                        }
                    },
                }
            ],
        }

        result = FinancialAgent._extract_calculation_result(
            execution_result
        )

        assert result is None

    def test_missing_output_returns_none(self) -> None:
        execution_result = {
            "tool_outputs": [
                {
                    "tool": "financial_calculator",
                }
            ],
        }

        result = FinancialAgent._extract_calculation_result(
            execution_result
        )

        assert result is None

    @pytest.mark.parametrize(
        "invalid_output",
        [
            None,
            "invalid",
            100,
            2.5,
            [],
            (),
        ],
    )
    def test_invalid_output_is_ignored(
        self,
        invalid_output: Any,
    ) -> None:
        execution_result = {
            "tool_outputs": [
                {
                    "tool": "financial_calculator",
                    "output": invalid_output,
                }
            ],
        }

        result = FinancialAgent._extract_calculation_result(
            execution_result
        )

        assert result is None

    def test_missing_calculation_returns_none(self) -> None:
        execution_result = {
            "tool_outputs": [
                {
                    "tool": "financial_calculator",
                    "output": {},
                }
            ],
        }

        result = FinancialAgent._extract_calculation_result(
            execution_result
        )

        assert result is None

    @pytest.mark.parametrize(
        "invalid_calculation",
        [
            None,
            "invalid",
            100,
            4.5,
            [],
            (),
        ],
    )
    def test_invalid_calculation_returns_none(
        self,
        invalid_calculation: Any,
    ) -> None:
        execution_result = {
            "tool_outputs": [
                {
                    "tool": "financial_calculator",
                    "output": {
                        "calculation": invalid_calculation,
                    },
                }
            ],
        }

        result = FinancialAgent._extract_calculation_result(
            execution_result
        )

        assert result is None

    def test_valid_calculation_is_returned(self) -> None:
        calculation = {
            "metric": "revenue_growth",
            "value": 12.5,
            "unit": "percent",
        }

        execution_result = {
            "tool_outputs": [
                {
                    "tool": "financial_calculator",
                    "output": {
                        "calculation": calculation,
                    },
                }
            ],
        }

        result = FinancialAgent._extract_calculation_result(
            execution_result
        )

        assert result == calculation

    def test_skips_invalid_match_and_returns_later_valid_result(
        self,
    ) -> None:
        calculation = {
            "metric": "revenue_growth",
            "value": 8.2,
        }

        execution_result = {
            "tool_outputs": [
                {
                    "tool": "financial_calculator",
                    "output": {
                        "calculation": "invalid",
                    },
                },
                {
                    "tool": "financial_calculator",
                    "output": {
                        "calculation": calculation,
                    },
                },
            ],
        }

        result = FinancialAgent._extract_calculation_result(
            execution_result
        )

        assert result == calculation

    def test_returns_first_valid_calculation(self) -> None:
        first_calculation = {
            "value": 10,
        }

        second_calculation = {
            "value": 20,
        }

        execution_result = {
            "tool_outputs": [
                {
                    "tool": "financial_calculator",
                    "output": {
                        "calculation": first_calculation,
                    },
                },
                {
                    "tool": "financial_calculator",
                    "output": {
                        "calculation": second_calculation,
                    },
                },
            ],
        }

        result = FinancialAgent._extract_calculation_result(
            execution_result
        )

        assert result == first_calculation


# ---------------------------------------------------------------------------
# Company comparison extractor
# ---------------------------------------------------------------------------


class TestExtractComparisonResult:
    """Tests for _extract_comparison_result()."""

    @pytest.mark.parametrize(
        "execution_result",
        [
            {},
            {"tool_outputs": None},
            {"tool_outputs": "invalid"},
            {"tool_outputs": 123},
            {"tool_outputs": {}},
            {"tool_outputs": ()},
        ],
    )
    def test_invalid_tool_outputs_returns_none(
        self,
        execution_result: dict[str, Any],
    ) -> None:
        result = FinancialAgent._extract_comparison_result(
            execution_result
        )

        assert result is None

    def test_empty_tool_outputs_returns_none(self) -> None:
        result = FinancialAgent._extract_comparison_result(
            {
                "tool_outputs": [],
            }
        )

        assert result is None

    @pytest.mark.parametrize(
        "invalid_step",
        [
            None,
            "invalid",
            100,
            4.5,
            [],
            (),
        ],
    )
    def test_invalid_step_is_ignored(
        self,
        invalid_step: Any,
    ) -> None:
        result = FinancialAgent._extract_comparison_result(
            {
                "tool_outputs": [
                    invalid_step,
                ],
            }
        )

        assert result is None

    def test_wrong_tool_name_returns_none(self) -> None:
        result = FinancialAgent._extract_comparison_result(
            {
                "tool_outputs": [
                    {
                        "tool": "financial_calculator",
                        "output": {
                            "winner": "Apple",
                        },
                    }
                ],
            }
        )

        assert result is None

    @pytest.mark.parametrize(
        "invalid_output",
        [
            None,
            "invalid",
            100,
            4.5,
            [],
            (),
        ],
    )
    def test_invalid_output_returns_none(
        self,
        invalid_output: Any,
    ) -> None:
        result = FinancialAgent._extract_comparison_result(
            {
                "tool_outputs": [
                    {
                        "tool": "company_comparison",
                        "output": invalid_output,
                    }
                ],
            }
        )

        assert result is None

    def test_valid_comparison_output_is_returned(self) -> None:
        comparison = {
            "success": True,
            "companies": [
                "Apple",
                "Microsoft",
            ],
            "winner": "Microsoft",
        }

        result = FinancialAgent._extract_comparison_result(
            {
                "tool_outputs": [
                    {
                        "tool": "company_comparison",
                        "output": comparison,
                    }
                ],
            }
        )

        assert result == comparison

    def test_skips_invalid_match_and_returns_later_valid_result(
        self,
    ) -> None:
        comparison = {
            "winner": "Apple",
        }

        result = FinancialAgent._extract_comparison_result(
            {
                "tool_outputs": [
                    {
                        "tool": "company_comparison",
                        "output": "invalid",
                    },
                    {
                        "tool": "company_comparison",
                        "output": comparison,
                    },
                ],
            }
        )

        assert result == comparison

    def test_returns_first_valid_comparison(self) -> None:
        first_comparison = {
            "winner": "Apple",
        }

        second_comparison = {
            "winner": "Microsoft",
        }

        result = FinancialAgent._extract_comparison_result(
            {
                "tool_outputs": [
                    {
                        "tool": "company_comparison",
                        "output": first_comparison,
                    },
                    {
                        "tool": "company_comparison",
                        "output": second_comparison,
                    },
                ],
            }
        )

        assert result == first_comparison


# ---------------------------------------------------------------------------
# Calculated comparison extractor
# ---------------------------------------------------------------------------


class TestExtractCalculatedComparisonResult:
    """Tests for _extract_calculated_comparison_result()."""

    @pytest.mark.parametrize(
        "execution_result",
        [
            {},
            {"tool_outputs": None},
            {"tool_outputs": "invalid"},
            {"tool_outputs": 123},
            {"tool_outputs": {}},
            {"tool_outputs": ()},
        ],
    )
    def test_invalid_tool_outputs_returns_none(
        self,
        execution_result: dict[str, Any],
    ) -> None:
        result = (
            FinancialAgent._extract_calculated_comparison_result(
                execution_result
            )
        )

        assert result is None

    def test_empty_tool_outputs_returns_none(self) -> None:
        result = (
            FinancialAgent._extract_calculated_comparison_result(
                {
                    "tool_outputs": [],
                }
            )
        )

        assert result is None

    @pytest.mark.parametrize(
        "invalid_step",
        [
            None,
            "invalid",
            100,
            4.5,
            [],
            (),
        ],
    )
    def test_invalid_step_is_ignored(
        self,
        invalid_step: Any,
    ) -> None:
        result = (
            FinancialAgent._extract_calculated_comparison_result(
                {
                    "tool_outputs": [
                        invalid_step,
                    ],
                }
            )
        )

        assert result is None

    def test_wrong_tool_name_returns_none(self) -> None:
        result = (
            FinancialAgent._extract_calculated_comparison_result(
                {
                    "tool_outputs": [
                        {
                            "tool": "company_comparison",
                            "output": {
                                "success": True,
                            },
                        }
                    ],
                }
            )
        )

        assert result is None

    @pytest.mark.parametrize(
        "invalid_output",
        [
            None,
            "invalid",
            100,
            4.5,
            [],
            (),
        ],
    )
    def test_invalid_output_returns_none(
        self,
        invalid_output: Any,
    ) -> None:
        result = (
            FinancialAgent._extract_calculated_comparison_result(
                {
                    "tool_outputs": [
                        {
                            "tool": "calculated_comparison",
                            "output": invalid_output,
                        }
                    ],
                }
            )
        )

        assert result is None

    def test_valid_calculated_comparison_is_returned(
        self,
    ) -> None:
        calculated_comparison = {
            "success": True,
            "metric": "revenue_growth",
            "companies": [
                "Apple",
                "Microsoft",
            ],
            "reasoning": {
                "winner": "Microsoft",
            },
        }

        result = (
            FinancialAgent._extract_calculated_comparison_result(
                {
                    "tool_outputs": [
                        {
                            "tool": "calculated_comparison",
                            "output": calculated_comparison,
                        }
                    ],
                }
            )
        )

        assert result == calculated_comparison

    def test_skips_invalid_match_and_returns_later_valid_result(
        self,
    ) -> None:
        valid_result = {
            "success": True,
        }

        result = (
            FinancialAgent._extract_calculated_comparison_result(
                {
                    "tool_outputs": [
                        {
                            "tool": "calculated_comparison",
                            "output": "invalid",
                        },
                        {
                            "tool": "calculated_comparison",
                            "output": valid_result,
                        },
                    ],
                }
            )
        )

        assert result == valid_result

    def test_returns_first_valid_calculated_comparison(
        self,
    ) -> None:
        first_result = {
            "winner": "Apple",
        }

        second_result = {
            "winner": "Microsoft",
        }

        result = (
            FinancialAgent._extract_calculated_comparison_result(
                {
                    "tool_outputs": [
                        {
                            "tool": "calculated_comparison",
                            "output": first_result,
                        },
                        {
                            "tool": "calculated_comparison",
                            "output": second_result,
                        },
                    ],
                }
            )
        )

        assert result == first_result


# ---------------------------------------------------------------------------
# Calculated comparison reasoning extractor
# ---------------------------------------------------------------------------


class TestExtractCalculatedComparisonReasoning:
    """Tests for _extract_calculated_comparison_reasoning()."""

    @pytest.mark.parametrize(
        "calculated_comparison",
        [
            None,
            "invalid",
            100,
            3.5,
            [],
            (),
        ],
    )
    def test_invalid_calculated_comparison_returns_none(
        self,
        calculated_comparison: Any,
    ) -> None:
        result = (
            FinancialAgent._extract_calculated_comparison_reasoning(
                calculated_comparison
            )
        )

        assert result is None

    def test_missing_reasoning_returns_none(self) -> None:
        result = (
            FinancialAgent._extract_calculated_comparison_reasoning(
                {
                    "success": True,
                }
            )
        )

        assert result is None

    @pytest.mark.parametrize(
        "invalid_reasoning",
        [
            None,
            "invalid",
            100,
            4.5,
            [],
            (),
        ],
    )
    def test_invalid_reasoning_returns_none(
        self,
        invalid_reasoning: Any,
    ) -> None:
        result = (
            FinancialAgent._extract_calculated_comparison_reasoning(
                {
                    "reasoning": invalid_reasoning,
                }
            )
        )

        assert result is None

    def test_valid_reasoning_is_returned(self) -> None:
        reasoning = {
            "winner": "Microsoft",
            "explanation": (
                "Microsoft had stronger calculated growth."
            ),
        }

        result = (
            FinancialAgent._extract_calculated_comparison_reasoning(
                {
                    "reasoning": reasoning,
                }
            )
        )

        assert result == reasoning

    def test_empty_reasoning_dictionary_is_returned(self) -> None:
        result = (
            FinancialAgent._extract_calculated_comparison_reasoning(
                {
                    "reasoning": {},
                }
            )
        )

        assert result == {}


# ---------------------------------------------------------------------------
# Risk result extractor
# ---------------------------------------------------------------------------


class TestExtractRiskResult:
    """Tests for _extract_risk_result()."""

    @pytest.mark.parametrize(
        "execution_result",
        [
            {},
            {"tool_outputs": None},
            {"tool_outputs": "invalid"},
            {"tool_outputs": 123},
            {"tool_outputs": {}},
            {"tool_outputs": ()},
        ],
    )
    def test_invalid_tool_outputs_returns_none(
        self,
        execution_result: dict[str, Any],
    ) -> None:
        result = FinancialAgent._extract_risk_result(
            execution_result
        )

        assert result is None

    def test_empty_tool_outputs_returns_none(self) -> None:
        result = FinancialAgent._extract_risk_result(
            {
                "tool_outputs": [],
            }
        )

        assert result is None

    @pytest.mark.parametrize(
        "invalid_step",
        [
            None,
            "invalid",
            100,
            4.5,
            [],
            (),
        ],
    )
    def test_invalid_step_is_ignored(
        self,
        invalid_step: Any,
    ) -> None:
        result = FinancialAgent._extract_risk_result(
            {
                "tool_outputs": [
                    invalid_step,
                ],
            }
        )

        assert result is None

    def test_wrong_tool_name_returns_none(self) -> None:
        result = FinancialAgent._extract_risk_result(
            {
                "tool_outputs": [
                    {
                        "tool": "company_comparison",
                        "output": {
                            "risks": [],
                        },
                    }
                ],
            }
        )

        assert result is None

    @pytest.mark.parametrize(
        "invalid_output",
        [
            None,
            "invalid",
            100,
            4.5,
            [],
            (),
        ],
    )
    def test_invalid_output_returns_none(
        self,
        invalid_output: Any,
    ) -> None:
        result = FinancialAgent._extract_risk_result(
            {
                "tool_outputs": [
                    {
                        "tool": "risk_analysis",
                        "output": invalid_output,
                    }
                ],
            }
        )

        assert result is None

    def test_valid_risk_result_is_returned(self) -> None:
        risk_result = {
            "success": True,
            "risks": [
                {
                    "category": "cybersecurity",
                    "description": "Cybersecurity threats.",
                }
            ],
            "sources": [],
        }

        result = FinancialAgent._extract_risk_result(
            {
                "tool_outputs": [
                    {
                        "tool": "risk_analysis",
                        "output": risk_result,
                    }
                ],
            }
        )

        assert result == risk_result

    def test_skips_invalid_match_and_returns_later_valid_result(
        self,
    ) -> None:
        valid_result = {
            "success": True,
            "risks": [],
        }

        result = FinancialAgent._extract_risk_result(
            {
                "tool_outputs": [
                    {
                        "tool": "risk_analysis",
                        "output": "invalid",
                    },
                    {
                        "tool": "risk_analysis",
                        "output": valid_result,
                    },
                ],
            }
        )

        assert result == valid_result

    def test_returns_first_valid_risk_result(self) -> None:
        first_result = {
            "risks": [
                "Risk one",
            ],
        }

        second_result = {
            "risks": [
                "Risk two",
            ],
        }

        result = FinancialAgent._extract_risk_result(
            {
                "tool_outputs": [
                    {
                        "tool": "risk_analysis",
                        "output": first_result,
                    },
                    {
                        "tool": "risk_analysis",
                        "output": second_result,
                    },
                ],
            }
        )

        assert result == first_result


# ---------------------------------------------------------------------------
# Mixed tool output scenarios
# ---------------------------------------------------------------------------


class TestMixedToolOutputs:
    """Tests where multiple tool outputs are present together."""

    def test_each_extractor_returns_its_own_tool_result(
        self,
    ) -> None:
        calculation = {
            "value": 15.2,
        }

        comparison = {
            "winner": "Apple",
        }

        calculated_comparison = {
            "reasoning": {
                "winner": "Microsoft",
            },
        }

        risk_result = {
            "success": True,
            "risks": [
                "Competition",
            ],
        }

        execution_result = {
            "tool_outputs": [
                {
                    "tool": "financial_calculator",
                    "output": {
                        "calculation": calculation,
                    },
                },
                {
                    "tool": "company_comparison",
                    "output": comparison,
                },
                {
                    "tool": "calculated_comparison",
                    "output": calculated_comparison,
                },
                {
                    "tool": "risk_analysis",
                    "output": risk_result,
                },
            ],
        }

        extracted_calculation = (
            FinancialAgent._extract_calculation_result(
                execution_result
            )
        )

        extracted_comparison = (
            FinancialAgent._extract_comparison_result(
                execution_result
            )
        )

        extracted_calculated_comparison = (
            FinancialAgent
            ._extract_calculated_comparison_result(
                execution_result
            )
        )

        extracted_reasoning = (
            FinancialAgent
            ._extract_calculated_comparison_reasoning(
                extracted_calculated_comparison
            )
        )

        extracted_risk = (
            FinancialAgent._extract_risk_result(
                execution_result
            )
        )

        assert extracted_calculation == calculation
        assert extracted_comparison == comparison
        assert (
            extracted_calculated_comparison
            == calculated_comparison
        )
        assert extracted_reasoning == {
            "winner": "Microsoft",
        }
        assert extracted_risk == risk_result

    def test_unrelated_tools_do_not_produce_results(
        self,
    ) -> None:
        execution_result = {
            "tool_outputs": [
                {
                    "tool": "document_retrieval",
                    "output": {
                        "chunks": [],
                    },
                },
                {
                    "tool": "unknown_tool",
                    "output": {
                        "value": 100,
                    },
                },
            ],
        }

        assert (
            FinancialAgent._extract_calculation_result(
                execution_result
            )
            is None
        )

        assert (
            FinancialAgent._extract_comparison_result(
                execution_result
            )
            is None
        )

        assert (
            FinancialAgent
            ._extract_calculated_comparison_result(
                execution_result
            )
            is None
        )

        assert (
            FinancialAgent._extract_risk_result(
                execution_result
            )
            is None
        )