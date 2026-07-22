from __future__ import annotations

from typing import Any

import pytest

from backend.app.services.multi_step_reasoning_service import (
    MultiStepReasoningService,
)


@pytest.fixture
def service() -> MultiStepReasoningService:
    return MultiStepReasoningService()


@pytest.fixture
def calculated_comparison_decomposition() -> dict[str, Any]:
    return {
        "is_complex": True,
        "reasoning_type": "calculated_comparison",
        "calculated_comparison": True,
        "companies": [
            "Apple",
            "Microsoft",
        ],
        "metric": "revenue",
    }


@pytest.fixture
def successful_calculation() -> dict[str, Any]:
    return {
        "success": True,
        "calculation_type": "growth_rate",
        "metric": "revenue",
        "results": [
            {
                "company": "Apple",
                "metric": "revenue",
                "current_year": 2024,
                "previous_year": 2023,
                "current_value": 391035.0,
                "previous_value": 383285.0,
                "result": 2.02,
                "unit": "percent",
                "formula": (
                    "((current_value - previous_value) "
                    "/ previous_value) * 100"
                ),
                "current_source": {
                    "document": "apple_2024.pdf",
                },
                "previous_source": {
                    "document": "apple_2023.pdf",
                },
            },
            {
                "company": "Microsoft",
                "metric": "revenue",
                "current_year": 2024,
                "previous_year": 2023,
                "current_value": 245122.0,
                "previous_value": 211915.0,
                "result": 15.67,
                "unit": "percent",
                "formula": (
                    "((current_value - previous_value) "
                    "/ previous_value) * 100"
                ),
                "current_source": {
                    "document": "microsoft_2024.pdf",
                },
                "previous_source": {
                    "document": "microsoft_2023.pdf",
                },
            },
        ],
        "error": None,
    }


class TestReason:
    def test_returns_none_for_invalid_decomposition(
        self,
        service: MultiStepReasoningService,
    ) -> None:
        result = service.reason(
            decomposition=None,  # type: ignore[arg-type]
            calculation=None,
        )

        assert result is None

    def test_returns_none_for_list_decomposition(
        self,
        service: MultiStepReasoningService,
    ) -> None:
        result = service.reason(
            decomposition=[],  # type: ignore[arg-type]
            calculation=None,
        )

        assert result is None

    def test_returns_none_when_not_calculated_comparison(
        self,
        service: MultiStepReasoningService,
    ) -> None:
        result = service.reason(
            decomposition={
                "calculated_comparison": False,
            },
            calculation=None,
        )

        assert result is None

    def test_returns_none_when_flag_is_missing(
        self,
        service: MultiStepReasoningService,
    ) -> None:
        result = service.reason(
            decomposition={},
            calculation=None,
        )

        assert result is None

    def test_returns_failure_when_calculation_is_none(
        self,
        service: MultiStepReasoningService,
        calculated_comparison_decomposition: dict[str, Any],
    ) -> None:
        result = service.reason(
            decomposition=calculated_comparison_decomposition,
            calculation=None,
        )

        assert result is not None
        assert result["success"] is False
        assert result["error"] == (
            "No verified calculation result was available."
        )

    def test_returns_failure_for_non_dict_calculation(
        self,
        service: MultiStepReasoningService,
        calculated_comparison_decomposition: dict[str, Any],
    ) -> None:
        result = service.reason(
            decomposition=calculated_comparison_decomposition,
            calculation=[],  # type: ignore[arg-type]
        )

        assert result is not None
        assert result["success"] is False
        assert result["error"] == (
            "No verified calculation result was available."
        )

    def test_returns_calculation_error_when_calculation_failed(
        self,
        service: MultiStepReasoningService,
        calculated_comparison_decomposition: dict[str, Any],
    ) -> None:
        result = service.reason(
            decomposition=calculated_comparison_decomposition,
            calculation={
                "success": False,
                "error": "Revenue values were missing.",
            },
        )

        assert result is not None
        assert result["success"] is False
        assert result["error"] == (
            "Revenue values were missing."
        )

    def test_returns_default_error_when_failed_calculation_has_no_error(
        self,
        service: MultiStepReasoningService,
        calculated_comparison_decomposition: dict[str, Any],
    ) -> None:
        result = service.reason(
            decomposition=calculated_comparison_decomposition,
            calculation={
                "success": False,
            },
        )

        assert result is not None
        assert result["success"] is False
        assert result["error"] == (
            "The required financial calculations were unsuccessful."
        )

    def test_returns_failure_when_results_are_not_a_list(
        self,
        service: MultiStepReasoningService,
        calculated_comparison_decomposition: dict[str, Any],
    ) -> None:
        result = service.reason(
            decomposition=calculated_comparison_decomposition,
            calculation={
                "success": True,
                "results": {
                    "company": "Apple",
                },
            },
        )

        assert result is not None
        assert result["success"] is False
        assert result["error"] == (
            "The calculation result did not contain "
            "a valid results list."
        )

    def test_returns_failure_when_results_are_missing(
        self,
        service: MultiStepReasoningService,
        calculated_comparison_decomposition: dict[str, Any],
    ) -> None:
        result = service.reason(
            decomposition=calculated_comparison_decomposition,
            calculation={
                "success": True,
            },
        )

        assert result is not None
        assert result["success"] is False
        assert result["error"] == (
            "At least two verified company calculations are "
            "required for multi-step comparison."
        )

    def test_returns_failure_for_empty_results(
        self,
        service: MultiStepReasoningService,
        calculated_comparison_decomposition: dict[str, Any],
    ) -> None:
        result = service.reason(
            decomposition=calculated_comparison_decomposition,
            calculation={
                "success": True,
                "results": [],
            },
        )

        assert result is not None
        assert result["success"] is False
        assert result["error"] == (
            "At least two verified company calculations are "
            "required for multi-step comparison."
        )

    def test_returns_failure_for_one_valid_company(
        self,
        service: MultiStepReasoningService,
        calculated_comparison_decomposition: dict[str, Any],
    ) -> None:
        result = service.reason(
            decomposition=calculated_comparison_decomposition,
            calculation={
                "success": True,
                "results": [
                    {
                        "company": "Apple",
                        "result": 10.0,
                    }
                ],
            },
        )

        assert result is not None
        assert result["success"] is False
        assert result["error"] == (
            "At least two verified company calculations are "
            "required for multi-step comparison."
        )

    def test_returns_successful_reasoning_result(
        self,
        service: MultiStepReasoningService,
        calculated_comparison_decomposition: dict[str, Any],
        successful_calculation: dict[str, Any],
    ) -> None:
        result = service.reason(
            decomposition=calculated_comparison_decomposition,
            calculation=successful_calculation,
        )

        assert result is not None
        assert result["success"] is True
        assert result["reasoning_type"] == (
            "calculated_comparison"
        )
        assert result["comparison_type"] == "growth_rate"
        assert result["metric"] == "revenue"
        assert result["error"] is None

    def test_ranks_highest_result_first(
        self,
        service: MultiStepReasoningService,
        calculated_comparison_decomposition: dict[str, Any],
        successful_calculation: dict[str, Any],
    ) -> None:
        result = service.reason(
            decomposition=calculated_comparison_decomposition,
            calculation=successful_calculation,
        )

        assert result is not None
        assert result["ranked_results"][0]["company"] == (
            "Microsoft"
        )
        assert result["ranked_results"][1]["company"] == (
            "Apple"
        )

    def test_identifies_strongest_company(
        self,
        service: MultiStepReasoningService,
        calculated_comparison_decomposition: dict[str, Any],
        successful_calculation: dict[str, Any],
    ) -> None:
        result = service.reason(
            decomposition=calculated_comparison_decomposition,
            calculation=successful_calculation,
        )

        assert result is not None
        assert result["strongest_company"] == "Microsoft"
        assert result["strongest_result"] == 15.67

    def test_identifies_weakest_company(
        self,
        service: MultiStepReasoningService,
        calculated_comparison_decomposition: dict[str, Any],
        successful_calculation: dict[str, Any],
    ) -> None:
        result = service.reason(
            decomposition=calculated_comparison_decomposition,
            calculation=successful_calculation,
        )

        assert result is not None
        assert result["weakest_company"] == "Apple"
        assert result["weakest_result"] == 2.02

    def test_calculates_difference_in_percentage_points(
        self,
        service: MultiStepReasoningService,
        calculated_comparison_decomposition: dict[str, Any],
        successful_calculation: dict[str, Any],
    ) -> None:
        result = service.reason(
            decomposition=calculated_comparison_decomposition,
            calculation=successful_calculation,
        )

        assert result is not None
        assert result["difference_percentage_points"] == 13.65

    def test_rounds_difference_to_two_decimal_places(
        self,
        service: MultiStepReasoningService,
        calculated_comparison_decomposition: dict[str, Any],
    ) -> None:
        result = service.reason(
            decomposition=calculated_comparison_decomposition,
            calculation={
                "success": True,
                "metric": "revenue",
                "results": [
                    {
                        "company": "Apple",
                        "result": 10.123,
                    },
                    {
                        "company": "Microsoft",
                        "result": 5.111,
                    },
                ],
            },
        )

        assert result is not None
        assert result["difference_percentage_points"] == 5.01

    def test_uses_default_comparison_type(
        self,
        service: MultiStepReasoningService,
        calculated_comparison_decomposition: dict[str, Any],
    ) -> None:
        result = service.reason(
            decomposition=calculated_comparison_decomposition,
            calculation={
                "success": True,
                "metric": "revenue",
                "results": [
                    {
                        "company": "Apple",
                        "result": 10.0,
                    },
                    {
                        "company": "Microsoft",
                        "result": 5.0,
                    },
                ],
            },
        )

        assert result is not None
        assert result["comparison_type"] == (
            "calculated_metric"
        )

    def test_metric_is_taken_from_calculation(
        self,
        service: MultiStepReasoningService,
        calculated_comparison_decomposition: dict[str, Any],
    ) -> None:
        result = service.reason(
            decomposition=calculated_comparison_decomposition,
            calculation={
                "success": True,
                "metric": "operating_margin",
                "results": [
                    {
                        "company": "Apple",
                        "result": 30.0,
                    },
                    {
                        "company": "Microsoft",
                        "result": 40.0,
                    },
                ],
            },
        )

        assert result is not None
        assert result["metric"] == "operating_margin"

    def test_supports_negative_results(
        self,
        service: MultiStepReasoningService,
        calculated_comparison_decomposition: dict[str, Any],
    ) -> None:
        result = service.reason(
            decomposition=calculated_comparison_decomposition,
            calculation={
                "success": True,
                "metric": "revenue",
                "results": [
                    {
                        "company": "Apple",
                        "result": -5.0,
                    },
                    {
                        "company": "Microsoft",
                        "result": -10.0,
                    },
                ],
            },
        )

        assert result is not None
        assert result["strongest_company"] == "Apple"
        assert result["weakest_company"] == "Microsoft"
        assert result["difference_percentage_points"] == 5.0

    def test_supports_more_than_two_companies(
        self,
        service: MultiStepReasoningService,
        calculated_comparison_decomposition: dict[str, Any],
    ) -> None:
        result = service.reason(
            decomposition=calculated_comparison_decomposition,
            calculation={
                "success": True,
                "metric": "revenue",
                "results": [
                    {
                        "company": "Apple",
                        "result": 10.0,
                    },
                    {
                        "company": "Microsoft",
                        "result": 20.0,
                    },
                    {
                        "company": "Google",
                        "result": 15.0,
                    },
                ],
            },
        )

        assert result is not None
        assert [
            item["company"]
            for item in result["ranked_results"]
        ] == [
            "Microsoft",
            "Google",
            "Apple",
        ]

    def test_tie_result_has_zero_difference(
        self,
        service: MultiStepReasoningService,
        calculated_comparison_decomposition: dict[str, Any],
    ) -> None:
        result = service.reason(
            decomposition=calculated_comparison_decomposition,
            calculation={
                "success": True,
                "metric": "revenue",
                "results": [
                    {
                        "company": "Apple",
                        "result": 10.0,
                    },
                    {
                        "company": "Microsoft",
                        "result": 10.0,
                    },
                ],
            },
        )

        assert result is not None
        assert result["difference_percentage_points"] == 0.0
        assert "reported the same revenue growth rate" in (
            result["conclusion"]
        )

    def test_invalid_results_are_filtered_before_reasoning(
        self,
        service: MultiStepReasoningService,
        calculated_comparison_decomposition: dict[str, Any],
    ) -> None:
        result = service.reason(
            decomposition=calculated_comparison_decomposition,
            calculation={
                "success": True,
                "metric": "revenue",
                "results": [
                    {
                        "company": "",
                        "result": 100.0,
                    },
                    {
                        "company": "Apple",
                        "result": 10.0,
                    },
                    {
                        "company": "Microsoft",
                        "result": 20.0,
                    },
                    "invalid result",
                ],
            },
        )

        assert result is not None
        assert result["success"] is True
        assert len(result["ranked_results"]) == 2


class TestNormalizeCalculationResults:
    def test_normalizes_valid_results(self) -> None:
        results = (
            MultiStepReasoningService
            ._normalize_calculation_results(
                [
                    {
                        "company": "Apple",
                        "metric": "revenue",
                        "current_year": 2024,
                        "previous_year": 2023,
                        "current_value": 391035.0,
                        "previous_value": 383285.0,
                        "result": 2.02,
                        "unit": "percent",
                        "formula": "growth formula",
                        "current_source": {
                            "document": "current.pdf",
                        },
                        "previous_source": {
                            "document": "previous.pdf",
                        },
                    }
                ]
            )
        )

        assert results == [
            {
                "company": "Apple",
                "metric": "revenue",
                "current_year": 2024,
                "previous_year": 2023,
                "current_value": 391035.0,
                "previous_value": 383285.0,
                "result": 2.02,
                "unit": "percent",
                "formula": "growth formula",
                "current_source": {
                    "document": "current.pdf",
                },
                "previous_source": {
                    "document": "previous.pdf",
                },
            }
        ]

    def test_strips_company_whitespace(self) -> None:
        results = (
            MultiStepReasoningService
            ._normalize_calculation_results(
                [
                    {
                        "company": " Apple ",
                        "result": 10.0,
                    }
                ]
            )
        )

        assert results[0]["company"] == "Apple"

    def test_removes_duplicate_companies_case_insensitively(
        self,
    ) -> None:
        results = (
            MultiStepReasoningService
            ._normalize_calculation_results(
                [
                    {
                        "company": "Apple",
                        "result": 10.0,
                    },
                    {
                        "company": "apple",
                        "result": 20.0,
                    },
                    {
                        "company": "APPLE",
                        "result": 30.0,
                    },
                ]
            )
        )

        assert len(results) == 1
        assert results[0]["company"] == "Apple"
        assert results[0]["result"] == 10.0

    @pytest.mark.parametrize(
        "invalid_result",
        [
            None,
            "invalid",
            123,
            [],
            (),
        ],
    )
    def test_ignores_non_dictionary_results(
        self,
        invalid_result: Any,
    ) -> None:
        results = (
            MultiStepReasoningService
            ._normalize_calculation_results(
                [
                    invalid_result,
                    {
                        "company": "Apple",
                        "result": 10.0,
                    },
                ]  # type: ignore[list-item]
            )
        )

        assert len(results) == 1
        assert results[0]["company"] == "Apple"

    @pytest.mark.parametrize(
        "invalid_company",
        [
            None,
            "",
            "   ",
            123,
            True,
            [],
            {},
        ],
    )
    def test_ignores_invalid_company_names(
        self,
        invalid_company: Any,
    ) -> None:
        results = (
            MultiStepReasoningService
            ._normalize_calculation_results(
                [
                    {
                        "company": invalid_company,
                        "result": 10.0,
                    }
                ]
            )
        )

        assert results == []

    @pytest.mark.parametrize(
        "invalid_value",
        [
            None,
            "10.0",
            [],
            {},
        ],
    )
    def test_ignores_non_numeric_results(
        self,
        invalid_value: Any,
    ) -> None:
        results = (
            MultiStepReasoningService
            ._normalize_calculation_results(
                [
                    {
                        "company": "Apple",
                        "result": invalid_value,
                    }
                ]
            )
        )

        assert results == []

    def test_accepts_integer_result(self) -> None:
        results = (
            MultiStepReasoningService
            ._normalize_calculation_results(
                [
                    {
                        "company": "Apple",
                        "result": 10,
                    }
                ]
            )
        )

        assert results[0]["result"] == 10.0

    def test_accepts_float_result(self) -> None:
        results = (
            MultiStepReasoningService
            ._normalize_calculation_results(
                [
                    {
                        "company": "Apple",
                        "result": 10.5,
                    }
                ]
            )
        )

        assert results[0]["result"] == 10.5

    def test_rounds_numeric_result_to_two_decimals(self) -> None:
        results = (
            MultiStepReasoningService
            ._normalize_calculation_results(
                [
                    {
                        "company": "Apple",
                        "result": 10.126,
                    }
                ]
            )
        )

        assert results[0]["result"] == 10.13

    def test_uses_default_unit_when_missing(self) -> None:
        results = (
            MultiStepReasoningService
            ._normalize_calculation_results(
                [
                    {
                        "company": "Apple",
                        "result": 10.0,
                    }
                ]
            )
        )

        assert results[0]["unit"] == "percent"

    def test_preserves_explicit_unit(self) -> None:
        results = (
            MultiStepReasoningService
            ._normalize_calculation_results(
                [
                    {
                        "company": "Apple",
                        "result": 10.0,
                        "unit": "percentage_points",
                    }
                ]
            )
        )

        assert results[0]["unit"] == (
            "percentage_points"
        )

    def test_missing_optional_fields_become_none(self) -> None:
        results = (
            MultiStepReasoningService
            ._normalize_calculation_results(
                [
                    {
                        "company": "Apple",
                        "result": 10.0,
                    }
                ]
            )
        )

        result = results[0]

        assert result["metric"] is None
        assert result["current_year"] is None
        assert result["previous_year"] is None
        assert result["current_value"] is None
        assert result["previous_value"] is None
        assert result["formula"] is None
        assert result["current_source"] is None
        assert result["previous_source"] is None

    def test_empty_results_return_empty_list(self) -> None:
        results = (
            MultiStepReasoningService
            ._normalize_calculation_results([])
        )

        assert results == []


class TestBuildConclusion:
    def test_builds_stronger_performer_conclusion(self) -> None:
        result = MultiStepReasoningService._build_conclusion(
            strongest={
                "company": "Microsoft",
                "result": 15.67,
            },
            weakest={
                "company": "Apple",
                "result": 2.02,
            },
            difference=13.65,
            metric="revenue",
        )

        assert result == (
            "Microsoft was the stronger performer based on "
            "revenue growth. Microsoft reported 15.67% growth "
            "compared with Apple's 2.02%, a difference of "
            "13.65 percentage points."
        )

    def test_formats_metric_with_underscores(self) -> None:
        result = MultiStepReasoningService._build_conclusion(
            strongest={
                "company": "Microsoft",
                "result": 20.0,
            },
            weakest={
                "company": "Apple",
                "result": 10.0,
            },
            difference=10.0,
            metric="operating_margin",
        )

        assert "operating margin growth" in result

    def test_uses_default_metric_for_none(self) -> None:
        result = MultiStepReasoningService._build_conclusion(
            strongest={
                "company": "Microsoft",
                "result": 20.0,
            },
            weakest={
                "company": "Apple",
                "result": 10.0,
            },
            difference=10.0,
            metric=None,
        )

        assert "financial metric growth" in result

    def test_uses_default_metric_for_empty_string(self) -> None:
        result = MultiStepReasoningService._build_conclusion(
            strongest={
                "company": "Microsoft",
                "result": 20.0,
            },
            weakest={
                "company": "Apple",
                "result": 10.0,
            },
            difference=10.0,
            metric="   ",
        )

        assert "financial metric growth" in result

    def test_builds_tie_conclusion(self) -> None:
        result = MultiStepReasoningService._build_conclusion(
            strongest={
                "company": "Apple",
                "result": 10.0,
            },
            weakest={
                "company": "Microsoft",
                "result": 10.0,
            },
            difference=0.0,
            metric="revenue",
        )

        assert result == (
            "Apple and Microsoft reported the same revenue "
            "growth rate of 10.00%."
        )

    def test_formats_results_to_two_decimal_places(self) -> None:
        result = MultiStepReasoningService._build_conclusion(
            strongest={
                "company": "Apple",
                "result": 10,
            },
            weakest={
                "company": "Microsoft",
                "result": 5,
            },
            difference=5,
            metric="revenue",
        )

        assert "10.00%" in result
        assert "5.00%" in result
        assert "5.00 percentage points" in result


class TestBuildAnswer:
    def test_returns_none_for_none_input(self) -> None:
        result = MultiStepReasoningService.build_answer(
            None
        )

        assert result is None

    def test_returns_none_for_non_dict_input(self) -> None:
        result = MultiStepReasoningService.build_answer(
            []  # type: ignore[arg-type]
        )

        assert result is None

    def test_builds_failure_answer(self) -> None:
        result = MultiStepReasoningService.build_answer(
            {
                "success": False,
                "error": "Calculation failed.",
            }
        )

        assert result == (
            "The calculated comparison could not be completed.\n\n"
            "Reason: Calculation failed."
        )

    def test_builds_default_failure_answer_when_error_missing(
        self,
    ) -> None:
        result = MultiStepReasoningService.build_answer(
            {
                "success": False,
            }
        )

        assert result == (
            "The calculated comparison could not be completed.\n\n"
            "Reason: The multi-step comparison could not "
            "be completed."
        )

    def test_builds_successful_answer(self) -> None:
        result = MultiStepReasoningService.build_answer(
            {
                "success": True,
                "ranked_results": [
                    {
                        "company": "Microsoft",
                        "result": 15.67,
                        "current_year": 2024,
                        "previous_year": 2023,
                    },
                    {
                        "company": "Apple",
                        "result": 2.02,
                        "current_year": 2024,
                        "previous_year": 2023,
                    },
                ],
                "conclusion": (
                    "Microsoft was the stronger performer."
                ),
            }
        )

        assert result == (
            "Verified calculated growth comparison:\n"
            "\n"
            "- Microsoft: 15.67% from 2023 to 2024\n"
            "- Apple: 2.02% from 2023 to 2024\n"
            "\n"
            "Microsoft was the stronger performer."
        )

    def test_uses_default_company_name(self) -> None:
        result = MultiStepReasoningService.build_answer(
            {
                "success": True,
                "ranked_results": [
                    {
                        "result": 10.0,
                        "current_year": 2024,
                        "previous_year": 2023,
                    }
                ],
                "conclusion": "",
            }
        )

        assert result is not None
        assert "- Unknown company: 10.00%" in result

    def test_uses_default_period_labels_when_years_missing(
        self,
    ) -> None:
        result = MultiStepReasoningService.build_answer(
            {
                "success": True,
                "ranked_results": [
                    {
                        "company": "Apple",
                        "result": 10.0,
                    }
                ],
                "conclusion": "",
            }
        )

        assert result is not None
        assert (
            "- Apple: 10.00% from previous period "
            "to current period"
        ) in result

    def test_ignores_non_numeric_result_values(self) -> None:
        result = MultiStepReasoningService.build_answer(
            {
                "success": True,
                "ranked_results": [
                    {
                        "company": "Apple",
                        "result": "10.0",
                    },
                    {
                        "company": "Microsoft",
                        "result": 20.0,
                    },
                ],
                "conclusion": "",
            }
        )

        assert result is not None
        assert "Apple" not in result
        assert "- Microsoft: 20.00%" in result

    def test_answer_without_conclusion(self) -> None:
        result = MultiStepReasoningService.build_answer(
            {
                "success": True,
                "ranked_results": [
                    {
                        "company": "Apple",
                        "result": 10.0,
                        "current_year": 2024,
                        "previous_year": 2023,
                    }
                ],
                "conclusion": "",
            }
        )

        assert result == (
            "Verified calculated growth comparison:\n"
            "\n"
            "- Apple: 10.00% from 2023 to 2024"
        )

    def test_answer_with_empty_ranked_results(self) -> None:
        result = MultiStepReasoningService.build_answer(
            {
                "success": True,
                "ranked_results": [],
                "conclusion": "No ranking was available.",
            }
        )

        assert result == (
            "Verified calculated growth comparison:\n"
            "\n"
            "\n"
            "No ranking was available."
        )

    def test_answer_defaults_when_ranked_results_missing(
        self,
    ) -> None:
        result = MultiStepReasoningService.build_answer(
            {
                "success": True,
                "conclusion": "",
            }
        )

        assert result == (
            "Verified calculated growth comparison:\n"
        )


class TestFailure:
    def test_failure_response_schema(self) -> None:
        result = MultiStepReasoningService._failure(
            "Comparison failed."
        )

        assert result == {
            "success": False,
            "reasoning_type": "calculated_comparison",
            "comparison_type": None,
            "metric": None,
            "ranked_results": [],
            "strongest_company": None,
            "strongest_result": None,
            "weakest_company": None,
            "weakest_result": None,
            "difference_percentage_points": None,
            "conclusion": None,
            "error": "Comparison failed.",
        }

    def test_failure_preserves_error_message(self) -> None:
        result = MultiStepReasoningService._failure(
            "At least two companies are required."
        )

        assert result["error"] == (
            "At least two companies are required."
        )