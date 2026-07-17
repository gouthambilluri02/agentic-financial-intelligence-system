from numbers import Real


class FinancialCalculatorError(ValueError):
    """
    Raised when a financial calculation cannot be completed
    because the supplied values are invalid.
    """


class FinancialCalculator:
    """
    Perform deterministic financial calculations using Python.

    The language model should explain these results, but it should
    not perform the arithmetic itself.
    """

    @staticmethod
    def calculate_growth_rate(
        current_value: Real,
        previous_value: Real,
    ) -> dict:
        """
        Calculate percentage growth from a previous value
        to a current value.

        Formula:
        ((current - previous) / previous) * 100
        """

        current = FinancialCalculator._validate_number(
            current_value,
            "current_value",
        )

        previous = FinancialCalculator._validate_number(
            previous_value,
            "previous_value",
        )

        if previous == 0:
            raise FinancialCalculatorError(
                "previous_value cannot be zero when calculating "
                "a growth rate."
            )

        result = (
            (current - previous)
            / previous
        ) * 100

        return {
            "calculation": "growth_rate",
            "current_value": current,
            "previous_value": previous,
            "result": round(result, 2),
            "unit": "percent",
            "formula": (
                "((current_value - previous_value) "
                "/ previous_value) * 100"
            ),
        }

    @staticmethod
    def calculate_gross_margin(
        revenue: Real,
        cost_of_revenue: Real,
    ) -> dict:
        """
        Calculate gross margin.

        Formula:
        ((revenue - cost_of_revenue) / revenue) * 100
        """

        revenue_value = FinancialCalculator._validate_number(
            revenue,
            "revenue",
        )

        cost_value = FinancialCalculator._validate_number(
            cost_of_revenue,
            "cost_of_revenue",
        )

        FinancialCalculator._ensure_nonzero(
            revenue_value,
            "revenue",
        )

        gross_profit = revenue_value - cost_value

        result = (
            gross_profit
            / revenue_value
        ) * 100

        return {
            "calculation": "gross_margin",
            "revenue": revenue_value,
            "cost_of_revenue": cost_value,
            "gross_profit": round(gross_profit, 2),
            "result": round(result, 2),
            "unit": "percent",
            "formula": (
                "((revenue - cost_of_revenue) / revenue) * 100"
            ),
        }

    @staticmethod
    def calculate_operating_margin(
        operating_income: Real,
        revenue: Real,
    ) -> dict:
        """
        Calculate operating margin.

        Formula:
        (operating_income / revenue) * 100
        """

        operating_income_value = (
            FinancialCalculator._validate_number(
                operating_income,
                "operating_income",
            )
        )

        revenue_value = FinancialCalculator._validate_number(
            revenue,
            "revenue",
        )

        FinancialCalculator._ensure_nonzero(
            revenue_value,
            "revenue",
        )

        result = (
            operating_income_value
            / revenue_value
        ) * 100

        return {
            "calculation": "operating_margin",
            "operating_income": operating_income_value,
            "revenue": revenue_value,
            "result": round(result, 2),
            "unit": "percent",
            "formula": (
                "(operating_income / revenue) * 100"
            ),
        }

    @staticmethod
    def calculate_net_profit_margin(
        net_income: Real,
        revenue: Real,
    ) -> dict:
        """
        Calculate net profit margin.

        Formula:
        (net_income / revenue) * 100
        """

        net_income_value = FinancialCalculator._validate_number(
            net_income,
            "net_income",
        )

        revenue_value = FinancialCalculator._validate_number(
            revenue,
            "revenue",
        )

        FinancialCalculator._ensure_nonzero(
            revenue_value,
            "revenue",
        )

        result = (
            net_income_value
            / revenue_value
        ) * 100

        return {
            "calculation": "net_profit_margin",
            "net_income": net_income_value,
            "revenue": revenue_value,
            "result": round(result, 2),
            "unit": "percent",
            "formula": (
                "(net_income / revenue) * 100"
            ),
        }

    @staticmethod
    def calculate_current_ratio(
        current_assets: Real,
        current_liabilities: Real,
    ) -> dict:
        """
        Calculate the current ratio.

        Formula:
        current_assets / current_liabilities
        """

        assets = FinancialCalculator._validate_number(
            current_assets,
            "current_assets",
        )

        liabilities = FinancialCalculator._validate_number(
            current_liabilities,
            "current_liabilities",
        )

        FinancialCalculator._ensure_nonzero(
            liabilities,
            "current_liabilities",
        )

        result = assets / liabilities

        return {
            "calculation": "current_ratio",
            "current_assets": assets,
            "current_liabilities": liabilities,
            "result": round(result, 2),
            "unit": "ratio",
            "formula": (
                "current_assets / current_liabilities"
            ),
        }

    @staticmethod
    def calculate_debt_ratio(
        total_liabilities: Real,
        total_assets: Real,
    ) -> dict:
        """
        Calculate the debt ratio.

        Formula:
        (total_liabilities / total_assets) * 100
        """

        liabilities = FinancialCalculator._validate_number(
            total_liabilities,
            "total_liabilities",
        )

        assets = FinancialCalculator._validate_number(
            total_assets,
            "total_assets",
        )

        FinancialCalculator._ensure_nonzero(
            assets,
            "total_assets",
        )

        result = (
            liabilities
            / assets
        ) * 100

        return {
            "calculation": "debt_ratio",
            "total_liabilities": liabilities,
            "total_assets": assets,
            "result": round(result, 2),
            "unit": "percent",
            "formula": (
                "(total_liabilities / total_assets) * 100"
            ),
        }

    @staticmethod
    def calculate_return_on_assets(
        net_income: Real,
        average_total_assets: Real,
    ) -> dict:
        """
        Calculate return on assets.

        Formula:
        (net_income / average_total_assets) * 100
        """

        income = FinancialCalculator._validate_number(
            net_income,
            "net_income",
        )

        assets = FinancialCalculator._validate_number(
            average_total_assets,
            "average_total_assets",
        )

        FinancialCalculator._ensure_nonzero(
            assets,
            "average_total_assets",
        )

        result = (
            income
            / assets
        ) * 100

        return {
            "calculation": "return_on_assets",
            "net_income": income,
            "average_total_assets": assets,
            "result": round(result, 2),
            "unit": "percent",
            "formula": (
                "(net_income / average_total_assets) * 100"
            ),
        }

    @staticmethod
    def calculate_return_on_equity(
        net_income: Real,
        average_shareholders_equity: Real,
    ) -> dict:
        """
        Calculate return on equity.

        Formula:
        (net_income / average_shareholders_equity) * 100
        """

        income = FinancialCalculator._validate_number(
            net_income,
            "net_income",
        )

        equity = FinancialCalculator._validate_number(
            average_shareholders_equity,
            "average_shareholders_equity",
        )

        FinancialCalculator._ensure_nonzero(
            equity,
            "average_shareholders_equity",
        )

        result = (
            income
            / equity
        ) * 100

        return {
            "calculation": "return_on_equity",
            "net_income": income,
            "average_shareholders_equity": equity,
            "result": round(result, 2),
            "unit": "percent",
            "formula": (
                "(net_income / average_shareholders_equity) * 100"
            ),
        }

    @staticmethod
    def calculate_free_cash_flow(
        operating_cash_flow: Real,
        capital_expenditures: Real,
    ) -> dict:
        """
        Calculate free cash flow.

        Formula:
        operating_cash_flow - capital_expenditures
        """

        operating_cash = FinancialCalculator._validate_number(
            operating_cash_flow,
            "operating_cash_flow",
        )

        capital_spending = FinancialCalculator._validate_number(
            capital_expenditures,
            "capital_expenditures",
        )

        result = operating_cash - capital_spending

        return {
            "calculation": "free_cash_flow",
            "operating_cash_flow": operating_cash,
            "capital_expenditures": capital_spending,
            "result": round(result, 2),
            "unit": "currency",
            "formula": (
                "operating_cash_flow - capital_expenditures"
            ),
        }

    @staticmethod
    def calculate_eps_growth(
        current_eps: Real,
        previous_eps: Real,
    ) -> dict:
        """
        Calculate earnings-per-share growth.
        """

        growth_result = (
            FinancialCalculator.calculate_growth_rate(
                current_value=current_eps,
                previous_value=previous_eps,
            )
        )

        growth_result["calculation"] = "eps_growth"
        growth_result["current_eps"] = (
            growth_result.pop("current_value")
        )
        growth_result["previous_eps"] = (
            growth_result.pop("previous_value")
        )

        return growth_result

    @staticmethod
    def _validate_number(
        value: Real,
        field_name: str,
    ) -> float:
        """
        Validate and convert a numeric input.

        Boolean values are rejected because bool is technically
        a subclass of int in Python.
        """

        if isinstance(value, bool) or not isinstance(value, Real):
            raise FinancialCalculatorError(
                f"{field_name} must be a valid numeric value."
            )

        return float(value)

    @staticmethod
    def _ensure_nonzero(
        value: float,
        field_name: str,
    ) -> None:
        """
        Prevent division by zero.
        """

        if value == 0:
            raise FinancialCalculatorError(
                f"{field_name} cannot be zero for this calculation."
            )


if __name__ == "__main__":
    calculator = FinancialCalculator()

    test_results = [
        calculator.calculate_growth_rate(
            current_value=391_035,
            previous_value=383_285,
        ),
        calculator.calculate_operating_margin(
            operating_income=123_216,
            revenue=391_035,
        ),
        calculator.calculate_net_profit_margin(
            net_income=93_736,
            revenue=391_035,
        ),
        calculator.calculate_current_ratio(
            current_assets=143_566,
            current_liabilities=145_308,
        ),
        calculator.calculate_free_cash_flow(
            operating_cash_flow=118_254,
            capital_expenditures=9_447,
        ),
        calculator.calculate_eps_growth(
            current_eps=6.08,
            previous_eps=6.13,
        ),
    ]

    for result in test_results:
        print("\n" + "=" * 70)

        for key, value in result.items():
            print(f"{key}: {value}")