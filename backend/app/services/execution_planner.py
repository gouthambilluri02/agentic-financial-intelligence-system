class ExecutionPlanner:
    """
    Create an ordered multi-tool execution plan.

    The planner decides which tools should run based on the
    primary tool selected by ToolRouter.
    """

    def create_plan(
        self,
        selected_tool: str,
        intent: str,
        companies: list[str],
    ) -> list[str]:
        """
        Return an ordered list of tools to execute.
        """

        normalized_tool = (
            selected_tool.strip()
            if isinstance(selected_tool, str)
            else "document_retrieval"
        )

        normalized_intent = (
            intent.strip()
            if isinstance(intent, str)
            else "general_question"
        )

        plan: list[str] = []

        if normalized_tool == "financial_calculator":
            plan.extend(
                [
                    "financial_calculator",
                    "document_retrieval",
                ]
            )

        elif normalized_tool == "company_comparison":
            plan.extend(
                [
                    "company_comparison",
                    "document_retrieval",
                ]
            )

        elif normalized_tool == "risk_analysis":
            plan.extend(
                [
                    "risk_analysis",
                    "document_retrieval",
                ]
            )

        elif normalized_tool == "report_summary":
            plan.append(
                "document_retrieval"
            )

        elif normalized_intent == "risk_analysis":
            plan.extend(
                [
                    "risk_analysis",
                    "document_retrieval",
                ]
            )

        else:
            plan.append(
                "document_retrieval"
            )

        return self._remove_duplicates(
            plan
        )

    @staticmethod
    def _remove_duplicates(
        plan: list[str],
    ) -> list[str]:
        """
        Remove duplicate tool names while preserving order.
        """

        unique_plan: list[str] = []
        seen: set[str] = set()

        for tool_name in plan:
            if tool_name in seen:
                continue

            unique_plan.append(
                tool_name
            )

            seen.add(
                tool_name
            )

        return unique_plan


if __name__ == "__main__":
    planner = ExecutionPlanner()

    tests = [
        (
            "financial_calculator",
            "financial_metric",
            ["Apple"],
        ),
        (
            "company_comparison",
            "comparison",
            [
                "Apple",
                "Microsoft",
            ],
        ),
        (
            "risk_analysis",
            "risk_analysis",
            ["Microsoft"],
        ),
        (
            "report_summary",
            "summary",
            ["Microsoft"],
        ),
        (
            "document_retrieval",
            "general_question",
            ["Apple"],
        ),
    ]

    for selected_tool, intent, companies in tests:
        print("=" * 70)

        result = planner.create_plan(
            selected_tool=selected_tool,
            intent=intent,
            companies=companies,
        )

        print(
            f"Selected tool: {selected_tool}"
        )
        print(
            f"Intent: {intent}"
        )
        print(
            f"Companies: {companies}"
        )
        print(
            f"Execution plan: {result}"
        )