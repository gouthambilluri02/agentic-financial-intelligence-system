from backend.app.tools.base_tool import BaseTool
from backend.app.tools.risk_analysis_tool import RiskAnalysisTool
from backend.app.tools.company_comparison_tool import (
    CompanyComparisonTool,
)
from backend.app.tools.financial_calculator_tool import (
    FinancialCalculatorTool,
)


class ToolRegistry:
    """
    Store and provide executable tools for the FinancialAgent.
    """

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

        self.register_tool(
            FinancialCalculatorTool()
        )

        self.register_tool(
            CompanyComparisonTool()
        )

        self.register_tool(
            RiskAnalysisTool()
        )

    def register_tool(
        self,
        tool: BaseTool,
    ) -> None:
        """
        Register a tool using its unique name.
        """

        if not isinstance(
            tool,
            BaseTool,
        ):
            raise TypeError(
                "Registered tools must inherit from BaseTool."
            )

        tool_name = getattr(
            tool,
            "name",
            None,
        )

        if (
            not isinstance(tool_name, str)
            or not tool_name.strip()
        ):
            raise ValueError(
                "Every tool must define a non-empty name."
            )

        normalized_name = tool_name.strip()

        if normalized_name in self._tools:
            raise ValueError(
                f"A tool named '{normalized_name}' "
                "is already registered."
            )

        self._tools[
            normalized_name
        ] = tool

    def get_tool(
        self,
        tool_name: str,
    ) -> BaseTool | None:
        """
        Return a registered tool by name.
        """

        if not isinstance(
            tool_name,
            str,
        ):
            return None

        return self._tools.get(
            tool_name.strip()
        )

    def require_tool(
        self,
        tool_name: str,
    ) -> BaseTool:
        """
        Return a tool or raise a clear error.
        """

        tool = self.get_tool(
            tool_name
        )

        if tool is None:
            available_tools = ", ".join(
                self.list_tool_names()
            )

            raise KeyError(
                f"Tool '{tool_name}' is not registered. "
                f"Available tools: "
                f"{available_tools or 'none'}."
            )

        return tool

    def has_tool(
        self,
        tool_name: str,
    ) -> bool:
        """
        Check whether a tool is registered.
        """

        return self.get_tool(
            tool_name
        ) is not None

    def list_tool_names(
        self,
    ) -> list[str]:
        """
        Return registered tool names.
        """

        return sorted(
            self._tools.keys()
        )

    def list_tools(
        self,
    ) -> list[dict[str, str]]:
        """
        Return metadata for all tools.
        """

        return [
            self._tools[
                tool_name
            ].get_metadata()
            for tool_name in self.list_tool_names()
        ]


if __name__ == "__main__":
    registry = ToolRegistry()

    print("Registered tool names:")
    print(
        registry.list_tool_names()
    )

    print("\nRegistered tool metadata:")

    for tool_metadata in registry.list_tools():
        print(tool_metadata)

    print(
        "\nFinancial calculator available:"
    )
    print(
        registry.has_tool(
            "financial_calculator"
        )
    )

    print(
        "\nCompany comparison available:"
    )
    print(
        registry.has_tool(
            "company_comparison"
        )
    )

    print(
        "\nRisk analysis available:"
    )
    print(
        registry.has_tool(
            "risk_analysis"
        )
    )