from __future__ import annotations

from time import perf_counter
from typing import Any

from backend.app.tools.tool_registry import ToolRegistry


class ToolExecutor:
    """
    Execute an ordered multi-tool plan.

    Responsibilities:
    - Receive a list of tool names from ExecutionPlanner.
    - Execute registered tools through ToolRegistry.
    - Handle document_retrieval as an existing retrieval result.
    - Capture tool failures without stopping the complete workflow.
    - Return structured outputs for the FinancialAgent and LLM.
    """

    def __init__(
        self,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        self.tool_registry = (
            tool_registry
            if tool_registry is not None
            else ToolRegistry()
        )

    def execute_plan(
        self,
        execution_plan: list[str],
        question: str,
        metric: str | None,
        companies: list[str],
        retrieved_chunks: list[dict],
    ) -> dict[str, Any]:
        """
        Execute all tools in the supplied order.

        Parameters:
        - execution_plan:
          Ordered names such as:
          [
              "financial_calculator",
              "document_retrieval",
          ]

        - question:
          Original user question.

        - metric:
          Detected financial metric.

        - companies:
          Resolved company names.

        - retrieved_chunks:
          Chunks already returned by RetrievalOrchestrator.

        Returns:
        {
            "success": bool,
            "execution_plan": [...],
            "executed_tools": [...],
            "successful_tools": [...],
            "failed_tools": [...],
            "tool_outputs": [...],
            "prompt_context": "...",
            "error": None | str,
        }
        """

        validation_error = self._validate_inputs(
            execution_plan=execution_plan,
            question=question,
            companies=companies,
            retrieved_chunks=retrieved_chunks,
        )

        if validation_error:
            return self._failure_response(
                execution_plan=execution_plan,
                message=validation_error,
            )

        normalized_plan = self._normalize_plan(
            execution_plan
        )

        tool_outputs: list[dict[str, Any]] = []
        executed_tools: list[str] = []
        successful_tools: list[str] = []
        failed_tools: list[str] = []

        total_start = perf_counter()

        for tool_name in normalized_plan:
            output = self._execute_one(
                tool_name=tool_name,
                question=question,
                metric=metric,
                companies=companies,
                retrieved_chunks=retrieved_chunks,
            )

            tool_outputs.append(output)
            executed_tools.append(tool_name)

            if output.get("success"):
                successful_tools.append(tool_name)
            else:
                failed_tools.append(tool_name)

        total_duration_ms = round(
            (perf_counter() - total_start) * 1000,
            2,
        )

        prompt_context = self.build_prompt_context(
            tool_outputs
        )

        overall_success = bool(
            successful_tools
        )

        overall_error = None

        if not overall_success:
            overall_error = (
                "Every tool in the execution plan failed."
            )

        return {
            "success": overall_success,
            "execution_plan": normalized_plan,
            "executed_tools": executed_tools,
            "successful_tools": successful_tools,
            "failed_tools": failed_tools,
            "tool_outputs": tool_outputs,
            "prompt_context": prompt_context,
            "duration_ms": total_duration_ms,
            "error": overall_error,
        }

    def _execute_one(
        self,
        tool_name: str,
        question: str,
        metric: str | None,
        companies: list[str],
        retrieved_chunks: list[dict],
    ) -> dict[str, Any]:
        """
        Execute one step from the plan.
        """

        start_time = perf_counter()

        if tool_name == "document_retrieval":
            result = self._execute_document_retrieval(
                retrieved_chunks
            )
        else:
            result = self._execute_registered_tool(
                tool_name=tool_name,
                question=question,
                metric=metric,
                companies=companies,
                retrieved_chunks=retrieved_chunks,
            )

        duration_ms = round(
            (perf_counter() - start_time) * 1000,
            2,
        )

        result["duration_ms"] = duration_ms

        return result

    def _execute_registered_tool(
        self,
        tool_name: str,
        question: str,
        metric: str | None,
        companies: list[str],
        retrieved_chunks: list[dict],
    ) -> dict[str, Any]:
        """
        Locate and execute a registered tool.
        """

        tool = self.tool_registry.get_tool(
            tool_name
        )

        if tool is None:
            return {
                "success": False,
                "tool": tool_name,
                "registered": False,
                "output": None,
                "prompt_context": (
                    f"Tool '{tool_name}' is not registered."
                ),
                "error": (
                    f"Tool '{tool_name}' is not registered."
                ),
            }

        try:
            output = tool.run(
                question=question,
                metric=metric,
                companies=companies,
                retrieved_chunks=retrieved_chunks,
            )

        except Exception as error:
            return {
                "success": False,
                "tool": tool_name,
                "registered": True,
                "output": None,
                "prompt_context": (
                    f"Tool '{tool_name}' failed during execution. "
                    "Do not invent a result."
                ),
                "error": str(error),
            }

        if not isinstance(output, dict):
            return {
                "success": False,
                "tool": tool_name,
                "registered": True,
                "output": None,
                "prompt_context": (
                    f"Tool '{tool_name}' returned an invalid "
                    "response."
                ),
                "error": (
                    "Registered tools must return a dictionary."
                ),
            }

        prompt_context = output.get(
            "prompt_context",
            "",
        )

        if not isinstance(prompt_context, str):
            prompt_context = ""

        return {
            "success": bool(
                output.get("success")
            ),
            "tool": tool_name,
            "registered": True,
            "output": output,
            "prompt_context": prompt_context,
            "error": output.get("error"),
        }

    @staticmethod
    def _execute_document_retrieval(
        retrieved_chunks: list[dict],
    ) -> dict[str, Any]:
        """
        Represent the existing retrieval result as a tool output.

        RetrievalOrchestrator already performed the actual search
        before ToolExecutor was called, so this step does not query
        ChromaDB again.
        """

        if not retrieved_chunks:
            return {
                "success": False,
                "tool": "document_retrieval",
                "registered": False,
                "output": {
                    "chunks": [],
                    "chunk_count": 0,
                },
                "prompt_context": (
                    "Document retrieval returned no relevant "
                    "financial-report context."
                ),
                "error": (
                    "No retrieved chunks were available."
                ),
            }

        summaries: list[str] = []

        for index, chunk in enumerate(
            retrieved_chunks,
            start=1,
        ):
            metadata = chunk.get(
                "metadata",
                {},
            )

            content = chunk.get(
                "content",
                "",
            )

            company = metadata.get(
                "company",
                "Unknown company",
            )

            source_file = metadata.get(
                "source_file",
                "Unknown source",
            )

            page = metadata.get(
                "page",
                "Unknown page",
            )

            content_preview = (
                content[:350].strip()
                if isinstance(content, str)
                else ""
            )

            summaries.append(
                (
                    f"Retrieved context {index}\n"
                    f"Company: {company}\n"
                    f"Source: {source_file}\n"
                    f"Stored page: {page}\n"
                    f"Content preview: {content_preview}"
                )
            )

        return {
            "success": True,
            "tool": "document_retrieval",
            "registered": False,
            "output": {
                "chunks": retrieved_chunks,
                "chunk_count": len(
                    retrieved_chunks
                ),
            },
            "prompt_context": (
                "Verified document retrieval output:\n\n"
                + "\n\n".join(summaries)
            ),
            "error": None,
        }

    @staticmethod
    def build_prompt_context(
        tool_outputs: list[dict[str, Any]],
    ) -> str:
        """
        Merge successful and failed tool outputs into one context
        block for the language model.
        """

        if not tool_outputs:
            return (
                "No tools were executed for this request."
            )

        sections = [
            "MULTI-TOOL EXECUTION RESULTS",
            (
                "Treat successful deterministic tool outputs as "
                "authoritative. Do not invent missing tool results."
            ),
        ]

        for index, result in enumerate(
            tool_outputs,
            start=1,
        ):
            tool_name = result.get(
                "tool",
                "unknown_tool",
            )

            success = result.get(
                "success",
                False,
            )

            sections.extend(
                [
                    "",
                    f"Tool step {index}: {tool_name}",
                    f"Success: {success}",
                ]
            )

            prompt_context = result.get(
                "prompt_context"
            )

            if (
                isinstance(prompt_context, str)
                and prompt_context.strip()
            ):
                sections.append(
                    prompt_context.strip()
                )

            error = result.get("error")

            if error:
                sections.append(
                    f"Tool error: {error}"
                )

        return "\n".join(sections)

    @staticmethod
    def _normalize_plan(
        execution_plan: list[str],
    ) -> list[str]:
        """
        Remove invalid names and duplicate tools while preserving
        their original order.
        """

        normalized_plan: list[str] = []
        seen_tools: set[str] = set()

        for tool_name in execution_plan:
            if not isinstance(
                tool_name,
                str,
            ):
                continue

            normalized_name = (
                tool_name.strip()
            )

            if not normalized_name:
                continue

            if normalized_name in seen_tools:
                continue

            normalized_plan.append(
                normalized_name
            )

            seen_tools.add(
                normalized_name
            )

        return normalized_plan

    @staticmethod
    def _validate_inputs(
        execution_plan: Any,
        question: Any,
        companies: Any,
        retrieved_chunks: Any,
    ) -> str | None:
        """
        Validate the complete execution request.
        """

        if not isinstance(
            execution_plan,
            list,
        ):
            return (
                "execution_plan must be provided as a list."
            )

        if not execution_plan:
            return (
                "execution_plan cannot be empty."
            )

        if (
            not isinstance(question, str)
            or not question.strip()
        ):
            return (
                "A non-empty financial question is required."
            )

        if not isinstance(
            companies,
            list,
        ):
            return (
                "companies must be provided as a list."
            )

        if not isinstance(
            retrieved_chunks,
            list,
        ):
            return (
                "retrieved_chunks must be provided as a list."
            )

        return None

    @staticmethod
    def _failure_response(
        execution_plan: Any,
        message: str,
    ) -> dict[str, Any]:
        """
        Return a consistent validation-failure response.
        """

        safe_plan = (
            execution_plan
            if isinstance(execution_plan, list)
            else []
        )

        return {
            "success": False,
            "execution_plan": safe_plan,
            "executed_tools": [],
            "successful_tools": [],
            "failed_tools": [],
            "tool_outputs": [],
            "prompt_context": (
                "Multi-tool execution was unavailable.\n"
                f"Reason: {message}"
            ),
            "duration_ms": 0.0,
            "error": message,
        }


if __name__ == "__main__":
    executor = ToolExecutor()

    test_chunks = [
        {
            "content": (
                "Year Ended September 28, 2024 2023 2022 "
                "Total net sales 391,035 383,285 394,328"
            ),
            "metadata": {
                "company": "Apple",
                "ticker": "AAPL",
                "fiscal_year": 2024,
                "document_type": "10K",
                "source_file": "apple_2024_10k.pdf",
                "page": 37,
            },
            "distance": 0.21,
        },
        {
            "content": (
                "Year Ended June 30, 2024 2023 2022 "
                "Revenue 245,122 211,915 198,270"
            ),
            "metadata": {
                "company": "Microsoft",
                "ticker": "MSFT",
                "fiscal_year": 2024,
                "document_type": "10K",
                "source_file": "microsoft_2024_10k.pdf",
                "page": 79,
            },
            "distance": 0.23,
        },
    ]

    test_cases = [
        {
            "name": "Revenue growth",
            "execution_plan": [
                "financial_calculator",
                "document_retrieval",
            ],
            "question": (
                "Calculate Apple's revenue growth."
            ),
            "metric": "revenue",
            "companies": ["Apple"],
        },
        {
            "name": "Company comparison",
            "execution_plan": [
                "company_comparison",
                "document_retrieval",
            ],
            "question": (
                "Compare Apple and Microsoft revenue."
            ),
            "metric": "revenue",
            "companies": [
                "Apple",
                "Microsoft",
            ],
        },
        {
            "name": "Document retrieval",
            "execution_plan": [
                "document_retrieval",
            ],
            "question": (
                "What risks did Microsoft disclose?"
            ),
            "metric": None,
            "companies": ["Microsoft"],
        },
    ]

    for test_case in test_cases:
        result = executor.execute_plan(
            execution_plan=(
                test_case["execution_plan"]
            ),
            question=test_case["question"],
            metric=test_case["metric"],
            companies=test_case["companies"],
            retrieved_chunks=test_chunks,
        )

        print("\n" + "=" * 80)
        print(f"Test: {test_case['name']}")

        print(
            "\nExecution plan:"
        )
        print(
            result["execution_plan"]
        )

        print(
            "\nExecuted tools:"
        )
        print(
            result["executed_tools"]
        )

        print(
            "\nSuccessful tools:"
        )
        print(
            result["successful_tools"]
        )

        print(
            "\nFailed tools:"
        )
        print(
            result["failed_tools"]
        )

        print(
            "\nDuration:"
        )
        print(
            f"{result['duration_ms']} ms"
        )

        print(
            "\nTool outputs:"
        )

        for tool_output in result[
            "tool_outputs"
        ]:
            print(
                {
                    "tool": tool_output.get(
                        "tool"
                    ),
                    "success": tool_output.get(
                        "success"
                    ),
                    "error": tool_output.get(
                        "error"
                    ),
                }
            )

        print(
            "\nCombined prompt context:"
        )
        print(
            result["prompt_context"]
        )