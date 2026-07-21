from __future__ import annotations

from typing import Any


class ExecutionTraceService:
    """
    Build a safe execution trace for the financial-agent workflow.
    """

    def build_trace(
        self,
        selected_tool: str,
        retrieval_plan: dict[str, Any],
        retrieved_chunks: list[dict[str, Any]],
        retry_performed: bool,
        retry_count: int,
        retrieval_sufficient: bool,
        execution_result: dict[str, Any],
        deterministic_answer_used: bool,
    ) -> list[dict[str, Any]]:
        """
        Return an ordered list of workflow execution steps.
        """

        safe_plan = (
            retrieval_plan
            if isinstance(retrieval_plan, dict)
            else {}
        )

        safe_chunks = (
            retrieved_chunks
            if isinstance(retrieved_chunks, list)
            else []
        )

        safe_execution_result = (
            execution_result
            if isinstance(execution_result, dict)
            else {}
        )

        trace: list[dict[str, Any]] = [
            {
                "step": 1,
                "component": "query_analysis",
                "status": "success",
                "duration_ms": None,
                "details": (
                    f"Primary tool selected: {selected_tool}"
                ),
            },
            {
                "step": 2,
                "component": "document_retrieval",
                "status": (
                    "success"
                    if safe_chunks
                    else "failed"
                ),
                "duration_ms": None,
                "details": (
                    f"Retrieved {len(safe_chunks)} chunks; "
                    f"top_k={safe_plan.get('top_k', 'Unavailable')}"
                ),
            },
            {
                "step": 3,
                "component": "retrieval_evaluation",
                "status": (
                    "success"
                    if retrieval_sufficient
                    else "limited"
                ),
                "duration_ms": None,
                "details": (
                    f"Sufficient={retrieval_sufficient}; "
                    f"retry_performed={retry_performed}; "
                    f"retry_count={retry_count}"
                ),
            },
        ]

        tool_outputs = safe_execution_result.get(
            "tool_outputs",
            [],
        )

        if not isinstance(tool_outputs, list):
            tool_outputs = []

        next_step = 4

        for tool_output in tool_outputs:
            if not isinstance(tool_output, dict):
                continue

            success = bool(
                tool_output.get(
                    "success",
                    False,
                )
            )

            error = tool_output.get(
                "error"
            )

            trace.append(
                {
                    "step": next_step,
                    "component": str(
                        tool_output.get(
                            "tool",
                            "unknown_tool",
                        )
                    ),
                    "status": (
                        "success"
                        if success
                        else "failed"
                    ),
                    "duration_ms": (
                        float(
                            tool_output["duration_ms"]
                        )
                        if isinstance(
                            tool_output.get(
                                "duration_ms"
                            ),
                            (int, float),
                        )
                        else None
                    ),
                    "details": (
                        "Tool completed successfully."
                        if success
                        else (
                            f"Tool failed: {error}"
                            if error
                            else "Tool execution failed."
                        )
                    ),
                }
            )

            next_step += 1

        trace.append(
            {
                "step": next_step,
                "component": "response_generation",
                "status": "success",
                "duration_ms": None,
                "details": (
                    "A deterministic response builder produced "
                    "the final answer."
                    if deterministic_answer_used
                    else (
                        "The language model produced the final "
                        "answer using verified context."
                    )
                ),
            }
        )

        return trace