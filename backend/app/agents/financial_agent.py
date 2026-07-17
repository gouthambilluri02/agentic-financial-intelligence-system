from typing import Any

import ollama

from backend.app.services.company_detector import CompanyDetector
from backend.app.services.conversation_memory import ConversationMemory
from backend.app.services.deterministic_response_builder import (
    DeterministicResponseBuilder,
)
from backend.app.services.execution_planner import ExecutionPlanner
from backend.app.services.financial_metrics import FinancialMetricDetector
from backend.app.services.intent_detector import IntentDetector
from backend.app.services.prompt_builder import PromptBuilder
from backend.app.services.retrieval_orchestrator import (
    RetrievalOrchestrator,
)
from backend.app.services.tool_executor import ToolExecutor
from backend.app.services.tool_router import ToolRouter
from backend.app.tools.tool_registry import ToolRegistry


OLLAMA_MODEL = "qwen2.5:7b"


class FinancialAgent:
    """
    Main orchestrator for the Agentic Financial Intelligence System.

    Workflow:
    1. Detect companies, intent, and financial metric.
    2. Resolve conversation memory.
    3. Select the primary tool.
    4. Retrieve relevant financial-report evidence.
    5. Create a multi-tool execution plan.
    6. Execute every planned tool.
    7. Extract deterministic tool results.
    8. Return deterministic calculation and comparison answers.
    9. Use structured risk-tool evidence for risk questions.
    10. Use Ollama for narrative report questions.
    """

    def __init__(self) -> None:
        self.company_detector = CompanyDetector()
        self.intent_detector = IntentDetector()
        self.metric_detector = FinancialMetricDetector()

        self.tool_router = ToolRouter()
        self.tool_registry = ToolRegistry()
        self.execution_planner = ExecutionPlanner()

        self.tool_executor = ToolExecutor(
            tool_registry=self.tool_registry
        )

        self.retrieval_orchestrator = RetrievalOrchestrator(
            max_retries=1
        )

        self.prompt_builder = PromptBuilder()

        self.deterministic_response_builder = (
            DeterministicResponseBuilder()
        )

        self.memory = ConversationMemory()

    def process_question(
        self,
        question: str,
    ) -> dict[str, Any]:
        """
        Process one financial question through the complete
        agentic workflow.
        """

        if not isinstance(question, str):
            return self._empty_question_response()

        cleaned_question = question.strip()

        if not cleaned_question:
            return self._empty_question_response()

        detected_companies = (
            self.company_detector.detect_companies(
                cleaned_question
            )
        )

        detected_intent = (
            self.intent_detector.detect_intent(
                cleaned_question
            )
        )

        detected_metric = (
            self.metric_detector.detect_metric(
                cleaned_question
            )
        )

        resolved_companies = self._resolve_companies(
            detected_companies
        )

        resolved_intent = self._resolve_intent(
            detected_intent
        )

        selected_tool = self.tool_router.select_tool(
            question=cleaned_question,
            intent=resolved_intent,
            metric=detected_metric,
            companies=resolved_companies,
        )

        retrieval_result = (
            self.retrieval_orchestrator.retrieve(
                question=cleaned_question,
                companies=resolved_companies,
                intent=resolved_intent,
                metric=detected_metric,
            )
        )

        retrieved_chunks = retrieval_result.get(
            "chunks",
            [],
        )

        retrieval_plan = retrieval_result.get(
            "plan",
            {},
        )

        retry_performed = retrieval_result.get(
            "retry_performed",
            False,
        )

        retry_count = retrieval_result.get(
            "retry_count",
            0,
        )

        retrieval_sufficient = retrieval_result.get(
            "retrieval_sufficient",
            False,
        )

        execution_plan = (
            self.execution_planner.create_plan(
                selected_tool=selected_tool,
                intent=resolved_intent,
                companies=resolved_companies,
            )
        )

        execution_result = (
            self.tool_executor.execute_plan(
                execution_plan=execution_plan,
                question=cleaned_question,
                metric=detected_metric,
                companies=resolved_companies,
                retrieved_chunks=retrieved_chunks,
            )
        )

        calculation_result = (
            self._extract_calculation_result(
                execution_result
            )
        )

        comparison_result = (
            self._extract_comparison_result(
                execution_result
            )
        )

        risk_result = (
            self._extract_risk_result(
                execution_result
            )
        )

        sources = self._extract_sources(
            retrieved_chunks
        )

        if not retrieved_chunks:
            self._update_memory(
                question=cleaned_question,
                companies=resolved_companies,
                intent=resolved_intent,
            )

            return self._build_response(
                answer=(
                    "I could not find relevant information in "
                    "the available financial reports."
                ),
                sources=[],
                calculation=calculation_result,
                comparison=comparison_result,
                risk_analysis=risk_result,
                selected_tool=selected_tool,
                execution_plan=execution_plan,
                execution_result=execution_result,
                companies=resolved_companies,
                intent=resolved_intent,
                metric=detected_metric,
                retrieval_plan=retrieval_plan,
                retry_performed=retry_performed,
                retry_count=retry_count,
                retrieval_sufficient=False,
                deterministic_answer_used=False,
            )

        deterministic_answer = (
            self.deterministic_response_builder.build_answer(
                selected_tool=selected_tool,
                calculation=calculation_result,
                comparison=comparison_result,
            )
        )

        if deterministic_answer is not None:
            self._update_memory(
                question=cleaned_question,
                companies=resolved_companies,
                intent=resolved_intent,
            )

            return self._build_response(
                answer=deterministic_answer,
                sources=sources,
                calculation=calculation_result,
                comparison=comparison_result,
                risk_analysis=risk_result,
                selected_tool=selected_tool,
                execution_plan=execution_plan,
                execution_result=execution_result,
                companies=resolved_companies,
                intent=resolved_intent,
                metric=detected_metric,
                retrieval_plan=retrieval_plan,
                retry_performed=retry_performed,
                retry_count=retry_count,
                retrieval_sufficient=retrieval_sufficient,
                deterministic_answer_used=True,
            )

        if (
            selected_tool == "risk_analysis"
            and isinstance(risk_result, dict)
            and risk_result.get("success")
        ):
            risk_answer = self._generate_risk_answer(
                question=cleaned_question,
                risk_result=risk_result,
            )

            risk_sources = risk_result.get(
                "sources",
                [],
            )

            if not isinstance(risk_sources, list):
                risk_sources = []

            self._update_memory(
                question=cleaned_question,
                companies=resolved_companies,
                intent=resolved_intent,
            )

            return self._build_response(
                answer=risk_answer,
                sources=risk_sources or sources,
                calculation=calculation_result,
                comparison=comparison_result,
                risk_analysis=risk_result,
                selected_tool=selected_tool,
                execution_plan=execution_plan,
                execution_result=execution_result,
                companies=resolved_companies,
                intent=resolved_intent,
                metric=detected_metric,
                retrieval_plan=retrieval_plan,
                retry_performed=retry_performed,
                retry_count=retry_count,
                retrieval_sufficient=retrieval_sufficient,
                deterministic_answer_used=False,
            )

        report_context = self._build_context(
            retrieved_chunks
        )

        system_prompt = (
            self.prompt_builder.build_system_prompt(
                resolved_intent
            )
        )

        user_prompt = (
            self.prompt_builder.build_user_prompt(
                question=cleaned_question,
                context=report_context,
                detected_companies=resolved_companies,
                intent=resolved_intent,
            )
        )

        user_prompt = self._add_agent_diagnostics(
            user_prompt=user_prompt,
            selected_tool=selected_tool,
            execution_plan=execution_plan,
            detected_metric=detected_metric,
        )

        user_prompt = self._add_multi_tool_context(
            user_prompt=user_prompt,
            execution_result=execution_result,
        )

        if not retrieval_sufficient:
            user_prompt = (
                "IMPORTANT RETRIEVAL LIMITATION\n"
                "The available financial-report context may be "
                "incomplete. Clearly identify missing information. "
                "Do not invent facts, dates, financial values, "
                "calculations, or sources.\n\n"
                f"{user_prompt}"
            )

        user_prompt = self._add_conversation_context(
            user_prompt
        )

        try:
            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
            )

            answer = self._extract_llm_answer(
                response
            )

        except Exception as error:
            answer = (
                "The financial information was retrieved, but the "
                "language model could not generate the final "
                f"explanation. Reason: {error}"
            )

        self._update_memory(
            question=cleaned_question,
            companies=resolved_companies,
            intent=resolved_intent,
        )

        return self._build_response(
            answer=answer,
            sources=sources,
            calculation=calculation_result,
            comparison=comparison_result,
            risk_analysis=risk_result,
            selected_tool=selected_tool,
            execution_plan=execution_plan,
            execution_result=execution_result,
            companies=resolved_companies,
            intent=resolved_intent,
            metric=detected_metric,
            retrieval_plan=retrieval_plan,
            retry_performed=retry_performed,
            retry_count=retry_count,
            retrieval_sufficient=retrieval_sufficient,
            deterministic_answer_used=False,
        )

    def _generate_risk_answer(
        self,
        question: str,
        risk_result: dict[str, Any],
    ) -> str:
        """
        Generate a focused risk answer using only structured output
        produced by RiskAnalysisTool.
        """

        system_prompt = self.prompt_builder.build_system_prompt(
            "risk_analysis"
        )

        risk_context = risk_result.get(
            "prompt_context",
            "",
        )

        if not isinstance(risk_context, str):
            risk_context = ""

        user_prompt = (
            "USER QUESTION\n\n"
            f"{question}\n\n"
            "VERIFIED RISK TOOL OUTPUT\n\n"
            f"{risk_context}\n\n"
            "FINAL RESPONSE RULES\n\n"
            "- Answer only the risk question.\n"
            "- Use only the verified risk evidence above.\n"
            "- Group related risks under short headings.\n"
            "- Do not discuss revenue, growth, products, strategy, "
            "or unrelated company achievements.\n"
            "- Do not direct the user to websites or external reports.\n"
            "- Do not add generic closing statements.\n"
            "- End immediately after the risk analysis."
        )

        try:
            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
            )

            return self._extract_llm_answer(
                response
            )

        except Exception as error:
            return (
                "The risk evidence was extracted successfully, "
                "but the final explanation could not be generated. "
                f"Reason: {error}"
            )

    @staticmethod
    def _build_response(
        answer: str,
        sources: list[dict],
        calculation: dict[str, Any] | None,
        comparison: dict[str, Any] | None,
        risk_analysis: dict[str, Any] | None,
        selected_tool: str,
        execution_plan: list[str],
        execution_result: dict[str, Any],
        companies: list[str],
        intent: str,
        metric: str | None,
        retrieval_plan: dict[str, Any],
        retry_performed: bool,
        retry_count: int,
        retrieval_sufficient: bool,
        deterministic_answer_used: bool,
    ) -> dict[str, Any]:
        """
        Build one consistent agent response.
        """

        return {
            "answer": answer,
            "sources": sources,
            "calculation": calculation,
            "comparison": comparison,
            "risk_analysis": risk_analysis,
            "selected_tool": selected_tool,
            "execution_plan": execution_plan,
            "executed_tools": execution_result.get(
                "executed_tools",
                [],
            ),
            "successful_tools": execution_result.get(
                "successful_tools",
                [],
            ),
            "failed_tools": execution_result.get(
                "failed_tools",
                [],
            ),
            "tool_outputs": execution_result.get(
                "tool_outputs",
                [],
            ),
            "tool_execution_success": execution_result.get(
                "success",
                False,
            ),
            "tool_execution_duration_ms": execution_result.get(
                "duration_ms",
                0.0,
            ),
            "detected_companies": companies,
            "detected_intent": intent,
            "detected_metric": metric,
            "plan": retrieval_plan,
            "retry_performed": retry_performed,
            "retry_count": retry_count,
            "retrieval_sufficient": retrieval_sufficient,
            "deterministic_answer_used": deterministic_answer_used,
        }

    @staticmethod
    def _extract_calculation_result(
        execution_result: dict[str, Any],
    ) -> dict[str, Any] | None:
        """
        Extract the FinancialCalculatorTool result.
        """

        tool_outputs = execution_result.get(
            "tool_outputs",
            [],
        )

        for step in tool_outputs:
            if step.get("tool") != "financial_calculator":
                continue

            output = step.get("output")

            if not isinstance(output, dict):
                continue

            calculation = output.get(
                "calculation"
            )

            if isinstance(calculation, dict):
                return calculation

        return None

    @staticmethod
    def _extract_comparison_result(
        execution_result: dict[str, Any],
    ) -> dict[str, Any] | None:
        """
        Extract the CompanyComparisonTool result.
        """

        tool_outputs = execution_result.get(
            "tool_outputs",
            [],
        )

        for step in tool_outputs:
            if step.get("tool") != "company_comparison":
                continue

            output = step.get("output")

            if isinstance(output, dict):
                return output

        return None

    @staticmethod
    def _extract_risk_result(
        execution_result: dict[str, Any],
    ) -> dict[str, Any] | None:
        """
        Extract the RiskAnalysisTool result.
        """

        tool_outputs = execution_result.get(
            "tool_outputs",
            [],
        )

        for step in tool_outputs:
            if step.get("tool") != "risk_analysis":
                continue

            output = step.get("output")

            if isinstance(output, dict):
                return output

        return None

    @staticmethod
    def _add_agent_diagnostics(
        user_prompt: str,
        selected_tool: str,
        execution_plan: list[str],
        detected_metric: str | None,
    ) -> str:
        """
        Add routing and planning information to the prompt.
        """

        metric_text = (
            detected_metric
            if detected_metric
            else "No financial metric detected"
        )

        plan_text = (
            ", ".join(execution_plan)
            if execution_plan
            else "No tools planned"
        )

        return (
            "AGENT ROUTING INFORMATION\n"
            f"Primary selected tool: {selected_tool}\n"
            f"Execution plan: {plan_text}\n"
            f"Detected financial metric: {metric_text}\n\n"
            f"{user_prompt}"
        )

    @staticmethod
    def _add_multi_tool_context(
        user_prompt: str,
        execution_result: dict[str, Any],
    ) -> str:
        """
        Add verified tool results to the LLM prompt.
        """

        prompt_context = execution_result.get(
            "prompt_context"
        )

        if (
            not isinstance(prompt_context, str)
            or not prompt_context.strip()
        ):
            return user_prompt

        return (
            "VERIFIED MULTI-TOOL OUTPUT\n"
            "Successful deterministic results are authoritative. "
            "Do not recalculate, replace, or contradict verified "
            "results.\n\n"
            f"{prompt_context.strip()}\n\n"
            "USER REQUEST AND REPORT CONTEXT\n"
            f"{user_prompt}"
        )

    def _resolve_companies(
        self,
        newly_detected_companies: list[str],
    ) -> list[str]:
        """
        Use companies detected in the current question.

        If none are detected, reuse companies stored in memory.
        """

        if newly_detected_companies:
            return newly_detected_companies

        remembered_companies = (
            self.memory.get_last_companies()
        )

        return remembered_companies or []

    def _resolve_intent(
        self,
        newly_detected_intent: str,
    ) -> str:
        """
        Reuse the previous intent for vague follow-up questions.
        """

        previous_intent = (
            self.memory.get_last_intent()
        )

        if (
            newly_detected_intent == "general_question"
            and previous_intent
        ):
            return previous_intent

        return newly_detected_intent

    def _add_conversation_context(
        self,
        user_prompt: str,
    ) -> str:
        """
        Add the previous question for conversational continuity.
        """

        previous_question = (
            self.memory.get_last_question()
        )

        if not previous_question:
            return user_prompt

        return (
            "PREVIOUS USER QUESTION\n"
            f"{previous_question}\n\n"
            "CURRENT REQUEST\n"
            f"{user_prompt}"
        )

    def _update_memory(
        self,
        question: str,
        companies: list[str],
        intent: str,
    ) -> None:
        """
        Store the latest conversation state.
        """

        self.memory.update(
            question=question,
            companies=companies,
            intent=intent,
        )

    @staticmethod
    def _build_context(
        retrieved_chunks: list[dict],
    ) -> str:
        """
        Convert retrieved chunks into structured report context.
        """

        context_sections: list[str] = []

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

            ticker = metadata.get(
                "ticker",
                "Unknown ticker",
            )

            fiscal_year = metadata.get(
                "fiscal_year",
                "Unknown year",
            )

            document_type = metadata.get(
                "document_type",
                "Unknown document type",
            )

            source_file = metadata.get(
                "source_file",
                "Unknown source",
            )

            stored_page = metadata.get(
                "page",
                "Unknown page",
            )

            display_page = stored_page

            if isinstance(stored_page, int):
                display_page = stored_page + 1

            distance = chunk.get(
                "distance",
                "Unknown distance",
            )

            context_sections.append(
                (
                    f"Context {index}\n"
                    f"Company: {company}\n"
                    f"Ticker: {ticker}\n"
                    f"Fiscal year: {fiscal_year}\n"
                    f"Document type: {document_type}\n"
                    f"Source file: {source_file}\n"
                    f"Page: {display_page}\n"
                    f"Retrieval distance: {distance}\n\n"
                    f"Content:\n{content}"
                )
            )

        return "\n\n".join(
            context_sections
        )

    @staticmethod
    def _extract_sources(
        retrieved_chunks: list[dict],
    ) -> list[dict]:
        """
        Extract unique verified source references.
        """

        sources: list[dict] = []
        seen_sources: set[tuple] = set()

        for chunk in retrieved_chunks:
            metadata = chunk.get(
                "metadata",
                {},
            )

            company = metadata.get(
                "company",
                "Unknown company",
            )

            ticker = metadata.get(
                "ticker",
                "Unknown ticker",
            )

            source_file = metadata.get(
                "source_file",
                "Unknown source",
            )

            fiscal_year = metadata.get(
                "fiscal_year",
                "Unknown year",
            )

            document_type = metadata.get(
                "document_type",
                "Unknown document type",
            )

            stored_page = metadata.get(
                "page",
                "Unknown page",
            )

            display_page = stored_page

            if isinstance(stored_page, int):
                display_page = stored_page + 1

            source_key = (
                company,
                ticker,
                source_file,
                fiscal_year,
                document_type,
                display_page,
            )

            if source_key in seen_sources:
                continue

            sources.append(
                {
                    "company": company,
                    "ticker": ticker,
                    "source_file": source_file,
                    "fiscal_year": fiscal_year,
                    "document_type": document_type,
                    "page": display_page,
                }
            )

            seen_sources.add(
                source_key
            )

        return sources

    @staticmethod
    def _extract_llm_answer(
        response: Any,
    ) -> str:
        """
        Safely extract text from an Ollama response.
        """

        try:
            if isinstance(response, dict):
                message = response.get(
                    "message",
                    {},
                )

                answer = message.get(
                    "content"
                )

            else:
                answer = response.message.content

        except (
            AttributeError,
            KeyError,
            TypeError,
        ):
            return (
                "The language model returned an unexpected response."
            )

        if not isinstance(answer, str):
            return (
                "The language model returned an invalid answer."
            )

        return answer.strip()

    @staticmethod
    def _empty_question_response() -> dict[str, Any]:
        """
        Return a consistent response for an empty question.
        """

        return {
            "answer": "Please enter a financial question.",
            "sources": [],
            "calculation": None,
            "comparison": None,
            "risk_analysis": None,
            "selected_tool": "document_retrieval",
            "execution_plan": [],
            "executed_tools": [],
            "successful_tools": [],
            "failed_tools": [],
            "tool_outputs": [],
            "tool_execution_success": False,
            "tool_execution_duration_ms": 0.0,
            "detected_companies": [],
            "detected_intent": "general_question",
            "detected_metric": None,
            "plan": {},
            "retry_performed": False,
            "retry_count": 0,
            "retrieval_sufficient": False,
            "deterministic_answer_used": False,
        }

    def clear_memory(self) -> None:
        """
        Clear the current in-memory conversation state.
        """

        self.memory.clear()


if __name__ == "__main__":
    financial_agent = FinancialAgent()

    test_questions = [
        "Calculate Apple's revenue growth.",
        "Compare Apple and Microsoft revenue.",
        "What risks did Microsoft disclose?",
    ]

    for test_question in test_questions:
        result = financial_agent.process_question(
            test_question
        )

        print("\n" + "=" * 80)
        print(f"Question:\n{test_question}")

        print(
            "\nDetected companies:\n"
            f"{result['detected_companies']}"
        )

        print(
            "\nDetected intent:\n"
            f"{result['detected_intent']}"
        )

        print(
            "\nDetected metric:\n"
            f"{result['detected_metric']}"
        )

        print(
            "\nPrimary selected tool:\n"
            f"{result['selected_tool']}"
        )

        print(
            "\nDeterministic answer used:\n"
            f"{result['deterministic_answer_used']}"
        )

        print("\nExecution plan:")

        for step_number, tool_name in enumerate(
            result.get(
                "execution_plan",
                [],
            ),
            start=1,
        ):
            print(
                f"{step_number}. {tool_name}"
            )

        print("\nExecuted tools:")
        print(
            result.get(
                "executed_tools",
                [],
            )
        )

        print("\nSuccessful tools:")
        print(
            result.get(
                "successful_tools",
                [],
            )
        )

        print("\nFailed tools:")
        print(
            result.get(
                "failed_tools",
                [],
            )
        )

        print(
            "\nRisk analysis result:\n"
            f"{result.get('risk_analysis')}"
        )

        print(
            f"\nAnswer:\n"
            f"{result['answer']}"
        )

        print("\nSources:")

        for source in result.get(
            "sources",
            [],
        ):
            print(
                f"- {source.get('company')} | "
                f"{source.get('source_file')} | "
                f"Page {source.get('page')}"
            )