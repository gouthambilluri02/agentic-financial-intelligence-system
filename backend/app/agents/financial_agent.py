from __future__ import annotations

from typing import Any

import ollama

from backend.app.services.company_detector import CompanyDetector
from backend.app.services.confidence_scoring_service import (
    ConfidenceScoringService,
)
from backend.app.services.conversation_memory import ConversationMemory
from backend.app.services.deterministic_response_builder import (
    DeterministicResponseBuilder,
)
from backend.app.services.execution_planner import ExecutionPlanner
from backend.app.services.execution_trace_service import (
    ExecutionTraceService,
)
from backend.app.services.financial_metrics import FinancialMetricDetector
from backend.app.services.intent_detector import IntentDetector
from backend.app.services.prompt_builder import PromptBuilder
from backend.app.services.retrieval_orchestrator import (
    RetrievalOrchestrator,
)
from backend.app.services.source_ranking_service import (
    SourceRankingService,
)
from backend.app.services.tool_executor import ToolExecutor
from backend.app.services.tool_router import ToolRouter
from backend.app.tools.tool_registry import ToolRegistry


OLLAMA_MODEL = "qwen2.5:7b"


class FinancialAgent:
    """
    Main orchestrator for the Agentic Financial Intelligence System.

    Workflow:
    1. Validate and clean the user question.
    2. Detect companies, intent, and financial metric.
    3. Resolve conversational context using memory.
    4. Select the primary analysis tool.
    5. Retrieve relevant financial-report evidence.
    6. Build and execute the multi-tool plan.
    7. Extract structured tool outputs.
    8. Rank verified report sources.
    9. Generate a deterministic or language-model answer.
    10. Calculate evidence-based confidence.
    11. Build a safe execution trace.
    12. Return one consistent structured response.
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

        self.source_ranking_service = (
            SourceRankingService()
        )

        self.confidence_scoring_service = (
            ConfidenceScoringService()
        )

        self.execution_trace_service = (
            ExecutionTraceService()
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

        if not isinstance(retrieved_chunks, list):
            retrieved_chunks = []

        retrieval_plan = retrieval_result.get(
            "plan",
            {},
        )

        if not isinstance(retrieval_plan, dict):
            retrieval_plan = {}

        retry_performed = bool(
            retrieval_result.get(
                "retry_performed",
                False,
            )
        )

        retry_count = retrieval_result.get(
            "retry_count",
            0,
        )

        if not isinstance(retry_count, int):
            retry_count = 0

        retrieval_sufficient = bool(
            retrieval_result.get(
                "retrieval_sufficient",
                False,
            )
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

        if not isinstance(execution_result, dict):
            execution_result = {
                "success": False,
                "execution_plan": execution_plan,
                "executed_tools": [],
                "successful_tools": [],
                "failed_tools": execution_plan,
                "tool_outputs": [],
                "prompt_context": (
                    "Tool execution returned an invalid result."
                ),
                "duration_ms": 0.0,
                "error": (
                    "ToolExecutor must return a dictionary."
                ),
            }

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

        ranked_sources = (
            self.source_ranking_service.rank_sources(
                retrieved_chunks
            )
        )

        if not retrieved_chunks:
            answer = (
                "I could not find relevant information in "
                "the available financial reports."
            )

            self._update_memory(
                question=cleaned_question,
                companies=resolved_companies,
                intent=resolved_intent,
            )

            return self._build_response(
                answer=answer,
                sources=[],
                retrieved_chunks=retrieved_chunks,
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
                response_generation_status="limited",
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
                sources=ranked_sources,
                retrieved_chunks=retrieved_chunks,
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
                response_generation_status="success",
            )

        if (
            selected_tool == "risk_analysis"
            and isinstance(risk_result, dict)
            and risk_result.get("success")
        ):
            risk_answer, generation_status = (
                self._generate_risk_answer(
                    question=cleaned_question,
                    risk_result=risk_result,
                )
            )

            risk_sources = risk_result.get(
                "sources",
                [],
            )

            if not isinstance(risk_sources, list):
                risk_sources = []

            final_risk_sources = (
                self._rank_tool_sources(
                    risk_sources=risk_sources,
                    ranked_sources=ranked_sources,
                )
            )

            self._update_memory(
                question=cleaned_question,
                companies=resolved_companies,
                intent=resolved_intent,
            )

            return self._build_response(
                answer=risk_answer,
                sources=final_risk_sources,
                retrieved_chunks=retrieved_chunks,
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
                response_generation_status=generation_status,
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

        response_generation_status = "success"

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

            if answer in {
                "The language model returned an unexpected response.",
                "The language model returned an invalid answer.",
            }:
                response_generation_status = "failed"

        except Exception as error:
            response_generation_status = "failed"

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
            sources=ranked_sources,
            retrieved_chunks=retrieved_chunks,
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
            response_generation_status=response_generation_status,
        )

    def _generate_risk_answer(
        self,
        question: str,
        risk_result: dict[str, Any],
    ) -> tuple[str, str]:
        """
        Generate a focused risk answer using only structured output
        produced by RiskAnalysisTool.

        Returns:
        - generated answer
        - response-generation status
        """

        system_prompt = (
            self.prompt_builder.build_system_prompt(
                "risk_analysis"
            )
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

            answer = self._extract_llm_answer(
                response
            )

            if answer in {
                "The language model returned an unexpected response.",
                "The language model returned an invalid answer.",
            }:
                return answer, "failed"

            return answer, "success"

        except Exception as error:
            return (
                "The risk evidence was extracted successfully, "
                "but the final explanation could not be generated. "
                f"Reason: {error}",
                "failed",
            )

    def _build_response(
        self,
        answer: str,
        sources: list[dict[str, Any]],
        retrieved_chunks: list[dict[str, Any]],
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
        response_generation_status: str,
    ) -> dict[str, Any]:
        """
        Build one consistent agent response.

        Confidence and execution trace are generated here so every
        response path uses the same intelligence metadata.
        """

        confidence = (
            self.confidence_scoring_service.calculate(
                retrieved_chunks=retrieved_chunks,
                retrieval_sufficient=retrieval_sufficient,
                retry_count=retry_count,
                execution_result=execution_result,
                deterministic_answer_used=(
                    deterministic_answer_used
                ),
            )
        )

        execution_trace = (
            self.execution_trace_service.build_trace(
                selected_tool=selected_tool,
                retrieval_plan=retrieval_plan,
                retrieved_chunks=retrieved_chunks,
                retry_performed=retry_performed,
                retry_count=retry_count,
                retrieval_sufficient=(
                    retrieval_sufficient
                ),
                execution_result=execution_result,
                deterministic_answer_used=(
                    deterministic_answer_used
                ),
            )
        )

        execution_trace = (
            self._apply_response_generation_status(
                execution_trace=execution_trace,
                status=response_generation_status,
            )
        )

        return {
            "answer": answer,
            "confidence": confidence,
            "sources": sources,
            "execution_trace": execution_trace,
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
            "deterministic_answer_used": (
                deterministic_answer_used
            ),
        }

    @staticmethod
    def _apply_response_generation_status(
        execution_trace: list[dict[str, Any]],
        status: str,
    ) -> list[dict[str, Any]]:
        """
        Correct the final trace step when response generation was
        limited or failed.
        """

        if not execution_trace:
            return execution_trace

        normalized_status = (
            status
            if status in {
                "success",
                "limited",
                "failed",
            }
            else "success"
        )

        final_step = execution_trace[-1]

        if (
            isinstance(final_step, dict)
            and final_step.get("component")
            == "response_generation"
        ):
            final_step["status"] = normalized_status

            if normalized_status == "failed":
                final_step["details"] = (
                    "The system retrieved evidence, but final "
                    "response generation failed."
                )

            elif normalized_status == "limited":
                final_step["details"] = (
                    "The system returned a limited response because "
                    "relevant report evidence was unavailable."
                )

        return execution_trace

    @staticmethod
    def _rank_tool_sources(
        risk_sources: list[dict[str, Any]],
        ranked_sources: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Match RiskAnalysisTool sources with ranked retrieval sources.

        This preserves the tool's verified source selection while
        adding rank, retrieval distance, and relevance score when a
        matching retrieved source exists.
        """

        if not risk_sources:
            return ranked_sources

        ranked_lookup: dict[
            tuple[Any, ...],
            dict[str, Any],
        ] = {}

        for source in ranked_sources:
            key = (
                source.get("company"),
                source.get("source_file"),
                source.get("page"),
            )

            ranked_lookup[key] = source

        enriched_sources: list[
            dict[str, Any]
        ] = []

        seen_sources: set[
            tuple[Any, ...]
        ] = set()

        for source in risk_sources:
            if not isinstance(source, dict):
                continue

            company = source.get(
                "company"
            )

            source_file = source.get(
                "source_file"
            )

            page = source.get(
                "page"
            )

            key = (
                company,
                source_file,
                page,
            )

            matched_source = ranked_lookup.get(
                key
            )

            if matched_source:
                enriched_source = {
                    **source,
                    "relevance_score": (
                        matched_source.get(
                            "relevance_score"
                        )
                    ),
                    "retrieval_distance": (
                        matched_source.get(
                            "retrieval_distance"
                        )
                    ),
                    "rank": matched_source.get(
                        "rank"
                    ),
                }
            else:
                enriched_source = {
                    **source,
                    "relevance_score": None,
                    "retrieval_distance": None,
                    "rank": None,
                }

            duplicate_key = (
                enriched_source.get(
                    "company"
                ),
                enriched_source.get(
                    "ticker"
                ),
                enriched_source.get(
                    "source_file"
                ),
                enriched_source.get(
                    "fiscal_year"
                ),
                enriched_source.get(
                    "document_type"
                ),
                enriched_source.get(
                    "page"
                ),
            )

            if duplicate_key in seen_sources:
                continue

            enriched_sources.append(
                enriched_source
            )

            seen_sources.add(
                duplicate_key
            )

        enriched_sources.sort(
            key=lambda source: (
                source.get("rank")
                if isinstance(
                    source.get("rank"),
                    int,
                )
                else float("inf")
            )
        )

        return enriched_sources

    @staticmethod
    def _extract_calculation_result(
        execution_result: dict[str, Any],
    ) -> dict[str, Any] | None:
        """
        Extract the FinancialCalculatorTool calculation result.
        """

        tool_outputs = execution_result.get(
            "tool_outputs",
            [],
        )

        if not isinstance(tool_outputs, list):
            return None

        for step in tool_outputs:
            if not isinstance(step, dict):
                continue

            if step.get("tool") != "financial_calculator":
                continue

            output = step.get(
                "output"
            )

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

        if not isinstance(tool_outputs, list):
            return None

        for step in tool_outputs:
            if not isinstance(step, dict):
                continue

            if step.get("tool") != "company_comparison":
                continue

            output = step.get(
                "output"
            )

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

        if not isinstance(tool_outputs, list):
            return None

        for step in tool_outputs:
            if not isinstance(step, dict):
                continue

            if step.get("tool") != "risk_analysis":
                continue

            output = step.get(
                "output"
            )

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
        retrieved_chunks: list[dict[str, Any]],
    ) -> str:
        """
        Convert retrieved chunks into structured report context.
        """

        context_sections: list[str] = []

        for index, chunk in enumerate(
            retrieved_chunks,
            start=1,
        ):
            if not isinstance(chunk, dict):
                continue

            metadata = chunk.get(
                "metadata",
                {},
            )

            if not isinstance(metadata, dict):
                metadata = {}

            content = chunk.get(
                "content",
                "",
            )

            if not isinstance(content, str):
                content = str(content)

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

        cleaned_answer = answer.strip()

        if not cleaned_answer:
            return (
                "The language model returned an invalid answer."
            )

        return cleaned_answer

    @staticmethod
    def _empty_question_response() -> dict[str, Any]:
        """
        Return a consistent response for an empty question.
        """

        return {
            "answer": "Please enter a financial question.",
            "confidence": {
                "score": 0.0,
                "percentage": 0.0,
                "level": "low",
                "reasons": [
                    "No valid financial question was provided."
                ],
                "components": {
                    "retrieval_relevance": 0.0,
                    "evidence_coverage": 0.0,
                    "retrieval_sufficiency": 0.0,
                    "tool_execution": 0.0,
                    "deterministic_support": 0.0,
                    "retry_stability": 0.0,
                },
                "method": (
                    "Evidence-quality heuristic based on retrieval, "
                    "tool execution, and deterministic support."
                ),
            },
            "sources": [],
            "execution_trace": [],
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

        print(
            "\nConfidence:\n"
            f"{result.get('confidence')}"
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

        print("\nExecution trace:")

        for trace_step in result.get(
            "execution_trace",
            [],
        ):
            print(
                f"{trace_step.get('step')}. "
                f"{trace_step.get('component')} | "
                f"{trace_step.get('status')} | "
                f"{trace_step.get('details')}"
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

        print("\nRanked sources:")

        for source in result.get(
            "sources",
            [],
        ):
            print(
                f"- Rank {source.get('rank')} | "
                f"{source.get('company')} | "
                f"{source.get('source_file')} | "
                f"Page {source.get('page')} | "
                f"Relevance "
                f"{source.get('relevance_score')}"
            )