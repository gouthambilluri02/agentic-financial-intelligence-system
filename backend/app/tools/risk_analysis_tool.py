from __future__ import annotations

import re
from typing import Any

from backend.app.tools.base_tool import BaseTool


class RiskAnalysisTool(BaseTool):
    """
    Extract and organize disclosed risk evidence from retrieved
    financial-report chunks.

    The tool does not invent risks or make unsupported predictions.
    It only categorizes risk statements found in the retrieved text.
    """

    name = "risk_analysis"

    description = (
        "Identifies and organizes disclosed business risks from retrieved "
        "financial-report passages, including cybersecurity, regulatory, "
        "operational, financial, supply-chain, competition, privacy, legal, "
        "macroeconomic, tax, technology, and reputation risks."
    )

    RISK_CATEGORIES: dict[str, tuple[str, ...]] = {
        "Cybersecurity and Information Security": (
            "cybersecurity",
            "cyber security",
            "security breach",
            "data breach",
            "cyberattack",
            "cyber attack",
            "malware",
            "ransomware",
            "security incident",
            "unauthorized access",
            "information security",
            "threat actor",
            "security vulnerability",
            "network security",
        ),
        "Privacy and Data Protection": (
            "privacy",
            "personal data",
            "data protection",
            "user data",
            "customer data",
            "data processing",
            "privacy regulation",
            "privacy law",
            "consent requirements",
            "data storage",
        ),
        "Regulatory and Compliance": (
            "regulation",
            "regulatory",
            "compliance",
            "antitrust",
            "competition law",
            "government investigation",
            "government inquiry",
            "regulatory authority",
            "regulatory requirements",
            "digital markets act",
            "digital services act",
        ),
        "Legal and Litigation": (
            "litigation",
            "lawsuit",
            "legal proceeding",
            "legal claim",
            "legal liability",
            "intellectual property claim",
            "patent claim",
            "court",
            "settlement",
            "legal dispute",
        ),
        "Competition": (
            "competition",
            "competitor",
            "competitive",
            "market share",
            "pricing pressure",
            "competitive pressure",
            "rapidly evolving market",
            "compete effectively",
        ),
        "Supply Chain and Third Parties": (
            "supply chain",
            "supplier",
            "vendor",
            "third party",
            "third-party",
            "component shortage",
            "manufacturing partner",
            "logistics",
            "availability of components",
            "business partner",
        ),
        "Operational and Business Continuity": (
            "operational risk",
            "business continuity",
            "service interruption",
            "system failure",
            "outage",
            "disruption",
            "business interruption",
            "availability",
            "reliability",
            "infrastructure failure",
        ),
        "Financial and Market": (
            "foreign currency",
            "exchange rate",
            "interest rate",
            "credit risk",
            "liquidity",
            "market risk",
            "financial condition",
            "impairment",
            "investment loss",
            "cash flow",
            "financing",
        ),
        "Macroeconomic and Geopolitical": (
            "macroeconomic",
            "economic conditions",
            "inflation",
            "recession",
            "geopolitical",
            "war",
            "trade restriction",
            "tariff",
            "sanction",
            "political instability",
            "global economy",
        ),
        "Technology and Artificial Intelligence": (
            "artificial intelligence",
            "responsible ai",
            "generative ai",
            "machine learning",
            "technology change",
            "technological change",
            "innovation",
            "ai regulation",
            "ai systems",
            "emerging technology",
        ),
        "Product and Service Quality": (
            "product defect",
            "service quality",
            "product quality",
            "software defect",
            "errors",
            "bugs",
            "product failure",
            "customer dissatisfaction",
            "performance issue",
        ),
        "Reputation and Customer Trust": (
            "reputation",
            "customer trust",
            "brand",
            "public perception",
            "negative publicity",
            "loss of trust",
            "customer confidence",
        ),
        "Tax": (
            "tax risk",
            "tax law",
            "tax regulation",
            "tax authority",
            "tax position",
            "tax liability",
            "tax examination",
            "tax assessment",
            "uncertain tax",
        ),
        "Human Capital": (
            "key personnel",
            "employee retention",
            "talent",
            "workforce",
            "labor",
            "personnel",
            "recruit",
            "retain employees",
            "human capital",
        ),
    }

    GENERAL_RISK_TERMS: tuple[str, ...] = (
        "risk",
        "risks",
        "could adversely affect",
        "may adversely affect",
        "material adverse effect",
        "could harm",
        "may harm",
        "uncertainty",
        "threat",
        "challenge",
        "exposure",
        "vulnerability",
        "failure to",
        "unable to",
    )

    def run(
        self,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Analyze retrieved report chunks for disclosed risks.

        Expected keyword arguments:
        - question
        - companies
        - retrieved_chunks
        """

        question = kwargs.get("question")
        companies = kwargs.get("companies", [])
        retrieved_chunks = kwargs.get(
            "retrieved_chunks",
            [],
        )

        validation_error = self._validate_inputs(
            question=question,
            companies=companies,
            retrieved_chunks=retrieved_chunks,
        )

        if validation_error:
            return self._failure(
                validation_error
            )

        risk_items = self._extract_risk_items(
            retrieved_chunks=retrieved_chunks,
        )

        if not risk_items:
            return self._failure(
                "No clearly disclosed risk statements were identified "
                "in the retrieved report passages."
            )

        grouped_risks = self._group_by_category(
            risk_items
        )

        sources = self._extract_sources(
            risk_items
        )

        result: dict[str, Any] = {
            "success": True,
            "tool": self.name,
            "companies": companies,
            "risk_categories": grouped_risks,
            "risk_count": len(risk_items),
            "category_count": len(grouped_risks),
            "sources": sources,
            "prompt_context": "",
            "error": None,
        }

        result["prompt_context"] = (
            self.build_prompt_context(
                result
            )
        )

        return result

    def _extract_risk_items(
        self,
        retrieved_chunks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Find risk-related sentences and assign categories.
        """

        risk_items: list[dict[str, Any]] = []
        seen_items: set[
            tuple[str, str, Any]
        ] = set()

        for chunk in retrieved_chunks:
            content = chunk.get(
                "content",
                "",
            )

            metadata = chunk.get(
                "metadata",
                {},
            )

            if (
                not isinstance(content, str)
                or not content.strip()
            ):
                continue

            if not isinstance(metadata, dict):
                metadata = {}

            sentences = self._split_sentences(
                content
            )

            for sentence in sentences:
                normalized_sentence = (
                    self._normalize_text(
                        sentence
                    )
                )

                categories = (
                    self._detect_categories(
                        normalized_sentence
                    )
                )

                general_risk_match = (
                    self._contains_any(
                        normalized_sentence,
                        self.GENERAL_RISK_TERMS,
                    )
                )

                if (
                    not categories
                    and not general_risk_match
                ):
                    continue

                if not categories:
                    categories = [
                        "Other Disclosed Business Risk"
                    ]

                cleaned_sentence = (
                    self._clean_sentence(
                        sentence
                    )
                )

                if len(cleaned_sentence) < 35:
                    continue

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

                for category in categories:
                    dedupe_key = (
                        category,
                        cleaned_sentence.lower(),
                        page,
                    )

                    if dedupe_key in seen_items:
                        continue

                    risk_items.append(
                        {
                            "category": category,
                            "company": company,
                            "statement": cleaned_sentence,
                            "source_file": source_file,
                            "page": page,
                            "fiscal_year": metadata.get(
                                "fiscal_year"
                            ),
                            "document_type": metadata.get(
                                "document_type"
                            ),
                            "ticker": metadata.get(
                                "ticker"
                            ),
                            "distance": chunk.get(
                                "distance"
                            ),
                        }
                    )

                    seen_items.add(
                        dedupe_key
                    )

        return self._rank_and_limit(
            risk_items
        )

    def _detect_categories(
        self,
        normalized_sentence: str,
    ) -> list[str]:
        """
        Match a sentence against configured risk categories.
        """

        detected: list[str] = []

        for (
            category,
            keywords,
        ) in self.RISK_CATEGORIES.items():
            if self._contains_any(
                normalized_sentence,
                keywords,
            ):
                detected.append(
                    category
                )

        return detected

    @staticmethod
    def _contains_any(
        text: str,
        keywords: tuple[str, ...],
    ) -> bool:
        """
        Return True when at least one keyword appears in the text.
        """

        return any(
            keyword in text
            for keyword in keywords
        )

    @staticmethod
    def _split_sentences(
        content: str,
    ) -> list[str]:
        """
        Split report text into sentence-like units.

        Also handles bullets and line breaks.
        """

        prepared = re.sub(
            r"\s*[\u2022•]\s*",
            ". ",
            content,
        )

        prepared = re.sub(
            r"\n+",
            ". ",
            prepared,
        )

        parts = re.split(
            r"(?<=[.!?])\s+",
            prepared,
        )

        return [
            part.strip()
            for part in parts
            if part.strip()
        ]

    @staticmethod
    def _normalize_text(
        text: str,
    ) -> str:
        """
        Normalize text for keyword matching.
        """

        normalized = text.lower()

        normalized = re.sub(
            r"\s+",
            " ",
            normalized,
        )

        return normalized.strip()

    @staticmethod
    def _clean_sentence(
        sentence: str,
    ) -> str:
        """
        Clean one extracted report sentence.
        """

        cleaned = re.sub(
            r"\s+",
            " ",
            sentence,
        ).strip()

        cleaned = cleaned.strip(
            " -•\t"
        )

        if len(cleaned) > 700:
            cleaned = (
                cleaned[:697].rstrip()
                + "..."
            )

        return cleaned

    def _rank_and_limit(
        self,
        risk_items: list[dict[str, Any]],
        max_items_per_category: int = 3,
        max_categories: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Keep only the strongest risk categories and evidence.

        Maximum output:
        - 5 risk categories
        - 3 evidence statements per category
        """

        grouped: dict[
            str,
            list[dict[str, Any]],
        ] = {}

        for item in risk_items:
            grouped.setdefault(
                item["category"],
                [],
            ).append(
                item
            )

        category_groups: list[
            tuple[
                str,
                list[dict[str, Any]],
            ]
        ] = []

        for (
            category,
            items,
        ) in grouped.items():
            ranked_items = sorted(
                items,
                key=self._risk_item_score,
                reverse=True,
            )

            category_groups.append(
                (
                    category,
                    ranked_items[
                        :max_items_per_category
                    ],
                )
            )

        category_groups.sort(
            key=lambda group: max(
                self._risk_item_score(
                    item
                )
                for item in group[1]
            ),
            reverse=True,
        )

        final_items: list[
            dict[str, Any]
        ] = []

        for (
            _,
            items,
        ) in category_groups[
            :max_categories
        ]:
            final_items.extend(
                items
            )

        return final_items

    @staticmethod
    def _risk_item_score(
        item: dict[str, Any],
    ) -> float:
        """
        Rank risk evidence using statement strength and
        retrieval distance.
        """

        statement = str(
            item.get(
                "statement",
                "",
            )
        ).lower()

        score = 0.0

        strong_phrases = (
            "adversely affect",
            "material adverse",
            "could harm",
            "may harm",
            "could result",
            "may result",
            "failure to",
            "unable to",
            "could impact",
            "may impact",
        )

        for phrase in strong_phrases:
            if phrase in statement:
                score += 2.0

        distance = item.get(
            "distance"
        )

        if isinstance(
            distance,
            (int, float),
        ):
            score += max(
                0.0,
                1.0 - float(distance),
            )

        score += min(
            len(statement) / 500,
            1.0,
        )

        return score

    @staticmethod
    def _group_by_category(
        risk_items: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Convert risk items into structured category groups.
        """

        grouped: dict[
            str,
            list[dict[str, Any]],
        ] = {}

        for item in risk_items:
            grouped.setdefault(
                item["category"],
                [],
            ).append(
                {
                    "company": item[
                        "company"
                    ],
                    "statement": item[
                        "statement"
                    ],
                    "source_file": item[
                        "source_file"
                    ],
                    "page": item[
                        "page"
                    ],
                }
            )

        return [
            {
                "category": category,
                "evidence": evidence,
                "evidence_count": len(
                    evidence
                ),
            }
            for (
                category,
                evidence,
            ) in grouped.items()
        ]

    @staticmethod
    def _extract_sources(
        risk_items: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Return unique source references for identified risks.
        """

        sources: list[
            dict[str, Any]
        ] = []

        seen: set[
            tuple[Any, ...]
        ] = set()

        for item in risk_items:
            source_key = (
                item.get("company"),
                item.get("ticker"),
                item.get("source_file"),
                item.get("fiscal_year"),
                item.get("document_type"),
                item.get("page"),
            )

            if source_key in seen:
                continue

            sources.append(
                {
                    "company": item.get(
                        "company"
                    ),
                    "ticker": item.get(
                        "ticker"
                    ),
                    "source_file": item.get(
                        "source_file"
                    ),
                    "fiscal_year": item.get(
                        "fiscal_year"
                    ),
                    "document_type": item.get(
                        "document_type"
                    ),
                    "page": item.get(
                        "page"
                    ),
                }
            )

            seen.add(
                source_key
            )

        return sources

    @staticmethod
    def build_prompt_context(
        result: dict[str, Any],
    ) -> str:
        """
        Build focused, verified risk evidence for the language model.
        """

        if not result.get(
            "success"
        ):
            return (
                "Verified risk analysis was unavailable.\n"
                f"Reason: {result.get('error')}"
            )

        sections = [
            "VERIFIED STRUCTURED RISK EVIDENCE",
            "",
            (
                "Use only the risk categories and disclosed "
                "statements listed below."
            ),
            (
                "Do not add unrelated financial highlights, "
                "products, strategies, websites, or general "
                "company information."
            ),
            "",
        ]

        for category_group in result.get(
            "risk_categories",
            [],
        ):
            category = category_group.get(
                "category",
                "Other Risk",
            )

            sections.append(
                f"Risk category: {category}"
            )

            for evidence in category_group.get(
                "evidence",
                [],
            ):
                sections.extend(
                    [
                        (
                            "- Disclosed statement: "
                            f"{evidence.get('statement')}"
                        ),
                        (
                            "  Source: "
                            f"{evidence.get('source_file')}, "
                            f"page {evidence.get('page')}"
                        ),
                    ]
                )

            sections.append("")

        sections.extend(
            [
                "Response requirements:",
                "- Return no more than 5 risk categories.",
                "- Answer only the user's risk question.",
                "- Use one short heading per category.",
                "- Provide one concise description per category.",
                "- Provide one concise possible business impact.",
                "- Do not repeat similar risks.",
                "- Do not invent severity rankings.",
                "- Do not include unrelated revenue or product details.",
                "- Do not direct the user to external websites.",
                "- Do not add a generic conclusion.",
                "- End immediately after the final risk category.",
            ]
        )

        return "\n".join(
            sections
        )

    @staticmethod
    def _validate_inputs(
        question: Any,
        companies: Any,
        retrieved_chunks: Any,
    ) -> str | None:
        """
        Validate inputs supplied by ToolExecutor.
        """

        if (
            not isinstance(question, str)
            or not question.strip()
        ):
            return (
                "A non-empty risk-analysis question is required."
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

        if not retrieved_chunks:
            return (
                "No retrieved financial-report passages were "
                "supplied to the risk-analysis tool."
            )

        return None

    def _failure(
        self,
        message: str,
    ) -> dict[str, Any]:
        """
        Return a consistent failed tool result.
        """

        return {
            "success": False,
            "tool": self.name,
            "companies": [],
            "risk_categories": [],
            "risk_count": 0,
            "category_count": 0,
            "sources": [],
            "prompt_context": (
                "Verified risk analysis was unavailable.\n"
                f"Reason: {message}"
            ),
            "error": message,
        }


if __name__ == "__main__":
    tool = RiskAnalysisTool()

    test_chunks = [
        {
            "content": (
                "Cybersecurity threats and security incidents could "
                "adversely affect our systems, customers, and reputation. "
                "We are also subject to changing privacy and data protection "
                "laws. Failure to comply with regulatory requirements could "
                "result in fines, litigation, or operational restrictions."
            ),
            "metadata": {
                "company": "Microsoft",
                "ticker": "MSFT",
                "source_file": "microsoft_2024_10k.pdf",
                "fiscal_year": 2024,
                "document_type": "10K",
                "page": 25,
            },
            "distance": 0.22,
        },
        {
            "content": (
                "Our business faces intense competition. Service outages, "
                "supplier disruptions, and geopolitical instability may "
                "harm operating results."
            ),
            "metadata": {
                "company": "Microsoft",
                "ticker": "MSFT",
                "source_file": "microsoft_2024_10k.pdf",
                "fiscal_year": 2024,
                "document_type": "10K",
                "page": 26,
            },
            "distance": 0.28,
        },
        {
            "content": (
                "Changes in tax laws and uncertain tax positions may result "
                "in additional tax liabilities. Rapid changes in artificial "
                "intelligence regulation could increase compliance costs."
            ),
            "metadata": {
                "company": "Microsoft",
                "ticker": "MSFT",
                "source_file": "microsoft_2024_10k.pdf",
                "fiscal_year": 2024,
                "document_type": "10K",
                "page": 27,
            },
            "distance": 0.31,
        },
    ]

    result = tool.run(
        question=(
            "What risks did Microsoft disclose?"
        ),
        companies=[
            "Microsoft"
        ],
        retrieved_chunks=test_chunks,
    )

    print("Tool metadata:")
    print(
        tool.get_metadata()
    )

    print("\nRisk-analysis result:")
    print(
        result
    )

    print("\nCategory count:")
    print(
        result.get(
            "category_count"
        )
    )

    print("\nVerified prompt context:")
    print(
        result.get(
            "prompt_context"
        )
    )