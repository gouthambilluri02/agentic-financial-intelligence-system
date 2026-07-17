import re


class FinancialValueExtractor:
    """
    Extract structured financial values from retrieved report text.

    This version focuses on flattened financial tables such as:

    Year Ended September 28, 2024 2023 2022
    Total net sales 391,035 383,285 394,328
    """

    METRIC_TERMS = {
        "revenue": [
            "total net sales",
            "net sales",
            "total revenue",
            "revenue",
        ],
        "net_income": [
            "net income",
            "net earnings",
        ],
        "operating_income": [
            "operating income",
            "operating profit",
        ],
        "eps": [
            "diluted earnings per share",
            "basic earnings per share",
            "earnings per share",
            "diluted eps",
            "basic eps",
        ],
    }

    def extract_year_values(
        self,
        retrieved_chunks: list[dict],
        metric: str,
    ) -> list[dict]:
        """
        Extract year/value pairs for the requested metric.
        """

        extracted_values: list[dict] = []

        for chunk in retrieved_chunks:
            metadata = chunk.get("metadata", {})
            content = chunk.get("content", "")

            chunk_values = self._extract_from_content(
                content=content,
                metric=metric,
            )

            for item in chunk_values:
                extracted_values.append(
                    {
                        "company": metadata.get(
                            "company",
                            "Unknown company",
                        ),
                        "metric": metric,
                        "year": item["year"],
                        "value": item["value"],
                        "source_file": metadata.get(
                            "source_file",
                            "Unknown source",
                        ),
                        "page": metadata.get(
                            "page",
                            "Unknown page",
                        ),
                    }
                )

        return self._remove_duplicates(
            extracted_values
        )

    def get_latest_two_values(
        self,
        extracted_values: list[dict],
        company: str,
    ) -> list[dict]:
        """
        Return the latest two unique yearly values for one company.
        """

        company_values = [
            item
            for item in extracted_values
            if item.get("company") == company
        ]

        values_by_year: dict[int, dict] = {}

        for item in company_values:
            year = item.get("year")

            if isinstance(year, int) and year not in values_by_year:
                values_by_year[year] = item

        sorted_values = sorted(
            values_by_year.values(),
            key=lambda item: item["year"],
            reverse=True,
        )

        return sorted_values[:2]

    def _extract_from_content(
        self,
        content: str,
        metric: str,
    ) -> list[dict]:
        """
        Extract metric values from one retrieved chunk.
        """

        if not content.strip():
            return []

        normalized_content = self._normalize_content(
            content
        )

        table_values = self._extract_table_values(
            content=normalized_content,
            metric=metric,
        )

        if table_values:
            return table_values

        return self._extract_sentence_values(
            content=normalized_content,
            metric=metric,
        )

    @staticmethod
    def _normalize_content(
        content: str,
    ) -> str:
        """
        Normalize spaces and financial number formatting.
        """

        normalized = content.replace(",", "")
        normalized = normalized.replace("$", "")
        normalized = normalized.replace("\t", " ")
        normalized = re.sub(
            r"\s+",
            " ",
            normalized,
        )

        return normalized.strip()

    def _extract_table_values(
        self,
        content: str,
        metric: str,
    ) -> list[dict]:
        """
        Extract values from flattened table text.

        Example:

        Year Ended September 28 2024 2023 2022
        Total net sales 391035 383285 394328
        """

        metric_terms = self.METRIC_TERMS.get(
            metric,
            [],
        )

        if not metric_terms:
            return []

        lower_content = content.lower()

        matching_term = next(
            (
                term
                for term in metric_terms
                if term in lower_content
            ),
            None,
        )

        if matching_term is None:
            return []

        years = self._extract_year_headers(
            content
        )

        if len(years) < 2:
            return []

        metric_position = lower_content.find(
            matching_term
        )

        metric_text = content[
            metric_position
            + len(matching_term):
        ]

        values = self._extract_financial_numbers(
            metric_text
        )

        if len(values) < len(years):
            return []

        results = []

        for year, value in zip(
            years,
            values,
        ):
            results.append(
                {
                    "year": year,
                    "value": value,
                }
            )

        return results

    @staticmethod
    def _extract_year_headers(
        content: str,
    ) -> list[int]:
        """
        Extract unique fiscal years in their displayed order.
        """

        year_matches = re.findall(
            r"\b(20\d{2})\b",
            content,
        )

        years: list[int] = []

        for year_text in year_matches:
            year = int(year_text)

            if year not in years:
                years.append(year)

        return years

    @staticmethod
    def _extract_financial_numbers(
        metric_text: str,
    ) -> list[float]:
        """
        Extract financial values appearing after the metric name.

        Stops unrelated date fragments such as:
        September 28, 2024
        from being interpreted as metric values because extraction
        begins only after the metric label.
        """

        number_matches = re.findall(
            r"(?<![\w.])-?\d+(?:\.\d+)?",
            metric_text,
        )

        values: list[float] = []

        for number_text in number_matches:
            try:
                number = float(number_text)
            except ValueError:
                continue

            if 1900 <= number <= 2100:
                continue

            values.append(number)

        return values

    def _extract_sentence_values(
        self,
        content: str,
        metric: str,
    ) -> list[dict]:
        """
        Fallback for sentences such as:

        Revenue was 391035 in 2024 and 383285 in 2023.
        """

        metric_terms = self.METRIC_TERMS.get(
            metric,
            [],
        )

        if not metric_terms:
            return []

        escaped_terms = "|".join(
            re.escape(term)
            for term in metric_terms
        )

        patterns = [
            (
                rf"(?i)(?:{escaped_terms}).{{0,100}}?"
                rf"\b(20\d{{2}})\b.{{0,40}}?"
                rf"(-?\d+(?:\.\d+)?)"
            ),
            (
                rf"(?i)\b(20\d{{2}})\b.{{0,100}}?"
                rf"(?:{escaped_terms}).{{0,40}}?"
                rf"(-?\d+(?:\.\d+)?)"
            ),
            (
                rf"(?i)(?:{escaped_terms}).{{0,60}}?"
                rf"(-?\d+(?:\.\d+)?).{{0,40}}?"
                rf"\b(20\d{{2}})\b"
            ),
        ]

        results: list[dict] = []

        for pattern_index, pattern in enumerate(
            patterns
        ):
            matches = re.findall(
                pattern,
                content,
            )

            for first, second in matches:
                try:
                    if pattern_index == 2:
                        value = float(first)
                        year = int(second)
                    else:
                        year = int(first)
                        value = float(second)
                except ValueError:
                    continue

                results.append(
                    {
                        "year": year,
                        "value": value,
                    }
                )

        return self._remove_duplicates(
            results
        )

    @staticmethod
    def _remove_duplicates(
        values: list[dict],
    ) -> list[dict]:
        """
        Remove exact duplicate values while preserving order.
        """

        unique_values: list[dict] = []
        seen_values: set[tuple] = set()

        for item in values:
            key = (
                item.get("company"),
                item.get("metric"),
                item.get("year"),
                item.get("value"),
                item.get("source_file"),
                item.get("page"),
            )

            if key in seen_values:
                continue

            seen_values.add(key)
            unique_values.append(item)

        return unique_values


if __name__ == "__main__":
    extractor = FinancialValueExtractor()

    test_chunks = [
        {
            "content": (
                "Year Ended September 28, 2024 2023 2022 "
                "Total net sales 391,035 383,285 394,328"
            ),
            "metadata": {
                "company": "Apple",
                "source_file": "apple_2024_10k.pdf",
                "page": 37,
            },
        }
    ]

    extracted_values = extractor.extract_year_values(
        retrieved_chunks=test_chunks,
        metric="revenue",
    )

    print("Extracted values:")

    for value in extracted_values:
        print(value)

    print("\nLatest two values:")

    latest_values = extractor.get_latest_two_values(
        extracted_values=extracted_values,
        company="Apple",
    )

    for value in latest_values:
        print(value)