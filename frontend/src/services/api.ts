import axios, { AxiosError } from "axios";

const API_BASE_URL = "http://127.0.0.1:8000";

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
  timeout: 120000,
});

export interface QuerySource {
  company?: string;
  ticker?: string;
  source_file?: string;
  fiscal_year?: string | number;
  document_type?: string;
  page?: number;
}

export interface QueryResponse {
  answer: string;
  sources?: QuerySource[];

  selected_tool?: string;
  detected_intent?: string;
  detected_metric?: string;

  duration_ms?: number;
  retry_performed?: boolean;
  retry_count?: number;
  retrieval_sufficient?: boolean;

  execution_plan?: unknown;
  executed_tools?: string[];
  successful_tools?: string[];
  failed_tools?: string[];
}

interface QueryRequest {
  question: string;
}

export async function askFinancialQuestion(
  question: string,
): Promise<QueryResponse> {
  try {
    const requestBody: QueryRequest = {
      question,
    };

    const response = await apiClient.post<QueryResponse>(
      "/api/v1/query",
      requestBody,
    );

    return response.data;
  } catch (error) {
    if (error instanceof AxiosError) {
      const backendMessage =
        typeof error.response?.data?.detail === "string"
          ? error.response.data.detail
          : null;

      throw new Error(
        backendMessage ??
          "Unable to connect to the financial intelligence backend.",
      );
    }

    throw new Error("An unexpected error occurred.");
  }
}