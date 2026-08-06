import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { MainLayout } from "../components/MainLayout";
import { QuestionInput } from "../components/chat/QuestionInput";
import { DashboardLayout } from "../components/layout/DashboardLayout";

import {
  askFinancialQuestion,
  type QueryResponse,
} from "../services/api";

interface ChatMessage {
  id: number;
  role: "user" | "assistant";
  content: string;
  response?: QueryResponse;
}

export function HomePage() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [latestResponse, setLatestResponse] =
    useState<QueryResponse | null>(null);
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  async function handleQuestionSubmit(question: string) {
    const userMessage: ChatMessage = {
      id: Date.now(),
      role: "user",
      content: question,
    };

    setMessages((currentMessages) => [
      ...currentMessages,
      userMessage,
    ]);

    setLatestResponse(null);
    setError("");
    setIsLoading(true);

    try {
      const result = await askFinancialQuestion(question);

      const assistantMessage: ChatMessage = {
        id: Date.now() + 1,
        role: "assistant",
        content: result.answer,
        response: result,
      };

      setMessages((currentMessages) => [
        ...currentMessages,
        assistantMessage,
      ]);

      setLatestResponse(result);
    } catch (requestError) {
      const message =
        requestError instanceof Error
          ? requestError.message
          : "Unable to process the question.";

      setError(message);
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <MainLayout>
      <DashboardLayout
        leftPanel={
          <div>
            <h3>Documents</h3>

            <p>Upload and manage financial reports.</p>

            <div className="document-empty-state">
              <span>No documents uploaded yet.</span>
            </div>
          </div>
        }
        mainPanel={
          <div className="chat-workspace">
            <div>
              <h2>Ask FinAgent AI</h2>

              <p>
                Ask questions about revenue, risks, margins,
                financial performance, or company comparisons.
              </p>
            </div>

            <div className="chat-history">
              {messages.length === 0 && (
                <div className="chat-empty-state">
                  <h3>Your financial analyst is ready</h3>

                  <p>
                    Try asking about company revenue, risks,
                    margins, or financial comparisons.
                  </p>
                </div>
              )}

              {messages.map((message) => {
                if (message.role === "user") {
                  return (
                    <div
                      className="submitted-question"
                      key={message.id}
                    >
                      <span>Your question</span>
                      <p>{message.content}</p>
                    </div>
                  );
                }

                return (
                  <div
                    className="assistant-response"
                    key={message.id}
                  >
                    <span>FinAgent AI</span>

                    <div className="markdown-response">
                      <ReactMarkdown
                        remarkPlugins={[remarkGfm]}
                      >
                        {message.content}
                      </ReactMarkdown>
                    </div>
                  </div>
                );
              })}

              {isLoading && (
                <div className="assistant-response">
                  <span>FinAgent AI</span>

                  <div className="loading-indicator">
                    <i />
                    <i />
                    <i />
                  </div>

                  <p>Analyzing financial reports...</p>
                </div>
              )}

              {error && (
                <div className="request-error" role="alert">
                  {error}
                </div>
              )}
            </div>

            <QuestionInput
              onSubmit={handleQuestionSubmit}
              isLoading={isLoading}
            />
          </div>
        }
        rightPanel={
          <div>
            <h3>Sources</h3>

            {!latestResponse && (
              <p>
                Retrieved sources and execution details will
                appear here.
              </p>
            )}

            {latestResponse && (
              <div className="execution-details">
                {latestResponse.selected_tool && (
                  <div className="detail-row">
                    <span>Selected tool</span>
                    <strong>
                      {latestResponse.selected_tool}
                    </strong>
                  </div>
                )}

                {latestResponse.detected_intent && (
                  <div className="detail-row">
                    <span>Detected intent</span>
                    <strong>
                      {latestResponse.detected_intent}
                    </strong>
                  </div>
                )}

                {latestResponse.detected_metric && (
                  <div className="detail-row">
                    <span>Detected metric</span>
                    <strong>
                      {latestResponse.detected_metric}
                    </strong>
                  </div>
                )}

                {latestResponse.duration_ms !== undefined && (
                  <div className="detail-row">
                    <span>Duration</span>
                    <strong>
                      {latestResponse.duration_ms} ms
                    </strong>
                  </div>
                )}

                {latestResponse.retry_performed !== undefined && (
                  <div className="detail-row">
                    <span>Retrieval retry</span>
                    <strong>
                      {latestResponse.retry_performed
                        ? "Yes"
                        : "No"}
                    </strong>
                  </div>
                )}

                <div className="sources-list">
                  <h4>Retrieved sources</h4>

                  {!latestResponse.sources?.length && (
                    <p>No source information was returned.</p>
                  )}

                  {latestResponse.sources?.map(
                    (source, index) => (
                      <article
                        className="source-card"
                        key={`${source.source_file ?? "source"}-${source.page ?? index}-${index}`}
                      >
                        <strong>
                          {source.source_file ??
                            "Financial report"}
                        </strong>

                        {source.company && (
                          <p>
                            Company: {source.company}
                          </p>
                        )}

                        {source.ticker && (
                          <p>Ticker: {source.ticker}</p>
                        )}

                        {source.fiscal_year !== undefined && (
                          <p>
                            Fiscal year:{" "}
                            {source.fiscal_year}
                          </p>
                        )}

                        {source.document_type && (
                          <p>
                            Type: {source.document_type}
                          </p>
                        )}

                        {source.page !== undefined && (
                          <p>Page: {source.page}</p>
                        )}
                      </article>
                    ),
                  )}
                </div>
              </div>
            )}
          </div>
        }
      />
    </MainLayout>
  );
}