import { useState, type FormEvent } from "react";
import { Send } from "lucide-react";

interface QuestionInputProps {
  onSubmit: (question: string) => void;
  isLoading?: boolean;
}

export function QuestionInput({
  onSubmit,
  isLoading = false,
}: QuestionInputProps) {
  const [question, setQuestion] = useState("");

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    const cleanedQuestion = question.trim();

    if (!cleanedQuestion || isLoading) {
      return;
    }

    onSubmit(cleanedQuestion);
    setQuestion("");
  }

  return (
    <form className="question-input" onSubmit={handleSubmit}>
      <textarea
        className="question-input__field"
        value={question}
        onChange={(event) => setQuestion(event.target.value)}
        placeholder="Ask about revenue, risks, margins, or company comparisons..."
        rows={4}
        disabled={isLoading}
      />

      <button
        type="submit"
        className="question-input__button"
        disabled={!question.trim() || isLoading}
      >
        <Send size={18} />

        <span>
          {isLoading ? "Thinking..." : "Ask AI"}
        </span>
      </button>
    </form>
  );
}