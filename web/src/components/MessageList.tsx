import type { AskError, AskResponse, Passage } from "../api";
import { AnswerText } from "./AnswerText";

export type Message = {
  id: number;
  question: string;
  state: { kind: "loading" } | { kind: "done"; data: AskResponse } | { kind: "error"; error: AskError };
};

const ERROR_TEXT: Record<string, string> = {
  invalid_question: "The question must be 1 to 500 characters.",
  daily_limit: "The daily question limit is reached. Try again tomorrow.",
  upstream_error: "The answer service failed. Try again.",
  timeout: "The answer took too long. Try again.",
  network: "Could not reach the server. Check that it is running.",
  bad_response: "The server sent a response I could not read.",
};

type Props = {
  messages: Message[];
  activeChip: string | null;
  onChip: (passage: Passage, key: string, el: HTMLButtonElement, message: Message) => void;
  onRetry: (message: Message) => void;
};

export function MessageList({ messages, activeChip, onChip, onRetry }: Props) {
  return (
    <ol className="messages">
      {messages.map((m) => (
        <li key={m.id}>
          <p className="question">{m.question}</p>
          <div className="reply">
            {m.state.kind === "loading" && (
              <p role="status" className="muted">
                Searching the books
              </p>
            )}
            {m.state.kind === "error" && (
              <p role="alert" className="error">
                {ERROR_TEXT[m.state.error.code] ?? m.state.error.message}{" "}
                {m.state.error.code !== "daily_limit" && m.state.error.code !== "invalid_question" && (
                  <button type="button" className="link" onClick={() => onRetry(m)}>
                    Retry
                  </button>
                )}
              </p>
            )}
            {m.state.kind === "done" && <Done m={m} state={m.state} activeChip={activeChip} onChip={onChip} />}
          </div>
        </li>
      ))}
    </ol>
  );
}

function Done({
  m,
  state,
  activeChip,
  onChip,
}: {
  m: Message;
  state: { kind: "done"; data: AskResponse };
  activeChip: string | null;
  onChip: Props["onChip"];
}) {
  const { data } = state;
  if (data.status === "refused") {
    return <p className="muted">Claude declined to answer this question.</p>;
  }
  return (
    <>
      {data.status === "no_answer" && <p className="muted">The books do not answer this question.</p>}
      <AnswerText
        segments={data.segments}
        passages={data.passages}
        activeChip={activeChip}
        chipKey={(s, n) => `${m.id}:${s}:${n}`}
        onChip={(p, key, el) => onChip(p, key, el, m)}
      />
    </>
  );
}
