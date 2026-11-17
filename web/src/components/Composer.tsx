import { useState, type KeyboardEvent } from "react";
import { MAX_QUESTION_LENGTH } from "../api";

type Props = { disabled: boolean; onSend: (question: string) => void };

export function Composer({ disabled, onSend }: Props) {
  const [value, setValue] = useState("");
  const trimmed = value.trim();
  const canSend = !disabled && trimmed.length > 0;

  function send() {
    if (!canSend) return;
    onSend(trimmed);
    setValue("");
  }

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      send();
    }
  }

  const remaining = MAX_QUESTION_LENGTH - value.length;
  return (
    <form
      className="composer"
      onSubmit={(e) => {
        e.preventDefault();
        send();
      }}
    >
      <textarea
        aria-label="Question"
        placeholder="Ask about the textbooks"
        rows={2}
        maxLength={MAX_QUESTION_LENGTH}
        value={value}
        onChange={(e) => setValue(e.target.value.slice(0, MAX_QUESTION_LENGTH))}
        onKeyDown={onKeyDown}
      />
      <div className="composer-row">
        <span className="count" aria-live="polite">
          {remaining <= 50 ? `${remaining} characters left` : ""}
        </span>
        <button type="submit" disabled={!canSend}>
          Send
        </button>
      </div>
    </form>
  );
}
