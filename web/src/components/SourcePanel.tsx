import { useEffect, useRef } from "react";
import type { Passage } from "../api";
import { highlight, pageLabel } from "../citations";

type Props = { passage: Passage; quotes: string[]; onClose: () => void };

export function SourcePanel({ passage, quotes, onClose }: Props) {
  const ref = useRef<HTMLElement>(null);

  useEffect(() => {
    ref.current?.focus();
  }, [passage]);

  return (
    <aside
      ref={ref}
      className="panel"
      role="dialog"
      aria-label={`Source ${passage.n}`}
      tabIndex={-1}
      onKeyDown={(e) => {
        if (e.key === "Escape") {
          e.stopPropagation();
          onClose();
        }
      }}
    >
      <div className="panel-head">
        <div>
          <strong>{passage.book_title}</strong>
          <div>
            {passage.section_title}, {pageLabel(passage)}
          </div>
        </div>
        <button type="button" onClick={onClose} aria-label="Close source">
          Close
        </button>
      </div>
      <p className="passage" data-testid="passage-text">
        {highlight(passage.text, quotes).map((piece, i) =>
          piece.mark ? <mark key={i}>{piece.text}</mark> : <span key={i}>{piece.text}</span>,
        )}
      </p>
    </aside>
  );
}
