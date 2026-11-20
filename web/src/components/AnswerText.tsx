import type { Passage, Segment } from "../api";
import { chipLabel } from "../citations";

type Props = {
  segments: Segment[];
  passages: Passage[];
  activeChip: string | null;
  chipKey: (segmentIndex: number, passage: number) => string;
  onChip: (passage: Passage, key: string, el: HTMLButtonElement) => void;
};

export function AnswerText({ segments, passages, activeChip, chipKey, onChip }: Props) {
  return (
    <p className="answer">
      {segments.map((seg, i) => {
        const distinct = [...new Set(seg.citations.map((c) => c.passage))];
        return (
          <span key={i}>
            {seg.text}
            {distinct.map((n) => {
              const passage = passages.find((p) => p.n === n);
              if (!passage) {
                return (
                  <span key={n} className="chip chip-missing" role="img" aria-label={`Source ${n}: unavailable`}>
                    {n}
                  </span>
                );
              }
              const key = chipKey(i, n);
              return (
                <button
                  key={n}
                  type="button"
                  className="chip"
                  aria-label={chipLabel(passage)}
                  aria-expanded={activeChip === key}
                  onClick={(e) => onChip(passage, key, e.currentTarget)}
                >
                  {n}
                </button>
              );
            })}
          </span>
        );
      })}
    </p>
  );
}
