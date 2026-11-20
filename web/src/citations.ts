import type { Passage } from "./api";

export function pageLabel(p: Passage): string {
  return p.page_start === p.page_end
    ? `page ${p.page_start}`
    : `pages ${p.page_start}–${p.page_end}`;
}

export function chipLabel(p: Passage): string {
  return `Source ${p.n}: ${p.section_title}, ${pageLabel(p)}`;
}

export type Piece = { text: string; mark: boolean };

// Splits text into pieces, marking every occurrence of any quote.
// Overlapping quotes merge. Quotes not found in the text are ignored.
export function highlight(text: string, quotes: string[]): Piece[] {
  const ranges: [number, number][] = [];
  for (const q of quotes) {
    if (!q) continue;
    let from = 0;
    for (;;) {
      const i = text.indexOf(q, from);
      if (i < 0) break;
      ranges.push([i, i + q.length]);
      from = i + q.length;
    }
  }
  ranges.sort((a, b) => a[0] - b[0]);
  const merged: [number, number][] = [];
  for (const r of ranges) {
    const last = merged[merged.length - 1];
    if (last && r[0] <= last[1]) last[1] = Math.max(last[1], r[1]);
    else merged.push([r[0], r[1]]);
  }
  const out: Piece[] = [];
  let pos = 0;
  for (const [s, e] of merged) {
    if (s > pos) out.push({ text: text.slice(pos, s), mark: false });
    out.push({ text: text.slice(s, e), mark: true });
    pos = e;
  }
  if (pos < text.length) out.push({ text: text.slice(pos), mark: false });
  return out;
}
