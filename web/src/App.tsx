import { useRef, useState } from "react";
import { ask, type Passage } from "./api";
import { Composer } from "./components/Composer";
import { MessageList, type Message } from "./components/MessageList";
import { SourcePanel } from "./components/SourcePanel";

type Open = { key: string; passage: Passage; quotes: string[]; el: HTMLButtonElement };

export const ATTRIBUTION =
  "Answers come from Introduction to Python Programming and Introduction to Computer Science (OpenStax, CC BY-NC-SA 4.0).";

export function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [open, setOpen] = useState<Open | null>(null);
  const nextId = useRef(1);
  const busy = messages.some((m) => m.state.kind === "loading");

  async function run(id: number, question: string) {
    const result = await ask(question);
    setMessages((ms) =>
      ms.map((m) =>
        m.id === id
          ? { ...m, state: result.ok ? { kind: "done", data: result.data } : { kind: "error", error: result.error } }
          : m,
      ),
    );
  }

  function send(question: string) {
    const id = nextId.current++;
    setMessages((ms) => [...ms, { id, question, state: { kind: "loading" } }]);
    void run(id, question);
  }

  function retry(m: Message) {
    setMessages((ms) => ms.map((x) => (x.id === m.id ? { ...x, state: { kind: "loading" } } : x)));
    void run(m.id, m.question);
  }

  function onChip(passage: Passage, key: string, el: HTMLButtonElement, message: Message) {
    if (message.state.kind !== "done") return;
    const quotes = message.state.data.segments.flatMap((s) =>
      s.citations.filter((c) => c.passage === passage.n).map((c) => c.cited_text),
    );
    setOpen({ key, passage, quotes, el });
  }

  function close() {
    open?.el.focus();
    setOpen(null);
  }

  return (
    <div className="app">
      <header>
        <h1>Docent</h1>
      </header>
      <main>
        <MessageList messages={messages} activeChip={open?.key ?? null} onChip={onChip} onRetry={retry} />
        {open && <SourcePanel passage={open.passage} quotes={open.quotes} onClose={close} />}
      </main>
      <Composer disabled={busy} onSend={send} />
      <footer>{ATTRIBUTION}</footer>
    </div>
  );
}
