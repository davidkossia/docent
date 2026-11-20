import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, vi } from "vitest";
import { App, ATTRIBUTION } from "./App";
import answered from "./test-fixtures/answered.json";
import noAnswer from "./test-fixtures/no_answer.json";
import refused from "./test-fixtures/refused.json";
import dailyLimit from "./test-fixtures/error_daily_limit.json";

function mockFetch(body: unknown, status = 200) {
  const fn = vi.fn(async () => new Response(JSON.stringify(body), { status }));
  vi.stubGlobal("fetch", fn);
  return fn;
}

async function askQuestion(text = "How does precedence work?") {
  const user = userEvent.setup();
  render(<App />);
  await user.type(screen.getByLabelText("Question"), text);
  await user.keyboard("{Enter}");
  return user;
}

beforeEach(() => vi.unstubAllGlobals());
afterEach(() => vi.unstubAllGlobals());

describe("answered", () => {
  it("places chips after the segments that cite, one per distinct passage", async () => {
    mockFetch(answered);
    await askQuestion();
    const chip1 = await screen.findAllByRole("button", { name: /^Source 1:/ });
    // Passage 1 is cited in segment 0 and segment 2.
    expect(chip1).toHaveLength(2);
    expect(screen.getAllByRole("button", { name: /^Source 2:/ })).toHaveLength(1);
    expect(screen.getAllByRole("button", { name: /^Source 3:/ })).toHaveLength(1);
    const segs = document.querySelectorAll(".answer > span");
    expect(segs).toHaveLength(3);
    expect(within(segs[0] as HTMLElement).getAllByRole("button")).toHaveLength(1);
    expect(within(segs[1] as HTMLElement).getByRole("button", { name: /^Source 2:/ })).toBeInTheDocument();
    expect(within(segs[2] as HTMLElement).getAllByRole("button")).toHaveLength(2);
  });

  it("gives chips an accessible name with section and pages", async () => {
    mockFetch(answered);
    await askQuestion();
    expect(await screen.findByRole("button", { name: "Source 2: 1.5 Number basics, page 30" })).toBeInTheDocument();
    expect(screen.getAllByRole("button", { name: "Source 1: 1.5 Number basics, pages 29–30" })).toHaveLength(2);
  });

  it("opens the matching passage and highlights cited text", async () => {
    mockFetch(answered);
    const user = await askQuestion();
    await user.click(await screen.findByRole("button", { name: /^Source 2:/ }));
    const panel = screen.getByRole("dialog");
    expect(within(panel).getByText("Introduction to Python Programming")).toBeInTheDocument();
    const marks = panel.querySelectorAll("mark");
    expect(marks).toHaveLength(1);
    expect(marks[0].textContent).toBe("Anything inside parentheses is evaluated first");
  });

  it("highlights every citation of a passage", async () => {
    mockFetch(answered);
    const user = await askQuestion();
    await user.click((await screen.findAllByRole("button", { name: /^Source 1:/ }))[0]);
    const marks = [...screen.getByRole("dialog").querySelectorAll("mark")].map((m) => m.textContent);
    expect(marks).toEqual([
      "each operator is evaluated in order of precedence",
      "multiplication happens before addition",
    ]);
  });

  it("closes on Escape and returns focus to the chip", async () => {
    mockFetch(answered);
    const user = await askQuestion();
    const chip = await screen.findByRole("button", { name: /^Source 2:/ });
    await user.click(chip);
    expect(screen.getByRole("dialog")).toHaveFocus();
    await user.keyboard("{Escape}");
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(chip).toHaveFocus();
  });

  it("renders passage text as text, not HTML", async () => {
    const evil = structuredClone(answered);
    evil.passages[1].text = 'Anything inside parentheses is evaluated first <img src=x onerror="alert(1)">';
    mockFetch(evil);
    const user = await askQuestion();
    await user.click(await screen.findByRole("button", { name: /^Source 2:/ }));
    expect(screen.getByRole("dialog").querySelector("img")).toBeNull();
    expect(screen.getByRole("dialog")).toHaveTextContent('<img src=x onerror="alert(1)">');
  });
});

describe("states", () => {
  it("shows loading while the request is pending", async () => {
    vi.stubGlobal("fetch", vi.fn(() => new Promise(() => {})));
    await askQuestion();
    expect(await screen.findByText("Searching the books")).toBeInTheDocument();
  });

  it("renders no_answer with the fixed sentence and no chips", async () => {
    mockFetch(noAnswer);
    await askQuestion();
    expect(await screen.findAllByText("The documentation does not answer this question.")).toHaveLength(1);
    expect(screen.queryByRole("button", { name: /^Source/ })).not.toBeInTheDocument();
  });

  it("renders refused", async () => {
    mockFetch(refused);
    await askQuestion();
    expect(await screen.findByText(/declined to answer/)).toBeInTheDocument();
  });

  it.each([
    ["invalid_question", 400, /1 to 500 characters/],
    ["daily_limit", 429, /daily question limit/],
    ["upstream_error", 502, /answer service failed/],
    ["timeout", 504, /too long/],
  ])("renders error %s", async (code, status, text) => {
    mockFetch({ error: { code, message: "x" } }, status as number);
    await askQuestion();
    expect(await screen.findByRole("alert")).toHaveTextContent(text as RegExp);
  });

  it("renders the daily_limit fixture without a retry button", async () => {
    mockFetch(dailyLimit, 429);
    await askQuestion();
    await screen.findByRole("alert");
    expect(screen.queryByRole("button", { name: "Retry" })).not.toBeInTheDocument();
  });

  it("renders a network failure", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => { throw new TypeError("fail"); }));
    await askQuestion();
    expect(await screen.findByRole("alert")).toHaveTextContent(/Could not reach the server/);
  });
});

describe("retry", () => {
  async function retryFlow(first: () => Promise<Response>) {
    const fn = vi
      .fn()
      .mockImplementationOnce(first)
      .mockImplementation(async () => new Response(JSON.stringify(answered), { status: 200 }));
    vi.stubGlobal("fetch", fn);
    const user = await askQuestion();
    await user.click(await screen.findByRole("button", { name: "Retry" }));
    expect(await screen.findAllByRole("button", { name: /^Source 1:/ })).toHaveLength(2);
    expect(fn).toHaveBeenCalledTimes(2);
  }

  it("retries after upstream_error", () =>
    retryFlow(async () => new Response(JSON.stringify({ error: { code: "upstream_error", message: "x" } }), { status: 502 })));
  it("retries after timeout", () =>
    retryFlow(async () => new Response(JSON.stringify({ error: { code: "timeout", message: "x" } }), { status: 504 })));
  it("retries after a network failure", () =>
    retryFlow(async () => {
      throw new TypeError("fail");
    }));

  it("maps an aborted request to timeout", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => {
      throw new DOMException("t", "TimeoutError");
    }));
    await askQuestion();
    expect(await screen.findByRole("alert")).toHaveTextContent(/too long/);
  });

  it("rejects a 200 body that is not the contract", async () => {
    mockFetch({ status: "weird", segments: [], passages: [] });
    await askQuestion();
    expect(await screen.findByRole("alert")).toHaveTextContent("The server sent an unreadable response.");
  });
});

describe("panel and chips", () => {
  it("opens a chip with Tab then Enter", async () => {
    mockFetch(answered);
    const user = await askQuestion();
    await screen.findAllByRole("button", { name: /^Source 1:/ });
    await user.click(document.body);
    await user.tab();
    expect(screen.getAllByRole("button", { name: /^Source 1:/ })[0]).toHaveFocus();
    await user.keyboard("{Enter}");
    expect(screen.getByRole("dialog")).toBeInTheDocument();
  });

  it("closes on Escape when focus is outside the panel", async () => {
    mockFetch(answered);
    const user = await askQuestion();
    await user.click(await screen.findByRole("button", { name: /^Source 2:/ }));
    await user.click(screen.getByLabelText("Question"));
    await user.keyboard("{Escape}");
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("closes when the open chip is clicked again", async () => {
    mockFetch(answered);
    const user = await askQuestion();
    const chip = await screen.findByRole("button", { name: /^Source 2:/ });
    await user.click(chip);
    await user.click(chip);
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("renders a citation to an unknown passage as a non-interactive marker", async () => {
    const bad = structuredClone(answered);
    bad.segments[0].citations.push({ passage: 9, cited_text: "x" });
    mockFetch(bad);
    await askQuestion();
    const marker = await screen.findByRole("img", { name: "Source 9: unavailable" });
    expect(marker.tagName).toBe("SPAN");
    expect(screen.queryByRole("button", { name: /^Source 9/ })).not.toBeInTheDocument();
  });

  it("renders refused with no chips and no answer text", async () => {
    mockFetch(refused);
    await askQuestion();
    await screen.findByText(/declined to answer/);
    expect(screen.queryByRole("button", { name: /^Source/ })).not.toBeInTheDocument();
    expect(document.querySelector(".answer")).toBeNull();
  });
});

describe("composer", () => {
  it("sends trimmed JSON to POST /ask", async () => {
    const fn = mockFetch(answered);
    await askQuestion("  hello  ");
    await screen.findAllByRole("button", { name: /^Source 1:/ });
    const [url, init] = fn.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe("http://localhost:8787/ask");
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body as string)).toEqual({ question: "hello" });
  });

  it("holds the 500-character limit", async () => {
    const user = userEvent.setup();
    render(<App />);
    const box = screen.getByLabelText("Question") as HTMLTextAreaElement;
    await user.click(box);
    await user.paste("a".repeat(600));
    expect(box.value).toHaveLength(500);
    expect(screen.getAllByText("0 characters left")).toHaveLength(2);
  });

  it("announces the counter only at 50, 10 and 0 left", async () => {
    const user = userEvent.setup();
    render(<App />);
    const box = screen.getByLabelText("Question");
    await user.click(box);
    await user.paste("a".repeat(449));
    expect(document.querySelector(".sr-only")).toHaveTextContent("");
    await user.type(box, "a");
    expect(document.querySelector(".sr-only")).toHaveTextContent("50 characters left");
    await user.type(box, "b");
    expect(document.querySelector(".sr-only")).toHaveTextContent("");
  });

  it("Shift+Enter adds a line and does not send", async () => {
    const fn = mockFetch(answered);
    const user = userEvent.setup();
    render(<App />);
    const box = screen.getByLabelText("Question") as HTMLTextAreaElement;
    await user.type(box, "a{Shift>}{Enter}{/Shift}b");
    expect(box.value).toBe("a\nb");
    expect(fn).not.toHaveBeenCalled();
  });

  it("does not send an empty question", async () => {
    const fn = mockFetch(answered);
    const user = userEvent.setup();
    render(<App />);
    await user.type(screen.getByLabelText("Question"), "   {Enter}");
    expect(fn).not.toHaveBeenCalled();
  });
});

it("shows the corpus attribution", () => {
  render(<App />);
  expect(screen.getByText(ATTRIBUTION)).toBeInTheDocument();
});
