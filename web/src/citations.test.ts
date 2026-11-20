import { highlight } from "./citations";

describe("highlight", () => {
  it("merges overlapping quotes into one mark", () => {
    expect(highlight("abcdefgh", ["abcd", "cdef"])).toEqual([
      { text: "abcdef", mark: true },
      { text: "gh", mark: false },
    ]);
  });
  it("ignores a quote that is not found", () => {
    expect(highlight("abc", ["xyz"])).toEqual([{ text: "abc", mark: false }]);
  });
  it("ignores an empty quote", () => {
    expect(highlight("abc", [""])).toEqual([{ text: "abc", mark: false }]);
  });
  it("marks every occurrence", () => {
    expect(highlight("a-b-a", ["a"]).filter((p) => p.mark)).toHaveLength(2);
  });
});
