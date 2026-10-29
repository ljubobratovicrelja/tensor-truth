import { describe, expect, it } from "vitest";
import { extractDetail } from "../client";

describe("extractDetail", () => {
  it("returns string details unchanged", () => {
    expect(extractDetail({ detail: "No elaboration with that id" })).toBe(
      "No elaboration with that id"
    );
  });

  it("reads the message out of a conflict object", () => {
    // The runs API returns 409 as {detail, run_id}; String(detail) used to
    // render "[object Object]" in the elaboration panel.
    expect(
      extractDetail({
        detail: {
          detail: "Session already has an active run",
          run_id: "abc123",
        },
      } as never)
    ).toBe("Session already has an active run");
  });

  it("serialises an object without a usable message", () => {
    expect(extractDetail({ detail: { code: 42 } } as never)).toBe('{"code":42}');
  });

  it("joins validation error arrays", () => {
    expect(extractDetail({ detail: [{ msg: "a" }, { msg: "b" }] } as never)).toBe("a; b");
  });

  it("returns undefined when there is no detail", () => {
    expect(extractDetail({} as never)).toBeUndefined();
  });
});
