import { describe, expect, it } from "vitest";
import { act, fireEvent, render, screen } from "@testing-library/react";
import { useAutoScroll } from "./useAutoScroll";

/**
 * jsdom has no layout, so the container's metrics are stubbed. `scrollTop` is
 * settable and - like a real browser - only dispatches a `scroll` event when
 * the value actually changes (assigning the current position is a no-op, which
 * is exactly what arms the old single-shot "programmatic" flag).
 */
interface Metrics {
  scrollHeight: number;
  clientHeight: number;
}

function installMetrics(el: HTMLElement, initial: Metrics): Metrics {
  const metrics: Metrics = { ...initial };
  let top = 0;

  Object.defineProperty(el, "scrollHeight", {
    configurable: true,
    get: () => metrics.scrollHeight,
  });
  Object.defineProperty(el, "clientHeight", {
    configurable: true,
    get: () => metrics.clientHeight,
  });
  Object.defineProperty(el, "scrollTop", {
    configurable: true,
    get: () => top,
    set: (next: number) => {
      const max = Math.max(0, metrics.scrollHeight - metrics.clientHeight);
      const clamped = Math.max(0, Math.min(next, max));
      const changed = clamped !== top;
      top = clamped;
      if (changed) el.dispatchEvent(new Event("scroll"));
    },
  });

  return metrics;
}

function Harness({ content, revision }: { content: number; revision: number }) {
  const { scrollRef, isScrolledAway, scrollToBottom, reEngage } = useAutoScroll({
    nearBottomThreshold: 24,
    deps: [content, revision],
  });

  return (
    <div>
      <div data-testid="scroller" ref={scrollRef} />
      <span data-testid="away">{isScrolledAway ? "yes" : "no"}</span>
      <button onClick={() => scrollToBottom("instant")}>jump</button>
      <button onClick={() => reEngage()}>reengage</button>
    </div>
  );
}

/** scrollHeight 1000 - clientHeight 500 → max scrollTop 500 */
function setup(props: { content?: number; revision?: number } = {}) {
  const initial = { content: props.content ?? 0, revision: props.revision ?? 0 };
  const utils = render(<Harness {...initial} />);
  const el = screen.getByTestId("scroller");
  const metrics = installMetrics(el, { scrollHeight: 1000, clientHeight: 500 });
  const rerenderWith = (next: { content: number; revision: number }) =>
    utils.rerender(<Harness {...next} />);
  return { ...utils, el, metrics, rerenderWith };
}

const away = () => screen.getByTestId("away").textContent;

/**
 * Move the container the way a user would. The resulting state update comes
 * from a native (non-React) listener, so it needs an explicit flush.
 */
const scrollTo = (el: HTMLElement, top: number) => {
  act(() => {
    el.scrollTop = top;
  });
};

describe("useAutoScroll", () => {
  it("follows content growth while the user is at the bottom", () => {
    const { el, metrics, rerenderWith } = setup();

    metrics.scrollHeight = 1200;
    rerenderWith({ content: 1, revision: 0 });

    expect(el.scrollTop).toBe(700);
    expect(away()).toBe("no");
  });

  it("REGRESSION: a wheel-up during streaming is not undone by incoming tokens", () => {
    const { el, metrics, rerenderWith } = setup();

    // First token: engaged, follows to the bottom.
    metrics.scrollHeight = 1200;
    rerenderWith({ content: 1, revision: 0 });
    expect(el.scrollTop).toBe(700);

    // An effect run that does not move the scroll position (messages array
    // identity changes every store tick) is what leaves the old single-shot
    // "programmatic" flag armed in a real browser.
    rerenderWith({ content: 1, revision: 1 });
    expect(el.scrollTop).toBe(700);

    // User wheels up, then drags the scrollbar up.
    fireEvent.wheel(el, { deltaY: -120 });
    expect(away()).toBe("yes");
    scrollTo(el, 400);
    expect(el.scrollTop).toBe(400);

    // Streaming keeps producing tokens.
    metrics.scrollHeight = 1600;
    rerenderWith({ content: 2, revision: 1 });

    expect(el.scrollTop).toBe(400);
    expect(away()).toBe("yes");
  });

  it("disengages on the first upward wheel even inside the near-bottom threshold", () => {
    const { el, metrics, rerenderWith } = setup();

    metrics.scrollHeight = 1200;
    rerenderWith({ content: 1, revision: 0 });
    expect(el.scrollTop).toBe(700); // distance from bottom: 0

    fireEvent.wheel(el, { deltaY: -40 });

    expect(away()).toBe("yes");
    expect(el.scrollTop).toBe(700);
  });

  it("does not disengage on a programmatic scroll that lands where we already were", () => {
    const { el, rerenderWith } = setup();

    rerenderWith({ content: 1, revision: 0 });
    // No growth: the auto-scroll assignment is a no-op, so no scroll event fires.
    rerenderWith({ content: 1, revision: 1 });
    rerenderWith({ content: 1, revision: 2 });

    expect(away()).toBe("no");
    expect(el.scrollTop).toBe(500);
  });

  it("disengages on a slow scrollbar drag delivered only as 2px scroll steps", () => {
    const { el, metrics, rerenderWith } = setup();

    metrics.scrollHeight = 1200;
    rerenderWith({ content: 1, revision: 0 });
    expect(el.scrollTop).toBe(700);

    // Native scrollbars dispatch no pointer events and a slow drag moves a
    // couple of pixels per event, so only accumulated travel can catch it.
    scrollTo(el, 698);
    expect(away()).toBe("no");
    scrollTo(el, 696);
    scrollTo(el, 694);
    expect(away()).toBe("no");
    scrollTo(el, 692);
    expect(away()).toBe("yes");

    metrics.scrollHeight = 1500;
    rerenderWith({ content: 2, revision: 0 });
    expect(el.scrollTop).toBe(692);
  });

  it("re-engages when the user scrolls back within 24px of the bottom", () => {
    const { el, metrics, rerenderWith } = setup();

    metrics.scrollHeight = 1200;
    rerenderWith({ content: 1, revision: 0 });

    fireEvent.wheel(el, { deltaY: -200 });
    scrollTo(el, 300);
    expect(away()).toBe("yes");

    // 40px from the bottom → still disengaged.
    scrollTo(el, 660);
    expect(away()).toBe("yes");

    // 10px from the bottom → re-engaged.
    scrollTo(el, 690);
    expect(away()).toBe("no");

    metrics.scrollHeight = 1400;
    rerenderWith({ content: 2, revision: 0 });
    expect(el.scrollTop).toBe(900);
  });

  it("keyboard upward scroll disengages", () => {
    const { el, rerenderWith } = setup();

    rerenderWith({ content: 1, revision: 0 });
    fireEvent.keyDown(el, { key: "ArrowUp" });

    expect(away()).toBe("yes");
  });

  it("does not move the scroll position at all when content grows while disengaged", () => {
    const { el, metrics, rerenderWith } = setup();

    metrics.scrollHeight = 1200;
    rerenderWith({ content: 1, revision: 0 });
    fireEvent.wheel(el, { deltaY: -100 });
    scrollTo(el, 350);

    metrics.scrollHeight = 2000;
    rerenderWith({ content: 2, revision: 0 });
    metrics.scrollHeight = 2600;
    rerenderWith({ content: 3, revision: 0 });

    expect(el.scrollTop).toBe(350);
  });

  it("reEngage() re-arms following", () => {
    const { el, metrics, rerenderWith } = setup();

    metrics.scrollHeight = 1200;
    rerenderWith({ content: 1, revision: 0 });
    fireEvent.wheel(el, { deltaY: -100 });
    scrollTo(el, 200);
    expect(away()).toBe("yes");

    fireEvent.click(screen.getByText("reengage"));
    expect(away()).toBe("no");

    metrics.scrollHeight = 1400;
    rerenderWith({ content: 2, revision: 0 });
    expect(el.scrollTop).toBe(900);
  });

  it("scrollToBottom() jumps to the bottom and resumes following", () => {
    const { el, metrics, rerenderWith } = setup();

    metrics.scrollHeight = 1200;
    rerenderWith({ content: 1, revision: 0 });
    fireEvent.wheel(el, { deltaY: -100 });
    scrollTo(el, 200);

    fireEvent.click(screen.getByText("jump"));

    expect(away()).toBe("no");
    expect(el.scrollTop).toBe(700);

    metrics.scrollHeight = 1500;
    rerenderWith({ content: 2, revision: 0 });
    expect(el.scrollTop).toBe(1000);
  });
});
