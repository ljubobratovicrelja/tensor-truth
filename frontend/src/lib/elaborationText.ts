/**
 * Text matching for elaboration anchors.
 *
 * A selection is captured with `Range.toString()`, which inserts newlines at
 * block boundaries, while the message's `textContent` concatenates blocks with
 * no separator. Matching the two directly works inside one paragraph but fails
 * for multi-paragraph selections, so both anchor creation and highlight
 * resolution go through the same two-step matcher here:
 *
 * 1. exact substring search (fast, unambiguous for single-block passages);
 * 2. whitespace-stripped search as a fallback, which tolerates the different
 *    whitespace the DOM reports at block boundaries.
 *
 * Because creation and resolution run the same algorithm over the same
 * rendered text, the persisted `occurrence_index` stays meaningful.
 */

export interface Occurrence {
  /** Offset in the haystack passed to `findOccurrences`. */
  start: number;
  end: number;
}

interface StrippedText {
  text: string;
  /** `map[i]` is the offset in the original string of compacted char `i`. */
  map: number[];
}

function stripWhitespaceWithMap(value: string): StrippedText {
  let text = "";
  const map: number[] = [];
  for (let i = 0; i < value.length; i += 1) {
    if (/\s/.test(value[i])) continue;
    text += value[i];
    map.push(i);
  }
  return { text, map };
}

/**
 * All non-overlapping occurrences of `needle` in `haystack`.
 *
 * Falls back to whitespace-insensitive matching only when the exact string is
 * absent, so ordinary selections keep exact (predictable) indices.
 */
export function findOccurrences(haystack: string, needle: string): Occurrence[] {
  if (!needle || !haystack) return [];

  if (haystack.includes(needle)) {
    const found: Occurrence[] = [];
    let index = haystack.indexOf(needle);
    while (index !== -1) {
      found.push({ start: index, end: index + needle.length });
      index = haystack.indexOf(needle, index + needle.length);
    }
    return found;
  }

  const compactHaystack = stripWhitespaceWithMap(haystack);
  const compactNeedle = stripWhitespaceWithMap(needle).text;
  if (!compactNeedle) return [];

  const found: Occurrence[] = [];
  let index = compactHaystack.text.indexOf(compactNeedle);
  while (index !== -1) {
    const start = compactHaystack.map[index];
    const end = compactHaystack.map[index + compactNeedle.length - 1] + 1;
    found.push({ start, end });
    index = compactHaystack.text.indexOf(compactNeedle, index + compactNeedle.length);
  }
  return found;
}

/** Character offset of (node, offset) in `container`'s concatenated text. */
export function textContentOffset(
  container: Node,
  node: Node,
  offset: number
): number | null {
  if (!container.contains(node)) return null;
  const doc = container.ownerDocument ?? document;
  const walker = doc.createTreeWalker(container, NodeFilter.SHOW_TEXT);
  let consumed = 0;
  let current = walker.nextNode();
  while (current) {
    if (current === node) {
      return consumed + Math.max(0, Math.min(offset, (current as Text).data.length));
    }
    consumed += (current as Text).data.length;
    current = walker.nextNode();
  }
  // The end boundary can point at the container itself; treat it as its end.
  return node === container ? consumed : null;
}

const textNodeCache = new WeakMap<Node, Text[]>();

function textNodes(container: Node): Text[] {
  const cached = textNodeCache.get(container);
  if (cached) return cached;
  const nodes: Text[] = [];
  const doc = container.ownerDocument ?? document;
  const walker = doc.createTreeWalker(container, NodeFilter.SHOW_TEXT);
  let current = walker.nextNode();
  while (current) {
    nodes.push(current as Text);
    current = walker.nextNode();
  }
  textNodeCache.set(container, nodes);
  return nodes;
}

/**
 * Resolve the `occurrenceIndex`-th occurrence of `text` inside `container` to a
 * DOM Range, or null when the passage is no longer rendered.
 */
export function rangeForOccurrence(
  container: Node,
  text: string,
  occurrenceIndex = 0
): Range | null {
  if (!text || !container) return null;

  const nodes = textNodes(container);
  if (nodes.length === 0) return null;

  const full = nodes.map((node) => node.data).join("");
  const matches = findOccurrences(full, text);
  const match = matches[occurrenceIndex];
  if (!match) return null;

  const locate = (offset: number): { node: Text; offset: number } | null => {
    let consumed = 0;
    for (const node of nodes) {
      const next = consumed + node.data.length;
      // At a boundary prefer the start of the following node, so a range that
      // begins exactly at a block edge does not end up at the previous end.
      if (offset < next) {
        return { node, offset: Math.max(0, offset - consumed) };
      }
      consumed = next;
    }
    const last = nodes[nodes.length - 1];
    return last ? { node: last, offset: last.data.length } : null;
  };

  const startPoint = locate(match.start);
  const endPoint = locate(match.end);
  if (!startPoint || !endPoint) return null;

  const doc = container.ownerDocument ?? document;
  const range = doc.createRange();
  range.setStart(startPoint.node, startPoint.offset);
  range.setEnd(endPoint.node, endPoint.offset);
  return range;
}
