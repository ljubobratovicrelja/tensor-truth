/** Extract short model ID from full model name (e.g., "BAAI/bge-m3" -> "bge-m3") */
export function getShortModelId(modelName: string | undefined): string {
  if (!modelName) return "";
  const parts = modelName.split("/");
  return parts[parts.length - 1].toLowerCase();
}

/** Infer doc_type and sort_order from module name pattern */
export function inferDocType(moduleName: string): {
  doc_type: string;
  sort_order: number;
} {
  const name = moduleName.toLowerCase();
  if (name.startsWith("book_")) {
    return { doc_type: "book", sort_order: 1 };
  }
  if (name.startsWith("papers_") || name.startsWith("paper_")) {
    return { doc_type: "paper", sort_order: 2 };
  }
  if (name.startsWith("library_")) {
    return { doc_type: "library_doc", sort_order: 3 };
  }
  return { doc_type: "unknown", sort_order: 4 };
}

/** Generate display name from module name (e.g., "book_deep_learning" -> "Deep Learning") */
export function generateDisplayName(moduleName: string): string {
  // Remove prefix (book_, library_, papers_, paper_)
  const name = moduleName
    .replace(/^book_/i, "")
    .replace(/^papers?_/i, "")
    .replace(/^library_/i, "");
  // Convert underscores to spaces and title case
  return name.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}
