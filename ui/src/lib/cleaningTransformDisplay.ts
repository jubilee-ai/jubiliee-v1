/**
 * Shared formatting for cleaning step tool calls (UI + markdown).
 * Row filters use `predicate`; column ops use `columns` / `column`.
 */

/** Text after the output dataset ref from format_result() — e.g. "(kept 100, removed 5)". */
export function parseToolResultAfterRef(result: string | undefined): string {
  if (!result) return ""
  const m = String(result).match(/→\s*`[^`]+`\s*(.*)/s)
  return m?.[1]?.trim() ?? ""
}

export function formatCleaningTransformationParts(t: unknown): { primary: string; secondary: string } {
  if (!t || typeof t !== "object") return { primary: "", secondary: "" }
  const transform = t as Record<string, unknown>
  const args = transform.args as Record<string, unknown> | undefined
  const result = transform.result as string | undefined

  let primary = ""
  const secondaryBits: string[] = []

  if (args) {
    const pred = args.predicate != null ? String(args.predicate).trim() : ""
    if (pred) primary = pred
    else if (args.columns)
      primary = Array.isArray(args.columns) ? args.columns.join(", ") : String(args.columns)
    else if (args.column) primary = String(args.column)

    if (args.dataset_ref) secondaryBits.push(`(${String(args.dataset_ref).slice(0, 40)}…)`)
    if (args.value !== undefined) secondaryBits.push(`= ${String(args.value)}`)
    if (args.strategy) secondaryBits.push(`(${String(args.strategy)})`)
  }

  const tail = parseToolResultAfterRef(result)
  if (tail) secondaryBits.push(tail)

  return { primary, secondary: secondaryBits.join(" ").trim() }
}

export function formatCleaningTransformationMarkdownLine(t: unknown): string {
  if (!t || typeof t !== "object") return `- ${String(t)}`
  const r = t as Record<string, unknown>
  const tool = String(r.tool ?? r.op ?? "transform").replace(/_tool$/, "")
  const { primary, secondary } = formatCleaningTransformationParts(t)
  let line = `- \`${tool}\``
  if (primary) line += ` **${primary}**`
  if (secondary) line += ` ${secondary}`
  return line
}
