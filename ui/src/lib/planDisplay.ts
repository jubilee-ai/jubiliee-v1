/** Short, human-friendly plan summary for the UI (no jargon). */

const MAX_GOAL_CHARS = 140

export function truncateGoalForCard(goal: string, maxChars = MAX_GOAL_CHARS): string {
  const g = goal.replace(/\s+/g, " ").trim()
  if (g.length <= maxChars) return g
  const cut = g.slice(0, maxChars)
  const lastSpace = cut.lastIndexOf(" ")
  const base = lastSpace > 40 ? cut.slice(0, lastSpace) : cut
  return `${base.trim()}…`
}

/** Remove leaked tool JSON (full propose_training_plan payload) from streamed agent text. */
export function stripLeakedPlanJson(content: string): string {
  let t = content.trim()

  const fence = /^```(?:json)?\s*([\s\S]*?)```\s*/m
  let fm = t.match(fence)
  while (fm) {
    const inner = fm[1]?.trim() ?? ""
    try {
      const j = JSON.parse(inner) as { goal?: unknown; dataset_refs?: unknown }
      const refsOk =
        j.dataset_refs === undefined ||
        (Array.isArray(j.dataset_refs) &&
          (j.dataset_refs as unknown[]).every((r) => typeof r === "string"))
      if (typeof j.goal === "string" && refsOk) {
        t = t.slice(fm[0].length).trim()
        fm = t.match(fence)
        continue
      }
    } catch {
      break
    }
    break
  }

  if (t.startsWith("{") && (t.includes('"dataset_refs"') || t.includes('"goal"'))) {
    let depth = 0
    let i = 0
    for (; i < t.length; i++) {
      const c = t[i]
      if (c === "{") depth++
      if (c === "}") {
        depth--
        if (depth === 0) {
          i++
          break
        }
      }
    }
    if (i > 1 && depth === 0) {
      t = t.slice(i).trim().replace(/^[\s\n]+/, "")
    }
  }

  return t
}

/** True when the whole (or partial stream) blob is the training plan JSON from the tool. */
export function looksLikeLeakedPlanJson(content: string): boolean {
  const t = content.trim()
  if (!t.startsWith("{") || !t.includes('"goal"')) {
    return false
  }
  try {
    const j = JSON.parse(t) as { goal?: unknown; dataset_refs?: unknown }
    if (typeof j.goal !== "string") return false
    if (j.dataset_refs === undefined) return true
    return Array.isArray(j.dataset_refs)
  } catch {
    return t.length > 120 && /"recap_steps"\s*:/.test(t)
  }
}
