/**
 * Builds expandable markdown for pipeline step chat cards from stream events.
 * Keeps the visible headline short; details explain what happened and why.
 */

import type { AgentStreamEvent } from "@/lib/api"

function fmtNum(n: unknown): string {
  if (typeof n === "number" && Number.isFinite(n)) return n.toLocaleString()
  return String(n ?? "")
}

function formatTransformLine(t: unknown): string {
  if (!t || typeof t !== "object") return `- ${String(t)}`
  const r = t as Record<string, unknown>
  const tool = String(r.tool ?? r.op ?? "transform").replace(/_tool$/, "")
  const args = r.args as Record<string, unknown> | undefined
  let extra = ""
  if (args) {
    if (Array.isArray(args.columns)) extra = ` **${args.columns.join(", ")}**`
    else if (args.column) extra = ` **${args.column}**`
    if (args.dataset_ref) extra += ` (${String(args.dataset_ref).slice(0, 40)}…)`
    if (args.value !== undefined) extra += ` → ${String(args.value)}`
    if (args.strategy) extra += ` (${String(args.strategy)})`
  }
  const res = r.result != null ? ` → ${String(r.result)}` : ""
  return `- \`${tool}\`${extra}${res}`
}

/** Top correlations for display */
function topCorrs(rows: unknown, limit: number): string[] {
  if (!Array.isArray(rows)) return []
  const out: string[] = []
  for (const c of rows.slice(0, limit)) {
    if (!c || typeof c !== "object") continue
    const o = c as { feature?: string; correlation?: number }
    if (o.feature != null && o.correlation != null) {
      out.push(`- **${o.feature}**: ${Number(o.correlation).toFixed(3)}`)
    }
  }
  return out
}

export function buildStepDetailMarkdown(nodeName: string, event: AgentStreamEvent): string {
  const summary = event.summary as Record<string, unknown> | undefined
  const details = event.details as Record<string, unknown> | undefined
  const state = event.state as Record<string, unknown> | undefined
  const blocks: string[] = []

  if (nodeName === "cleaning" || nodeName === "cleaning_and_standardization") {
    const transforms = summary?.transformations as unknown[] | undefined
    if (transforms && transforms.length > 0) {
      blocks.push("### What we changed")
      transforms.slice(0, 25).forEach((t) => blocks.push(formatTransformLine(t)))
      if (transforms.length > 25) {
        blocks.push(`\n*…and ${transforms.length - 25} more*`)
      }
    } else {
      blocks.push("*No structural transformations were required.*")
    }
    const reason = summary?.reason
    if (typeof reason === "string" && reason.trim()) {
      blocks.push(`\n### Summary\n${reason}`)
    }
  }

  if (nodeName === "feature_selection_specification" || nodeName === "feature_engineering_executor") {
    const trace = (state?.analysis_trace ?? details?.analysis_trace) as Array<Record<string, unknown>> | undefined
    const ks = trace?.[0]?.key_stats as Record<string, unknown> | undefined
    if (ks?.summary_text && typeof ks.summary_text === "string" && ks.summary_text.trim()) {
      blocks.push(`### Analysis summary\n${ks.summary_text.trim()}`)
    }
    const overview = ks?.dataset_overview as Record<string, unknown> | undefined
    if (overview?.rows != null) {
      blocks.push(
        `### Dataset\n**${fmtNum(overview.rows)}** rows × **${fmtNum(overview.columns)}** columns ` +
          `(numeric ${fmtNum(overview.numeric_columns)}, categorical ${fmtNum(overview.categorical_columns)})`,
      )
    }
    const leakage = ks?.leakage_warnings as unknown[] | undefined
    if (leakage && leakage.length > 0) {
      blocks.push("### Leakage checks\n" + leakage.map((w) => `- ⚠️ ${String(w)}`).join("\n"))
    }
    const corrs = topCorrs(ks?.feature_correlations, 8)
    if (corrs.length > 0) {
      blocks.push("### Strongest signals vs target\n" + corrs.join("\n"))
    }
    const names = summary?.feature_names as unknown[] | undefined
    const fs = state?.feature_spec as { features?: Array<{ name?: string }> } | undefined
    const fromSpec = fs?.features?.map((f) => f.name).filter(Boolean) as string[] | undefined
    const nameList = Array.isArray(names) && names.length > 0 ? names : fromSpec
    const nf = summary?.num_features ?? (Array.isArray(nameList) ? nameList.length : undefined)
    if (typeof nf === "number" && nf > 0) {
      blocks.push(`### Features selected (${nf})\n`)
      if (Array.isArray(nameList) && nameList.length > 0) {
        blocks.push(nameList.slice(0, 24).map((n) => `- \`${String(n)}\``).join("\n"))
        if (nameList.length > 24) blocks.push(`\n*…+${nameList.length - 24} more*`)
      }
    }
    if (nodeName === "feature_engineering_executor") {
      const shapes = summary?.shapes as Record<string, unknown> | undefined
      const fmtShape = (s: unknown) =>
        Array.isArray(s) && s.length >= 2 ? `${s[0]} × ${s[1]}` : String(s ?? "?")
      if (shapes && Object.keys(shapes).length > 0) {
        blocks.push(
          "### Matrix shapes\n" +
            [
              shapes.train != null ? `Train **${fmtShape(shapes.train)}**` : "",
              shapes.val != null ? `Val **${fmtShape(shapes.val)}**` : "",
              shapes.test != null ? `Test **${fmtShape(shapes.test)}**` : "",
            ]
              .filter(Boolean)
              .join(" · "),
        )
      }
      const created = summary?.features_created as unknown[] | undefined
      if (Array.isArray(created) && created.length > 0) {
        blocks.push(
          `### Columns after encoding (${created.length})\n` +
            created
              .slice(0, 20)
              .map((c) => `\`${String(c)}\``)
              .join(", ") +
            (created.length > 20 ? ` *…+${created.length - 20}*` : ""),
        )
      }
    }
  }

  if (nodeName === "training") {
    const dm = details as Record<string, unknown> | undefined
    const summ = typeof dm?.summary === "string" ? dm.summary : undefined
    if (summ?.trim()) {
      blocks.push("### What happened\n" + summ.trim().slice(0, 1200) + (summ.length > 1200 ? "…" : ""))
    }
    const iters = dm?.iterations as Array<Record<string, unknown>> | undefined
    if (Array.isArray(iters) && iters.length > 0) {
      blocks.push("### Models tried (validation)\n")
      iters.slice(0, 15).forEach((it, i) => {
        const name = it.model_name ?? it.tool ?? "model"
        const ok = it.success !== false ? "✓" : "✗"
        const bits: string[] = []
        if (it.val_accuracy != null) bits.push(`val acc ${(Number(it.val_accuracy) * 100).toFixed(1)}%`)
        if (it.val_roc_auc != null) bits.push(`val AUC ${Number(it.val_roc_auc).toFixed(3)}`)
        if (it.val_r2 != null) bits.push(`val R² ${Number(it.val_r2).toFixed(4)}`)
        blocks.push(`${i + 1}. ${ok} **${String(name)}**${bits.length ? ` — ${bits.join(", ")}` : ""}`)
      })
    }
    const rec = dm?.recommendations
    if (typeof rec === "string" && rec.trim()) {
      blocks.push(`\n### Notes\n${rec.trim().slice(0, 800)}`)
    }
  }

  if (nodeName === "label_split_definition" && summary) {
    const g = summary.grain
    if (g) blocks.push(`**Grain:** ${String(g)}`)
  }

  return blocks.filter(Boolean).join("\n\n").trim()
}
