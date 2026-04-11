import { useState } from "react"
import { ChevronDown } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import type { TrainingAgentState } from "@/types/agent"
import { Section, MetricRow } from "./shared"
import { cn } from "@/lib/utils"
import { getIterationMetrics } from "./utils"
import {
  getIterationInlinePartsFromIter,
  getValidationTestRows,
  resolveModelFamily,
} from "./metricDefs"

const INITIAL_ITERATIONS_SHOWN = 3

function TrainingIterationsList({
  metrics,
  family,
  bestIterName,
  expanded,
  onToggleExpanded,
}: {
  metrics: TrainingAgentState["training_metrics"]
  family: ReturnType<typeof resolveModelFamily>
  bestIterName: string | undefined
  expanded: boolean
  onToggleExpanded: () => void
}) {
  const iterations = metrics?.iterations
  const total = iterations?.length ?? 0

  if (!iterations || total === 0) {
    return (
      <div className="p-2.5 rounded-lg bg-foreground/5 ring-1 ring-foreground/10">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-xs font-medium text-foreground">Iteration 1</span>
          <Badge
            variant="secondary"
            className="text-overline px-1.5 py-0 h-5"
            title="Highest validation score for task type"
          >
            Best (val)
          </Badge>
        </div>
        <p className="text-xs text-muted-foreground mt-1.5">Training completed in 1 iteration</p>
      </div>
    )
  }

  const visible =
    expanded || total <= INITIAL_ITERATIONS_SHOWN
      ? iterations
      : iterations.slice(0, INITIAL_ITERATIONS_SHOWN)
  const hiddenCount = Math.max(0, total - INITIAL_ITERATIONS_SHOWN)

  return (
    <div className="space-y-1.5">
      {visible.map((iter, i) => {
        const iterMetrics = getIterationMetrics(iter)
        const isBest = bestIterName
          ? iterMetrics.model_name === bestIterName
          : i === total - 1
        const inlineParts = getIterationInlinePartsFromIter(iter, family)

        return (
          <div
            key={`iter-${i}-${String(iterMetrics.model_name ?? "")}`}
            className={cn(
              "p-2.5 rounded-lg",
              isBest ? "bg-foreground/5 ring-1 ring-foreground/10" : "bg-muted/30",
            )}
          >
            <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between sm:gap-3 min-w-0">
              <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5 shrink-0 min-w-0">
                <span className="text-xs font-medium text-foreground">
                  Iteration {iter.iteration ?? i + 1}
                </span>
                {isBest && (
                  <Badge
                    variant="secondary"
                    className="text-overline px-1.5 py-0 h-5"
                    title={
                      family === "unsupervised"
                        ? "Best iteration by silhouette / Davies-Bouldin (as logged)"
                        : "Highest validation ROC-AUC then accuracy (classification), or val R² (regression)"
                    }
                  >
                    Best (val)
                  </Badge>
                )}
                {iterMetrics.success === false && (
                  <Badge variant="destructive" className="text-overline px-1.5 py-0 h-5">
                    Failed
                  </Badge>
                )}
              </div>
              <div className="flex flex-wrap gap-x-3 gap-y-0.5 text-xs min-w-0 sm:justify-end">
                {inlineParts.length > 0 ? (
                  inlineParts.map((p) => (
                    <span key={p.label} className="text-muted-foreground">
                      {p.label}:{" "}
                      <span className="text-foreground font-medium">{p.value}</span>
                    </span>
                  ))
                ) : (
                  <span className="text-muted-foreground">
                    {iterMetrics.success === false ? "Failed" : "Completed"}
                  </span>
                )}
              </div>
            </div>

            {iterMetrics.model_name && (
              <div className="mt-1.5 text-caption text-muted-foreground min-w-0 [overflow-wrap:anywhere] leading-snug">
                Model:{" "}
                <code className="text-foreground break-all align-baseline text-caption">{iterMetrics.model_name}</code>
              </div>
            )}

            {iterMetrics.error && (
              <div className="mt-1.5 text-caption text-destructive bg-destructive/10 rounded px-2 py-1">
                {iterMetrics.error}
              </div>
            )}
          </div>
        )
      })}

      {hiddenCount > 0 && (
        <button
          type="button"
          onClick={onToggleExpanded}
          aria-expanded={expanded}
          className="flex w-full items-center justify-center gap-1.5 py-2 text-xs text-muted-foreground transition-colors hover:text-foreground"
        >
          <ChevronDown
            className={cn("h-3.5 w-3.5 shrink-0 transition-transform duration-200", expanded && "rotate-180")}
            aria-hidden
          />
          {expanded ? "Show fewer iterations" : `See ${hiddenCount} more iteration${hiddenCount !== 1 ? "s" : ""}`}
        </button>
      )}
    </div>
  )
}

interface MetricsTabProps {
  agentState: TrainingAgentState
}

export function MetricsTab({ agentState }: MetricsTabProps) {
  const metrics = agentState.training_metrics
  const family = resolveModelFamily(metrics, agentState)
  const { left, right } = getValidationTestRows(metrics, agentState)
  const bestIter = metrics?.best_iteration
  const bestIterName: string | undefined = (() => {
    if (typeof bestIter !== "object" || bestIter === null || Array.isArray(bestIter)) return undefined
    const name = (bestIter as Record<string, unknown>).model_name
    return typeof name === "string" ? name : undefined
  })()
  const [iterationsExpanded, setIterationsExpanded] = useState(false)

  return (
    <div className="space-y-6">
      <div className="grid md:grid-cols-2 gap-5 min-w-0 [&>div]:min-w-0">
        <Section title={left.title}>
          <div className="space-y-3">
            {left.rows.length > 0 ? (
              left.rows.map((row) => (
                <MetricRow key={row.key} label={row.label} value={row.value} highlight={row.highlight} />
              ))
            ) : family === "unsupervised" ? (
              <p className="text-sm text-muted-foreground">
                No training diagnostics in state. If training finished, check Trace for iteration logs.
              </p>
            ) : (
              <p className="text-sm text-muted-foreground">No validation metrics available</p>
            )}
          </div>
        </Section>

        <Section title={right.title}>
          <div className="space-y-3">
            {right.rows.length > 0 ? (
              right.rows.map((row) => (
                <MetricRow key={row.key} label={row.label} value={row.value} highlight={row.highlight} />
              ))
            ) : family === "unsupervised" ? null : (
              <div className="space-y-2">
                <MetricRow label="Status" value={metrics?.success ? "Success" : "Completed"} highlight />
                <MetricRow label="Model" value={metrics?.model_name || "N/A"} />
              </div>
            )}
          </div>
        </Section>
      </div>

      <Section title="Training Iterations">
        <TrainingIterationsList
          metrics={metrics}
          family={family}
          bestIterName={bestIterName}
          expanded={iterationsExpanded}
          onToggleExpanded={() => setIterationsExpanded((e) => !e)}
        />
      </Section>
    </div>
  )
}
