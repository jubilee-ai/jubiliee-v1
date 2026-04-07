import { Badge } from "@/components/ui/badge"
import type { TrainingAgentState } from "@/types/agent"
import { Section, MetricRow } from "./shared"
import { getIterationMetrics } from "./utils"
import {
  getIterationInlinePartsFromIter,
  getValidationTestRows,
  resolveModelFamily,
} from "./metricDefs"

interface MetricsTabProps {
  agentState: TrainingAgentState
}

export function MetricsTab({ agentState }: MetricsTabProps) {
  const metrics = agentState.training_metrics
  const family = resolveModelFamily(metrics, agentState)
  const { left, right } = getValidationTestRows(metrics, agentState)
  const bestIter = metrics?.best_iteration
  const bestIterName =
    typeof bestIter === "object" && bestIter !== null && !Array.isArray(bestIter)
      ? (bestIter as Record<string, unknown>).model_name
      : undefined

  return (
    <div className="space-y-8">
      <div className="grid md:grid-cols-2 gap-6 min-w-0 [&>div]:min-w-0">
        <Section title={left.title}>
          <div className="space-y-4">
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
          <div className="space-y-4">
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
        <p className="text-xs text-muted-foreground -mt-2 mb-3">
          Best (val) picks the strongest validation score: ROC-AUC, then accuracy for classification;
          R² for regression; silhouette / Davies-Bouldin for unsupervised (as logged).
        </p>
        <div className="space-y-2">
          {metrics?.iterations && metrics.iterations.length > 0 ? (
            metrics.iterations.map((iter, i) => {
              const iterMetrics = getIterationMetrics(iter)
              const isBest = bestIterName
                ? iterMetrics.model_name === bestIterName
                : i === (metrics.iterations?.length || 1) - 1
              const inlineParts = getIterationInlinePartsFromIter(iter, family)

              return (
                <div
                  key={i}
                  className={`p-4 rounded-xl ${
                    isBest ? "bg-foreground/5 ring-1 ring-foreground/10" : "bg-muted/30"
                  }`}
                >
                  <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between sm:gap-4 min-w-0">
                    <div className="flex flex-wrap items-center gap-x-3 gap-y-1 shrink-0 min-w-0">
                      <span className="font-medium">Iteration {iter.iteration ?? i + 1}</span>
                      {isBest && (
                        <Badge
                          variant="secondary"
                          className="text-xs"
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
                        <Badge variant="destructive" className="text-xs">
                          Failed
                        </Badge>
                      )}
                    </div>
                    <div className="flex flex-wrap gap-x-4 gap-y-1 text-sm min-w-0 sm:justify-end">
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
                    <div className="mt-2 text-xs text-muted-foreground min-w-0 [overflow-wrap:anywhere]">
                      Model:{" "}
                      <code className="text-foreground break-all align-baseline">{iterMetrics.model_name}</code>
                    </div>
                  )}

                  {iterMetrics.error && (
                    <div className="mt-2 text-xs text-destructive bg-destructive/10 rounded px-2 py-1">
                      {iterMetrics.error}
                    </div>
                  )}
                </div>
              )
            })
          ) : (
            <div className="p-4 rounded-xl bg-foreground/5 ring-1 ring-foreground/10">
              <div className="flex items-center gap-3">
                <span className="font-medium">Iteration 1</span>
                <Badge
                  variant="secondary"
                  className="text-xs"
                  title="Highest validation score for task type"
                >
                  Best (val)
                </Badge>
              </div>
              <p className="text-sm text-muted-foreground mt-2">Training completed in 1 iteration</p>
            </div>
          )}
        </div>
      </Section>
    </div>
  )
}
