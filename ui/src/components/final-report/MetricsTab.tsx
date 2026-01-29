import { Badge } from "@/components/ui/badge"
import type { TrainingAgentState } from "@/types/agent"
import { formatNumber, formatPercent } from "@/lib/utils"
import { Section, MetricRow } from "./shared"
import { getIterationMetrics } from "./utils"

interface MetricsTabProps {
  agentState: TrainingAgentState
}

export function MetricsTab({ agentState }: MetricsTabProps) {
  const metrics = agentState.training_metrics
  const lastIteration = metrics?.iterations?.[metrics.iterations.length - 1]
  const lastIterMetrics = lastIteration ? getIterationMetrics(lastIteration) : null
  const testR2 = metrics?.test_r2 ?? lastIterMetrics?.test_r2
  const testRmse = metrics?.test_rmse ?? lastIterMetrics?.test_rmse
  const testMae = metrics?.test_mae ?? lastIterMetrics?.test_mae
  const valR2 = metrics?.val_r2 ?? lastIterMetrics?.val_r2
  const valRmse = metrics?.val_rmse ?? lastIterMetrics?.val_rmse
  const valMae = metrics?.val_mae ?? lastIterMetrics?.val_mae

  const hasClassificationMetrics = metrics?.test_accuracy != null || metrics?.test_roc_auc != null
  const hasRegressionMetrics = testR2 != null || valR2 != null

  return (
    <div className="space-y-8">
      {/* Validation vs Test comparison */}
      <div className="grid md:grid-cols-2 gap-6">
        <Section title="Validation Metrics">
          <div className="space-y-4">
            {hasClassificationMetrics ? (
              <>
                <MetricRow label="Accuracy" value={formatPercent(metrics?.val_accuracy)} />
                <MetricRow label="ROC-AUC" value={formatNumber(metrics?.val_roc_auc, 3)} />
              </>
            ) : hasRegressionMetrics ? (
              <>
                <MetricRow label="R² Score" value={formatNumber(valR2, 4)} />
                <MetricRow label="RMSE" value={formatNumber(valRmse, 2)} />
                <MetricRow label="MAE" value={formatNumber(valMae, 2)} />
              </>
            ) : (
              <p className="text-sm text-muted-foreground">No validation metrics available</p>
            )}
          </div>
        </Section>

        <Section title="Test Metrics">
          <div className="space-y-4">
            {hasClassificationMetrics ? (
              <>
                <MetricRow label="Accuracy" value={formatPercent(metrics?.test_accuracy)} highlight />
                <MetricRow label="ROC-AUC" value={formatNumber(metrics?.test_roc_auc, 3)} highlight />
              </>
            ) : hasRegressionMetrics ? (
              <>
                <MetricRow label="R² Score" value={formatNumber(testR2, 4)} highlight />
                <MetricRow label="RMSE" value={formatNumber(testRmse, 2)} highlight />
                <MetricRow label="MAE" value={formatNumber(testMae, 2)} highlight />
              </>
            ) : (
              <div className="space-y-2">
                <MetricRow label="Status" value={metrics?.success ? "Success" : "Completed"} highlight />
                <MetricRow label="Model" value={metrics?.model_name || "N/A"} />
              </div>
            )}
          </div>
        </Section>
      </div>

      {/* Iteration History */}
      <Section title="Training Iterations">
        <div className="space-y-2">
          {metrics?.iterations && metrics.iterations.length > 0 ? (
            metrics.iterations.map((iter, i) => {
              const iterMetrics = getIterationMetrics(iter)
              const isBest = i === (metrics.iterations?.length || 1) - 1
              const hasIterClassificationMetrics =
                iterMetrics.val_accuracy != null || iterMetrics.val_roc_auc != null
              const hasIterRegressionMetrics =
                iterMetrics.val_r2 != null || iterMetrics.test_r2 != null

              return (
                <div
                  key={i}
                  className={`p-4 rounded-xl ${
                    isBest ? "bg-foreground/5 ring-1 ring-foreground/10" : "bg-muted/30"
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <span className="font-medium">Iteration {iter.iteration ?? i + 1}</span>
                      {isBest && (
                        <Badge variant="secondary" className="text-xs">
                          Best
                        </Badge>
                      )}
                      {iterMetrics.success === false && (
                        <Badge variant="destructive" className="text-xs">
                          Failed
                        </Badge>
                      )}
                    </div>
                    <div className="flex gap-6 text-sm">
                      {hasIterClassificationMetrics ? (
                        <>
                          <span className="text-muted-foreground">
                            Accuracy:{" "}
                            <span className="text-foreground font-medium">
                              {formatPercent(iterMetrics.val_accuracy)}
                            </span>
                          </span>
                          <span className="text-muted-foreground">
                            AUC:{" "}
                            <span className="text-foreground font-medium">
                              {formatNumber(iterMetrics.val_roc_auc, 3)}
                            </span>
                          </span>
                        </>
                      ) : hasIterRegressionMetrics ? (
                        <>
                          <span className="text-muted-foreground">
                            Val R²:{" "}
                            <span className="text-foreground font-medium">
                              {formatNumber(iterMetrics.val_r2, 4)}
                            </span>
                          </span>
                          <span className="text-muted-foreground">
                            Test R²:{" "}
                            <span className="text-foreground font-medium">
                              {formatNumber(iterMetrics.test_r2, 4)}
                            </span>
                          </span>
                        </>
                      ) : (
                        <span className="text-muted-foreground">
                          {iterMetrics.success === false ? "Failed" : "Completed"}
                        </span>
                      )}
                    </div>
                  </div>

                  {/* Show model name and tool used */}
                  <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
                    {iterMetrics.model_name && (
                      <span>
                        Model: <code className="text-foreground">{iterMetrics.model_name}</code>
                      </span>
                    )}
                    {iterMetrics.tool && (
                      <span>
                        Tool: <code className="text-foreground">{iterMetrics.tool}</code>
                      </span>
                    )}
                  </div>

                  {/* Show error if present */}
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
                <Badge variant="secondary" className="text-xs">
                  Best
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
