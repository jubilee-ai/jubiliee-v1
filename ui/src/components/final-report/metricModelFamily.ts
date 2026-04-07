import type { TrainingAgentState, TrainingMetrics } from "@/types/agent"

export type ModelFamily = "classification" | "regression" | "unsupervised"

export function pickNumber(record: Record<string, unknown>, key: string): number | undefined {
  const v = record[key]
  return typeof v === "number" && Number.isFinite(v) ? v : undefined
}

/** Merge top-level training_metrics with best_iteration dict for fallbacks */
export function mergeTrainingMetricSources(
  metrics: TrainingMetrics | null | undefined,
): Record<string, unknown> {
  if (!metrics) return {}
  const bi = metrics.best_iteration
  const best =
    typeof bi === "object" && bi !== null && !Array.isArray(bi) ? (bi as Record<string, unknown>) : {}
  const top = metrics as Record<string, unknown>
  const out: Record<string, unknown> = { ...best, ...top }
  if (out.davies_bouldin == null && pickNumber(out, "davies_bouldin_score") != null) {
    out.davies_bouldin = pickNumber(out, "davies_bouldin_score")
  }
  return out
}

export function resolveModelFamily(
  metrics: TrainingMetrics | null | undefined,
  agentState?: Pick<TrainingAgentState, "selected_model" | "label_definition">,
): ModelFamily {
  const isUnsupervisedFlow =
    agentState?.selected_model === "unsupervised" ||
    agentState?.label_definition?.split_strategy === "none"

  const merged = mergeTrainingMetricSources(metrics)

  const hasClassification =
    pickNumber(merged, "test_accuracy") != null || pickNumber(merged, "test_roc_auc") != null
  const hasRegression =
    pickNumber(merged, "test_r2") != null ||
    pickNumber(merged, "test_rmse") != null ||
    pickNumber(merged, "test_mae") != null ||
    (!isUnsupervisedFlow && pickNumber(merged, "val_r2") != null)
  const hasUnsupervised =
    pickNumber(merged, "silhouette_score") != null ||
    pickNumber(merged, "davies_bouldin") != null ||
    pickNumber(merged, "inertia") != null ||
    pickNumber(merged, "reconstruction_loss") != null

  if (hasClassification) return "classification"
  if (hasRegression && !isUnsupervisedFlow) return "regression"
  if (hasUnsupervised || isUnsupervisedFlow) return "unsupervised"
  if (hasRegression) return "regression"
  return "classification"
}
