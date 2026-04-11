import type { TrainingAgentState, TrainingIteration, TrainingMetrics } from "@/types/agent"
import { formatNumber, formatPercent } from "@/lib/utils"
import { getIterationMetrics } from "./utils"
import {
  type ModelFamily,
  mergeTrainingMetricSources,
  pickNumber,
  resolveModelFamily,
} from "./metricModelFamily"

export type { ModelFamily }
export { mergeTrainingMetricSources, pickNumber, resolveModelFamily } from "./metricModelFamily"

type MetricFormat = "percent" | "number"

type MetricDef = {
  label: string
  format: MetricFormat
  decimals: number
  family: ModelFamily
}

/** Single source of truth for metric labels and formatting */
const DEF: Record<string, MetricDef> = {
  val_accuracy: { label: "Accuracy", format: "percent", decimals: 2, family: "classification" },
  val_roc_auc: { label: "ROC-AUC", format: "number", decimals: 3, family: "classification" },
  test_accuracy: { label: "Accuracy", format: "percent", decimals: 2, family: "classification" },
  test_roc_auc: { label: "ROC-AUC", format: "number", decimals: 3, family: "classification" },
  val_r2: { label: "R² Score", format: "number", decimals: 4, family: "regression" },
  val_rmse: { label: "RMSE", format: "number", decimals: 2, family: "regression" },
  val_mae: { label: "MAE", format: "number", decimals: 2, family: "regression" },
  test_r2: { label: "R² Score", format: "number", decimals: 4, family: "regression" },
  test_rmse: { label: "RMSE", format: "number", decimals: 2, family: "regression" },
  test_mae: { label: "MAE", format: "number", decimals: 2, family: "regression" },
  silhouette_score: { label: "Silhouette", format: "number", decimals: 4, family: "unsupervised" },
  davies_bouldin: { label: "Davies-Bouldin", format: "number", decimals: 4, family: "unsupervised" },
  inertia: { label: "Inertia", format: "number", decimals: 1, family: "unsupervised" },
  reconstruction_loss: { label: "Reconstruction loss", format: "number", decimals: 4, family: "unsupervised" },
  val_silhouette_score: { label: "Silhouette (holdout)", format: "number", decimals: 4, family: "unsupervised" },
  val_davies_bouldin_score: { label: "Davies-Bouldin (holdout)", format: "number", decimals: 4, family: "unsupervised" },
  n_clusters_or_groups: { label: "Clusters / groups", format: "number", decimals: 0, family: "unsupervised" },
}

const UNSUPERVISED_ORDER = [
  "silhouette_score",
  "davies_bouldin",
  "inertia",
  "reconstruction_loss",
  "n_clusters_or_groups",
] as const

const HOLDOUT_KEYS = ["val_silhouette_score", "val_davies_bouldin_score"] as const

function formatValue(key: string, value: number): string {
  const meta = DEF[key]
  if (!meta) {
    return Number.isInteger(value) ? String(value) : formatNumber(value, 4)
  }
  if (meta.format === "percent") return formatPercent(value)
  return formatNumber(value, meta.decimals)
}

export type HeroMetric = { label: string; value: string; highlight?: boolean }

export function getHeroMetrics(
  metrics: TrainingMetrics | null | undefined,
  agentState: Pick<TrainingAgentState, "selected_model" | "label_definition">,
): HeroMetric[] {
  const family = resolveModelFamily(metrics, agentState)
  const merged = mergeTrainingMetricSources(metrics)

  if (family === "classification") {
    const ta = pickNumber(merged, "test_accuracy")
    const tr = pickNumber(merged, "test_roc_auc")
    const rows: HeroMetric[] = []
    if (ta != null) rows.push({ label: "Test Accuracy", value: formatValue("test_accuracy", ta), highlight: true })
    if (tr != null) rows.push({ label: "Test ROC-AUC", value: formatValue("test_roc_auc", tr), highlight: true })
    return rows.length ? rows : [{ label: "Status", value: metrics?.success ? "Success" : "Complete", highlight: true }]
  }

  if (family === "regression") {
    const r2 = pickNumber(merged, "test_r2")
    const rmse = pickNumber(merged, "test_rmse")
    const rows: HeroMetric[] = []
    if (r2 != null) rows.push({ label: "Test R²", value: formatValue("test_r2", r2), highlight: true })
    if (rmse != null) rows.push({ label: "Test RMSE", value: formatValue("test_rmse", rmse), highlight: true })
    return rows.length ? rows : [{ label: "Status", value: metrics?.success ? "Success" : "Complete", highlight: true }]
  }

  const unsup: HeroMetric[] = []
  for (const key of UNSUPERVISED_ORDER) {
    const v = pickNumber(merged, key)
    if (v != null && DEF[key]) {
      unsup.push({ label: DEF[key].label, value: formatValue(key, v), highlight: unsup.length < 2 })
      if (unsup.length >= 2) break
    }
  }
  if (unsup.length) return unsup
  return [
    { label: "Status", value: metrics?.success ? "Success" : "Complete", highlight: true },
    {
      label: "Model",
      value: String(metrics?.model_type || agentState.selected_model || "N/A"),
    },
  ]
}

export type MetricRowSpec = { key: string; label: string; value: string; highlight?: boolean }

export function getValidationTestRows(
  metrics: TrainingMetrics | null | undefined,
  agentState: Pick<TrainingAgentState, "selected_model" | "label_definition">,
): { family: ModelFamily; left: { title: string; rows: MetricRowSpec[] }; right: { title: string; rows: MetricRowSpec[] } } {
  const family = resolveModelFamily(metrics, agentState)
  const merged = mergeTrainingMetricSources(metrics)

  if (family === "classification") {
    const left: MetricRowSpec[] = []
    const va = pickNumber(merged, "val_accuracy")
    const vr = pickNumber(merged, "val_roc_auc")
    if (va != null) left.push({ key: "val_accuracy", label: DEF.val_accuracy.label, value: formatValue("val_accuracy", va) })
    if (vr != null) left.push({ key: "val_roc_auc", label: DEF.val_roc_auc.label, value: formatValue("val_roc_auc", vr) })
    const right: MetricRowSpec[] = []
    const ta = pickNumber(merged, "test_accuracy")
    const tr = pickNumber(merged, "test_roc_auc")
    if (ta != null) right.push({ key: "test_accuracy", label: DEF.test_accuracy.label, value: formatValue("test_accuracy", ta), highlight: true })
    if (tr != null) right.push({ key: "test_roc_auc", label: DEF.test_roc_auc.label, value: formatValue("test_roc_auc", tr), highlight: true })
    return {
      family,
      left: { title: "Validation metrics", rows: left },
      right: { title: "Test metrics", rows: right },
    }
  }

  if (family === "regression") {
    const left: MetricRowSpec[] = []
    for (const key of ["val_r2", "val_rmse", "val_mae"] as const) {
      const v = pickNumber(merged, key)
      if (v != null && DEF[key]) left.push({ key, label: DEF[key].label, value: formatValue(key, v) })
    }
    const right: MetricRowSpec[] = []
    for (const key of ["test_r2", "test_rmse", "test_mae"] as const) {
      const v = pickNumber(merged, key)
      if (v != null && DEF[key]) right.push({ key, label: DEF[key].label, value: formatValue(key, v), highlight: true })
    }
    return {
      family,
      left: { title: "Validation metrics", rows: left },
      right: { title: "Test metrics", rows: right },
    }
  }

  const trainingRows: MetricRowSpec[] = []
  for (const key of UNSUPERVISED_ORDER) {
    const v = pickNumber(merged, key)
    if (v != null && DEF[key]) trainingRows.push({ key, label: DEF[key].label, value: formatValue(key, v) })
  }

  const holdoutRows: MetricRowSpec[] = []
  for (const key of HOLDOUT_KEYS) {
    const v = pickNumber(merged, key)
    if (v != null && DEF[key]) holdoutRows.push({ key, label: DEF[key].label, value: formatValue(key, v), highlight: true })
  }

  return {
    family,
    left: {
      title: holdoutRows.length ? "Training metrics (full fit)" : "Training metrics",
      rows: trainingRows,
    },
    right: {
      title: holdoutRows.length ? "Holdout evaluation" : "Additional",
      rows: holdoutRows.length
        ? holdoutRows
        : [
            {
              key: "model_name",
              label: "Saved model",
              value: String(metrics?.model_name ?? "—"),
              highlight: true,
            },
          ],
    },
  }
}

/** Compact iteration summary for inline lists (Metrics tab) */
export function getIterationInlineParts(
  iter: Record<string, unknown>,
  family: ModelFamily,
): { label: string; value: string }[] {
  const parts: { label: string; value: string }[] = []

  if (family === "classification") {
    const va = pickNumber(iter, "val_accuracy")
    const vr = pickNumber(iter, "val_roc_auc")
    if (va != null) parts.push({ label: "Accuracy", value: formatValue("val_accuracy", va) })
    if (vr != null) parts.push({ label: "ROC-AUC", value: formatValue("val_roc_auc", vr) })
    return parts
  }

  if (family === "regression") {
    const v2 = pickNumber(iter, "val_r2")
    const t2 = pickNumber(iter, "test_r2")
    if (v2 != null) parts.push({ label: "Val R²", value: formatValue("val_r2", v2) })
    if (t2 != null) parts.push({ label: "Test R²", value: formatValue("test_r2", t2) })
    return parts
  }

  const sil = pickNumber(iter, "silhouette_score")
  const db = pickNumber(iter, "davies_bouldin") ?? pickNumber(iter, "davies_bouldin_score")
  if (sil != null) parts.push({ label: "Silhouette", value: formatValue("silhouette_score", sil) })
  if (db != null) parts.push({ label: "D–B", value: formatValue("davies_bouldin", db) })
  if (parts.length === 0) {
    const inc = pickNumber(iter, "inertia")
    if (inc != null) parts.push({ label: "Inertia", value: formatValue("inertia", inc) })
  }
  return parts
}

/** Same as getIterationInlineParts but accepts a TrainingIteration */
export function getIterationInlinePartsFromIter(
  iter: TrainingIteration,
  family: ModelFamily,
): { label: string; value: string }[] {
  return getIterationInlineParts(getIterationMetrics(iter) as Record<string, unknown>, family)
}
