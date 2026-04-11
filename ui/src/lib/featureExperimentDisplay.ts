import type { ExperimentScoutRow } from "@/types/agent"

export function pickPrimaryMetricKey(
  metrics: Record<string, number> | undefined,
): "r2" | "roc_auc" | "accuracy" | null {
  if (!metrics) return null
  if (metrics.r2 != null) return "r2"
  if (metrics.roc_auc != null) return "roc_auc"
  if (metrics.accuracy != null) return "accuracy"
  return null
}

export function primaryMetricLabel(key: string): string {
  if (key === "r2") return "R²"
  if (key === "roc_auc") return "ROC-AUC"
  if (key === "accuracy") return "Accuracy"
  return key
}

export function formatModelFamily(mf: string | undefined): string {
  if (mf === "hgb") return "HGB"
  if (mf === "rf") return "RF"
  return mf ?? "—"
}

/** Sort scouts by primary validation metric (best first). */
export function sortScoutsByPrimary(grid: ExperimentScoutRow[]): ExperimentScoutRow[] {
  return [...grid].sort((a, b) => {
    const ka = pickPrimaryMetricKey(a.metrics)
    const kb = pickPrimaryMetricKey(b.metrics)
    const va = ka && a.metrics ? (a.metrics[ka] ?? -Infinity) : -Infinity
    const vb = kb && b.metrics ? (b.metrics[kb] ?? -Infinity) : -Infinity
    return vb - va
  })
}

export function featureRankingEntries(
  rankings: Record<string, number> | null | undefined,
  limit = 40,
): Array<{ feature: string; score: number }> {
  if (!rankings || typeof rankings !== "object") return []
  return Object.entries(rankings)
    .filter(([, v]) => typeof v === "number" && Number.isFinite(v))
    .sort((a, b) => b[1] - a[1])
    .slice(0, limit)
    .map(([feature, score]) => ({ feature, score }))
}
