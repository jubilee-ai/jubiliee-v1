import { useEffect, useState } from "react"
import {
  Box, Download, Activity, Hash, Calendar, Layers, Target,
  Search, ChevronDown, ChevronRight,
} from "lucide-react"
import { Badge } from "@/components/ui/badge"
import type { ModelType, TrainedModelEntry } from "@/lib/api"
import { getTrainedModels } from "@/lib/api"

interface ModelsPageProps {
  modelTypes: ModelType[]
}

function MetricPill({ label, value }: { label: string; value: number }) {
  const display = value < 1 && value > 0 ? (value * 100).toFixed(1) + "%" : value.toFixed(4)
  return (
    <span className="inline-flex items-center gap-1 rounded-md bg-muted px-2 py-0.5 text-[11px] font-mono">
      <span className="text-muted-foreground">{label}</span>
      <span className="font-medium text-foreground tabular-nums">{display}</span>
    </span>
  )
}

function pickMetrics(metrics: Record<string, number>): Array<{ label: string; value: number }> {
  const priority = [
    ["test_roc_auc", "ROC AUC"],
    ["test_accuracy", "Accuracy"],
    ["val_roc_auc", "Val ROC AUC"],
    ["val_accuracy", "Val Accuracy"],
    ["test_r2", "R\u00b2"],
    ["test_rmse", "RMSE"],
    ["val_r2", "Val R\u00b2"],
    ["val_rmse", "Val RMSE"],
  ] as const

  const result: Array<{ label: string; value: number }> = []
  for (const [key, label] of priority) {
    if (key in metrics && typeof metrics[key] === "number") {
      result.push({ label, value: metrics[key] })
      if (result.length >= 3) break
    }
  }
  return result
}

function FeatureList({ features }: { features: string[] }) {
  const [open, setOpen] = useState(false)
  if (!features.length) return null

  return (
    <div>
      <button
        onClick={() => setOpen(!open)}
        className="inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
      >
        {open ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
        <Layers className="h-3 w-3" />
        <span className="font-medium">{features.length} features</span>
      </button>
      {open && (
        <div className="mt-1.5 flex flex-wrap gap-1">
          {features.map((f) => (
            <span key={f} className="rounded bg-muted px-1.5 py-0.5 text-[10px] font-mono text-muted-foreground">
              {f}
            </span>
          ))}
        </div>
      )}
    </div>
  )
}

function formatDate(iso: string): string {
  if (!iso) return "--"
  try {
    return new Date(iso).toLocaleDateString(undefined, {
      month: "short", day: "numeric", year: "numeric",
    })
  } catch {
    return "--"
  }
}

export function ModelsPage({ modelTypes: _modelTypes }: ModelsPageProps) {
  const [trainedModels, setTrainedModels] = useState<TrainedModelEntry[]>([])
  const [loading, setLoading] = useState(true)
  const [search, setSearch] = useState("")

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    getTrainedModels()
      .then((data) => {
        if (!cancelled) setTrainedModels(Object.values(data))
      })
      .catch(() => {
        if (!cancelled) setTrainedModels([])
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => { cancelled = true }
  }, [])

  const filteredTrained = trainedModels.filter((m) =>
    !search || m.model_name.toLowerCase().includes(search.toLowerCase()) ||
    m.model_type.toLowerCase().includes(search.toLowerCase())
  )

  return (
    <div className="flex-1 overflow-auto">
      <div className="max-w-5xl mx-auto px-8 py-10">
        <div className="flex items-start gap-3">
          <div>
            <span className="text-[10px] font-bold text-muted-foreground tracking-widest uppercase">
              Model Management
            </span>
            <h1 className="font-headline text-3xl font-semibold text-foreground tracking-tight mt-1">
              Model Registry
            </h1>
          </div>
        </div>
        <p className="mt-2 text-muted-foreground text-sm leading-relaxed max-w-lg">
          Monitor deployments, track performance metrics, and manage your model lifecycle.
        </p>

        {/* ── Trained Models ─────────────────────────────────── */}
        <section className="mt-10">
          <div className="flex items-end justify-between">
            <h2 className="text-xs font-bold text-muted-foreground tracking-widest uppercase">
              Trained Models
            </h2>
            {trainedModels.length > 0 && (
              <Badge variant="secondary" className="mb-0.5 tabular-nums">
                {trainedModels.length} model{trainedModels.length !== 1 ? "s" : ""}
              </Badge>
            )}
          </div>

          {trainedModels.length > 0 && (
            <div className="relative mt-3">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
              <input
                type="text"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Filter trained models..."
                className="w-full max-w-sm rounded-lg border border-border bg-card pl-9 pr-3 py-2 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring"
              />
            </div>
          )}

          {loading ? (
            <div className="mt-6 rounded-xl bg-card p-8 text-center text-muted-foreground text-sm animate-pulse">
              Loading trained models...
            </div>
          ) : filteredTrained.length > 0 ? (
            <div className="mt-4 space-y-2">
              {filteredTrained.map((m) => {
                const metrics = pickMetrics(m.metrics)
                return (
                  <div
                    key={m.model_name}
                    className="rounded-xl border border-border bg-card p-4 transition-colors hover:bg-accent/30"
                  >
                    <div className="flex items-start justify-between gap-4">
                      <div className="min-w-0 flex-1">
                        <div className="flex items-center gap-2 flex-wrap">
                          <Box className="h-4 w-4 shrink-0 text-primary" />
                          <span className="font-medium text-sm text-foreground">{m.model_name}</span>
                          <Badge variant="secondary" className="text-[10px] px-1.5 py-0">
                            {m.model_type}
                          </Badge>
                          <span className="flex items-center gap-0.5 text-[10px] text-muted-foreground">
                            <Hash className="h-3 w-3" />v{m.version}
                          </span>
                        </div>
                        {m.target_column && (
                          <p className="mt-1 pl-6 text-xs text-muted-foreground flex items-center gap-1">
                            <Target className="h-3 w-3 shrink-0" />
                            <span>Predicting</span> <span className="font-mono text-foreground/90">{m.target_column}</span>
                          </p>
                        )}
                      </div>

                      <a
                        href={`/api/trained-models/${encodeURIComponent(m.model_name)}/download`}
                        className="shrink-0 inline-flex items-center gap-1 rounded-md border border-border px-2.5 py-1.5 text-xs font-medium text-muted-foreground hover:text-foreground hover:bg-accent transition-colors"
                      >
                        <Download className="h-3 w-3" />
                        Download
                      </a>
                    </div>

                    {/* Metrics */}
                    {metrics.length > 0 && (
                      <div className="mt-3 pl-6 flex items-center gap-2 flex-wrap">
                        <Activity className="h-3 w-3 text-muted-foreground shrink-0" />
                        {metrics.map((met) => (
                          <MetricPill key={met.label} label={met.label} value={met.value} />
                        ))}
                      </div>
                    )}

                    {/* Features + meta */}
                    <div className="mt-2 pl-6 flex items-center gap-6 text-xs text-muted-foreground">
                      <FeatureList features={m.feature_names} />
                      {m.training_samples > 0 && (
                        <span className="tabular-nums">{m.training_samples.toLocaleString()} samples</span>
                      )}
                      {m.created_at && (
                        <span className="flex items-center gap-1">
                          <Calendar className="h-3 w-3" />
                          {formatDate(m.created_at)}
                        </span>
                      )}
                    </div>
                  </div>
                )
              })}
            </div>
          ) : trainedModels.length > 0 ? (
            <div className="mt-6 rounded-xl bg-card p-8 text-center text-muted-foreground text-sm">
              No trained models match "{search}".
            </div>
          ) : (
            <div className="mt-6 rounded-xl bg-card border border-border p-10 text-center">
              <Box className="mx-auto h-8 w-8 text-muted-foreground/50" />
              <p className="mt-3 text-sm text-muted-foreground">
                No trained models yet. Complete a training run in the Experiment Lab.
              </p>
            </div>
          )}
        </section>
      </div>
    </div>
  )
}
