import { useState, useEffect, useMemo } from "react"
import {
  Box, Download, Activity, Layers, Target,
  Search, ChevronDown, ChevronRight, FileText, FlaskConical,
} from "lucide-react"
import { Badge } from "@/components/ui/badge"
import type {
  TrainedModelEntry,
  ModelRiskInventoryItem,
  ModelRiskTier,
} from "@/lib/api"
import { USE_MODEL_RISK_MOCK, mergeMockInventoryWithTrainedModelNames } from "@/lib/modelRiskMock"
import { useModelRiskList } from "@/lib/queries"
import { cn } from "@/lib/utils"

function tierBadgeClass(tier: ModelRiskTier): string {
  switch (tier) {
    case "low":
      return "border-emerald-500/30 bg-emerald-500/10 text-emerald-800 dark:text-emerald-300"
    case "medium":
      return "border-amber-500/35 bg-amber-500/10 text-amber-900 dark:text-amber-200"
    case "high":
      return "border-orange-500/40 bg-orange-500/10 text-orange-900 dark:text-orange-200"
    case "critical":
      return "border-destructive/45 bg-destructive/10 text-destructive"
    default:
      return ""
  }
}

function MetricPill({ label, value }: { label: string; value: number }) {
  const display = value < 1 && value > 0 ? (value * 100).toFixed(1) + "%" : value.toFixed(4)
  return (
    <span className="inline-flex items-baseline gap-1 rounded-md border border-border/60 bg-muted/30 px-2 py-0.5 text-[11px] leading-tight">
      <span className="text-muted-foreground font-medium">{label}</span>
      <span className="font-mono text-xs font-semibold text-foreground tabular-nums">{display}</span>
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
        type="button"
        onClick={(e) => {
          e.stopPropagation()
          setOpen(!open)
        }}
        className="inline-flex items-center gap-1 text-[11px] text-muted-foreground hover:text-foreground transition-colors"
      >
        {open ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
        <Layers className="h-3 w-3" />
        <span className="font-medium">{features.length} features</span>
      </button>
      {open && (
        <div className="mt-1.5 flex flex-wrap gap-1">
          {features.map((f) => (
            <span key={f} className="rounded bg-muted px-1.5 py-0.5 text-[10px] font-mono leading-tight text-muted-foreground">
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

interface ModelsPageProps {
  trainedModels: TrainedModelEntry[]
  loading: boolean
  onOpenExperiment?: (experimentId: string) => void
  /** Opens the shared app training report dialog (same as Experiment chat). */
  onViewReport?: (modelName: string) => void
  /** When set (e.g. from Experiment lab), scroll to this model row and briefly highlight it. */
  scrollToModelName?: string | null
  onScrollToModelConsumed?: () => void
  /** When false, skip model-risk inventory fetch (e.g. tab not visible). Default true. */
  riskInventoryEnabled?: boolean
  /** Opens model risk profile for the given registry name. */
  onOpenModelRisk?: (modelName: string) => void
}

export function ModelsPage({
  trainedModels,
  loading,
  onOpenExperiment,
  onViewReport,
  scrollToModelName,
  onScrollToModelConsumed,
  riskInventoryEnabled = true,
  onOpenModelRisk,
}: ModelsPageProps) {
  const [search, setSearch] = useState("")
  const riskList = useModelRiskList({ enabled: riskInventoryEnabled })
  const riskByName = useMemo(() => {
    const raw = riskList.data ?? []
    const merged =
      USE_MODEL_RISK_MOCK && riskInventoryEnabled
        ? mergeMockInventoryWithTrainedModelNames(trainedModels.map((t) => t.model_name))
        : raw
    const m = new Map<string, ModelRiskInventoryItem>()
    for (const it of merged) {
      m.set(it.model_name, it)
    }
    return m
  }, [riskList.data, riskInventoryEnabled, trainedModels])

  useEffect(() => {
    if (!scrollToModelName?.trim()) return
    const name = scrollToModelName.trim()
    let cancelled = false
    let pollTimer: ReturnType<typeof setTimeout> | undefined
    let highlightTimer: ReturnType<typeof setTimeout> | undefined
    const t0 = Date.now()
    /** Poll until the row exists (handles loading / async refresh without clearing scroll intent). */
    const POLL_MS = 120
    const MAX_WAIT_MS = 12_000

    const HIGHLIGHT_CLASS = ["ring-2", "ring-primary/45", "shadow-[0_0_0_1px_hsl(var(--primary)/0.2)]"] as const

    const tryScroll = (): boolean => {
      const escaped =
        typeof CSS !== "undefined" && typeof CSS.escape === "function"
          ? CSS.escape(name)
          : name.replace(/\\/g, "\\\\").replace(/"/g, '\\"')
      const el = document.querySelector(`[data-trained-model="${escaped}"]`)
      if (!(el instanceof HTMLElement) || cancelled) return false
      el.scrollIntoView({ block: "center", behavior: "smooth" })
      for (const c of HIGHLIGHT_CLASS) el.classList.add(c)
      highlightTimer = window.setTimeout(() => {
        if (cancelled) return
        for (const c of HIGHLIGHT_CLASS) el.classList.remove(c)
        onScrollToModelConsumed?.()
      }, 2400)
      return true
    }

    const poll = () => {
      if (cancelled) return
      if (tryScroll()) return
      if (Date.now() - t0 > MAX_WAIT_MS) {
        onScrollToModelConsumed?.()
        return
      }
      pollTimer = window.setTimeout(poll, POLL_MS)
    }

    poll()

    return () => {
      cancelled = true
      if (pollTimer !== undefined) window.clearTimeout(pollTimer)
      if (highlightTimer !== undefined) window.clearTimeout(highlightTimer)
    }
  }, [scrollToModelName, trainedModels, onScrollToModelConsumed])

  const filteredTrained = trainedModels.filter((m) => {
    if (scrollToModelName && m.model_name === scrollToModelName) return true
    return (
      !search ||
      m.model_name.toLowerCase().includes(search.toLowerCase()) ||
      m.model_type.toLowerCase().includes(search.toLowerCase())
    )
  })

  return (
    <div className="flex-1 overflow-auto">
      <div className="max-w-5xl mx-auto px-8 py-10">
        <div className="flex items-start gap-3">
          <div>
            <h1 className="font-headline text-2xl sm:text-3xl font-semibold text-foreground tracking-tight mt-1">
              Model Registry
            </h1>
          </div>
        </div>

        {/* Trained Models */}
        <section className="mt-8">
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

          {riskList.isError && riskInventoryEnabled && (
            <p className="mt-3 text-xs text-amber-700 dark:text-amber-300">
              Model risk metadata unavailable ({riskList.error.message}). Registry rows still load.
            </p>
          )}

          {loading && trainedModels.length === 0 ? (
            <div className="mt-6 rounded-xl bg-card p-8 text-center text-muted-foreground text-sm animate-pulse">
              Loading trained models...
            </div>
          ) : filteredTrained.length > 0 ? (
            <div className="mt-4 space-y-2">
              {filteredTrained.map((m) => {
                const metrics = pickMetrics(m.metrics)
                const risk = riskByName.get(m.model_name)
                const metaParts: string[] = []
                if (m.training_samples > 0) metaParts.push(`${m.training_samples.toLocaleString()} samples`)
                if (m.created_at) metaParts.push(`trained ${formatDate(m.created_at)}`)
                return (
                  <div
                    key={m.model_name}
                    data-trained-model={m.model_name}
                    role={onOpenModelRisk ? "button" : undefined}
                    tabIndex={onOpenModelRisk ? 0 : undefined}
                    onClick={(e) => {
                      if (!onOpenModelRisk) return
                      if ((e.target as HTMLElement).closest("a, button")) return
                      onOpenModelRisk(m.model_name)
                    }}
                    onKeyDown={(e) => {
                      if (!onOpenModelRisk) return
                      if (e.key === "Enter" || e.key === " ") {
                        e.preventDefault()
                        onOpenModelRisk(m.model_name)
                      }
                    }}
                    className={cn(
                      "group rounded-xl border border-border/80 bg-card p-3 shadow-sm transition-all duration-200",
                      "hover:border-primary/20 hover:shadow-md hover:bg-card",
                      onOpenModelRisk && "cursor-pointer",
                    )}
                  >
                    <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between sm:gap-3">
                      <div className="flex min-w-0 flex-1 gap-2.5">
                        <div className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-primary/8 text-primary">
                          <Box className="h-3.5 w-3.5" />
                        </div>
                        <div className="min-w-0 flex-1 space-y-1">
                          <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5">
                            <h3 className="font-semibold text-sm text-foreground tracking-tight">
                              {m.model_name}
                            </h3>
                            <span className="text-[11px] text-muted-foreground tabular-nums">v{m.version}</span>
                            {risk && (
                              <Badge
                                variant="outline"
                                className={cn(
                                  "h-5 px-1.5 text-[10px] font-medium capitalize",
                                  tierBadgeClass(risk.risk_tier),
                                )}
                              >
                                {risk.risk_tier} risk
                              </Badge>
                            )}
                            {riskList.isPending && riskInventoryEnabled && !risk && (
                              <span className="text-[11px] text-muted-foreground/70">Risk…</span>
                            )}
                          </div>
                          <p className="text-[11px] leading-snug text-muted-foreground font-mono truncate" title={m.model_type}>
                            {m.model_type}
                          </p>
                          {(metaParts.length > 0 || m.feature_names.length > 0) && (
                            <div className="flex flex-wrap items-center gap-x-1.5 gap-y-0.5 text-[11px] text-muted-foreground">
                              {metaParts.length > 0 && (
                                <span className="tabular-nums">{metaParts.join(" · ")}</span>
                              )}
                              {metaParts.length > 0 && m.feature_names.length > 0 && (
                                <span className="text-muted-foreground/40" aria-hidden>
                                  ·
                                </span>
                              )}
                              {m.feature_names.length > 0 && <FeatureList features={m.feature_names} />}
                            </div>
                          )}
                          {m.target_column && (
                            <p className="text-[11px] leading-snug text-foreground/85">
                              <Target className="mr-0.5 inline h-3 w-3 -translate-y-px text-muted-foreground" />
                              <span className="text-muted-foreground">Target </span>
                              <span className="font-mono font-medium">{m.target_column}</span>
                            </p>
                          )}
                          {metrics.length > 0 && (
                            <div className="flex flex-wrap items-center gap-1.5 pt-0.5">
                              <Activity className="h-3 w-3 shrink-0 text-muted-foreground" />
                              {metrics.map((met) => (
                                <MetricPill key={met.label} label={met.label} value={met.value} />
                              ))}
                            </div>
                          )}
                        </div>
                      </div>

                      <div className="flex shrink-0 flex-wrap items-center gap-1 sm:justify-end sm:pt-0.5">
                        {m.experiment_id && onOpenExperiment && (
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation()
                              onOpenExperiment(m.experiment_id!)
                            }}
                            className="inline-flex items-center justify-center gap-1 rounded-md border border-border bg-background px-2 py-1 text-[11px] font-medium text-foreground shadow-sm transition-colors hover:bg-accent"
                          >
                            <FlaskConical className="h-3 w-3 opacity-70" />
                            Open experiment
                          </button>
                        )}
                        {m.report_available && onViewReport && (
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation()
                              onViewReport(m.model_name)
                            }}
                            className="inline-flex items-center justify-center gap-1 rounded-md border border-border bg-background px-2 py-1 text-[11px] font-medium text-foreground shadow-sm transition-colors hover:bg-accent"
                          >
                            <FileText className="h-3 w-3 opacity-70" />
                            View report
                          </button>
                        )}
                        <a
                          href={`/api/trained-models/${encodeURIComponent(m.model_name)}/download`}
                          onClick={(e) => e.stopPropagation()}
                          className="inline-flex items-center justify-center gap-1 rounded-md border border-border bg-background px-2 py-1 text-[11px] font-medium text-foreground shadow-sm transition-colors hover:bg-accent"
                        >
                          <Download className="h-3 w-3 opacity-70" />
                          Download
                        </a>
                      </div>
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
