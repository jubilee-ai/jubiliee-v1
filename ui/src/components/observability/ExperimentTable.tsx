import { useMemo, useState } from "react"
import { Badge } from "@/components/ui/badge"
import { Search, CheckCircle2, Loader2, AlertCircle, Clock, Layers } from "lucide-react"
import type { ObservabilitySummaryRow } from "@/lib/api"
import { cn } from "@/lib/utils"
import { formatDistanceToNow } from "date-fns"

interface ExperimentTableProps {
  experiments: ObservabilitySummaryRow[]
  onSelectExperiment: (id: string) => void
  selectedExperimentId?: string | null
}

const statusConfig: Record<string, { icon: React.ReactNode; badge: string }> = {
  completed: {
    icon: <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500" />,
    badge: "bg-emerald-500/15 text-emerald-500 border-transparent",
  },
  running: {
    icon: <Loader2 className="h-3.5 w-3.5 animate-spin text-blue-500" />,
    badge: "bg-blue-500/15 text-blue-500 border-transparent",
  },
  failed: {
    icon: <AlertCircle className="h-3.5 w-3.5 text-red-500" />,
    badge: "bg-red-500/15 text-red-500 border-transparent",
  },
  created: {
    icon: <Clock className="h-3.5 w-3.5 text-muted-foreground" />,
    badge: "bg-muted text-muted-foreground border-transparent",
  },
}

function formatDuration(ms: number | null): string {
  if (ms == null) return "--"
  if (ms < 1000) return `${ms}ms`
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`
  return `${(ms / 60_000).toFixed(1)}m`
}

function pickMetric(metrics: Record<string, number>): { label: string; value: string } | null {
  const priority: [string, string][] = [
    ["test_roc_auc", "AUC"],
    ["test_accuracy", "Acc"],
    ["test_r2", "R\u00b2"],
    ["test_rmse", "RMSE"],
  ]
  for (const [key, label] of priority) {
    const v = metrics[key]
    if (v != null && typeof v === "number") {
      return { label, value: v < 1 && v > 0 ? `${(v * 100).toFixed(1)}%` : v.toFixed(4) }
    }
  }
  return null
}

export function ExperimentTable({ experiments, onSelectExperiment, selectedExperimentId }: ExperimentTableProps) {
  const [search, setSearch] = useState("")

  const filtered = useMemo(
    () =>
      experiments.filter(
        (e) =>
          !search ||
          e.name.toLowerCase().includes(search.toLowerCase()) ||
          e.goal?.toLowerCase().includes(search.toLowerCase()) ||
          e.status.toLowerCase().includes(search.toLowerCase())
      ),
    [experiments, search]
  )

  if (experiments.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center rounded-xl border border-border bg-card p-12 text-center">
        <Layers className="h-8 w-8 text-muted-foreground/40" />
        <p className="mt-3 text-sm text-muted-foreground">
          No experiments yet. Run your first experiment to see observability data.
        </p>
      </div>
    )
  }

  return (
    <div className="space-y-3">
      <div className="relative">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
        <input
          type="text"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Filter experiments..."
          className="w-full max-w-sm rounded-lg border border-border bg-card pl-9 pr-3 py-2 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring"
        />
      </div>

      <div className="rounded-xl border border-border bg-card overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border/50 bg-muted/30">
              <th className="text-left px-4 py-2.5 text-[11px] font-semibold text-muted-foreground uppercase tracking-wider">Experiment</th>
              <th className="text-left px-4 py-2.5 text-[11px] font-semibold text-muted-foreground uppercase tracking-wider">Status</th>
              <th className="text-right px-4 py-2.5 text-[11px] font-semibold text-muted-foreground uppercase tracking-wider">Steps</th>
              <th className="text-right px-4 py-2.5 text-[11px] font-semibold text-muted-foreground uppercase tracking-wider">Duration</th>
              <th className="text-right px-4 py-2.5 text-[11px] font-semibold text-muted-foreground uppercase tracking-wider">Metric</th>
              <th className="text-right px-4 py-2.5 text-[11px] font-semibold text-muted-foreground uppercase tracking-wider">When</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((exp) => {
              const sc = statusConfig[exp.job_status || exp.status] || statusConfig.created
              const metric = pickMetric(exp.metrics)
              const isSelected = selectedExperimentId === exp.experiment_id
              return (
                <tr
                  key={exp.experiment_id}
                  onClick={() => onSelectExperiment(exp.experiment_id)}
                  className={cn(
                    "border-b border-border/30 cursor-pointer transition-colors hover:bg-muted/40",
                    isSelected && "bg-primary/5"
                  )}
                >
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-2">
                      {sc.icon}
                      <div className="min-w-0">
                        <div className="font-medium text-foreground truncate max-w-[200px]">{exp.name}</div>
                        {exp.goal && (
                          <div className="text-xs text-muted-foreground truncate max-w-[200px] mt-0.5">{exp.goal}</div>
                        )}
                      </div>
                    </div>
                  </td>
                  <td className="px-4 py-3">
                    <Badge className={cn("text-[10px] px-1.5 py-0", sc.badge)}>
                      {exp.job_status || exp.status}
                    </Badge>
                  </td>
                  <td className="px-4 py-3 text-right tabular-nums text-muted-foreground">
                    {exp.steps_completed}
                  </td>
                  <td className="px-4 py-3 text-right tabular-nums text-muted-foreground">
                    {formatDuration(exp.total_duration_ms)}
                  </td>
                  <td className="px-4 py-3 text-right">
                    {metric ? (
                      <span className="inline-flex items-center gap-1 rounded-md bg-muted px-2 py-0.5 text-[11px] font-mono">
                        <span className="text-muted-foreground">{metric.label}</span>
                        <span className="font-medium text-foreground tabular-nums">{metric.value}</span>
                      </span>
                    ) : (
                      <span className="text-muted-foreground/40">--</span>
                    )}
                  </td>
                  <td className="px-4 py-3 text-right text-xs text-muted-foreground/60 tabular-nums">
                    {exp.updated_at
                      ? formatDistanceToNow(new Date(exp.updated_at), { addSuffix: true })
                      : "--"}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}
