import { useMemo } from "react"
import { FileText, Loader2 } from "lucide-react"
import type { TrainingAgentState, TaskPlanSummary } from "@/types/agent"
import type { Dataset as ApiDataset } from "@/lib/api"
import { resolveDatasetDisplayNames } from "@/lib/datasetDisplay"
import { Button } from "@/components/ui/button"

export interface TaskStatusViewProps {
  agentState: TrainingAgentState
  onViewReport: () => void
  datasets?: ApiDataset[]
}

export function TaskStatusView({ agentState, onViewReport, datasets }: TaskStatusViewProps) {
  const status = agentState.task_status
  const plan = agentState.task_plan as TaskPlanSummary | null | undefined
  const err = agentState.task_error

  const datasetLine = useMemo(() => {
    if (!plan?.datasetLabels?.length) return ""
    const refs =
      plan.datasetRefs?.length ? plan.datasetRefs : (agentState.linked_datasets ?? [])
    return resolveDatasetDisplayNames(plan.datasetLabels, refs, datasets ?? []).join(", ")
  }, [plan, agentState.linked_datasets, datasets])

  const isRunning = status === "running" || status === "pending"
  const isFailed = status === "failed"
  const isDone = status === "completed"

  return (
    <div className="flex-1 min-h-0 overflow-y-auto">
      <div className="max-w-4xl mx-auto px-6 sm:px-10 py-10">
        <span className="text-overline font-bold text-muted-foreground/60 tracking-[0.2em] uppercase block mb-2">
          Background task
        </span>
        <h2 className="font-headline text-xl sm:text-2xl font-semibold tracking-tight mb-1 leading-snug">
          {plan?.goal || agentState.goal || "Training task"}
        </h2>
        {datasetLine ? (
          <p className="text-sm text-muted-foreground mb-6">
            Datasets: {datasetLine}
          </p>
        ) : (
          <div className="mb-6" />
        )}

        {isRunning && (
          <div className="space-y-2 mb-6">
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Loader2 className="h-4 w-4 animate-spin text-primary" />
              Jubilee is working through the pipeline…
            </div>
            <p className="text-xs text-muted-foreground/90 leading-relaxed pl-6">
              Safe to navigate away or close this tab — progress is saved on this experiment. Find it anytime in the
              sidebar.
            </p>
          </div>
        )}

        {isFailed && (
          <div className="rounded-xl border border-destructive/30 bg-destructive/5 px-4 py-3 text-sm text-destructive mb-6">
            {err != null && String(err).trim() ? String(err) : "Task failed."}
          </div>
        )}

        {isDone && agentState.training_metrics?.success && (
          <div className="rounded-xl border border-[hsl(var(--step-complete)/0.35)] bg-[hsl(var(--step-complete)/0.06)] px-4 py-3 text-sm mb-6 flex items-center justify-between gap-3">
            <span className="text-foreground/90">Training finished. Open the full audit report for metrics and trace.</span>
            <Button type="button" size="sm" className="shrink-0 gap-1.5" onClick={onViewReport}>
              <FileText className="h-3.5 w-3.5" />
              Report
            </Button>
          </div>
        )}
      </div>
    </div>
  )
}
