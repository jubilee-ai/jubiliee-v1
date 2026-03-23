import { Check, Loader2, Circle } from "lucide-react"
import type { StepInfo } from "@/types/agent"
import { cn } from "@/lib/utils"

interface TrainingProgressCardProps {
  steps: StepInfo[]
  progress: number
  isComplete: boolean
  metrics?: {
    accuracy?: number
    roc_auc?: number
    r2?: number
    rmse?: number
    model_name?: string
    duration?: string
    iterations?: number
  } | null
  onViewReport?: () => void
}

function StepStatusIcon({ status }: { status: string }) {
  switch (status) {
    case "completed":
      return (
        <div className="w-4 h-4 rounded-full bg-[hsl(var(--step-complete)/0.15)] flex items-center justify-center">
          <Check className="h-2.5 w-2.5 text-[hsl(var(--step-complete))]" />
        </div>
      )
    case "running":
      return <Loader2 className="h-4 w-4 text-[hsl(var(--step-active))] animate-spin" />
    default:
      return <Circle className="h-3 w-3 text-muted-foreground/40" />
  }
}

export function TrainingProgressCard({
  steps,
  progress,
  isComplete,
  metrics,
  onViewReport,
}: TrainingProgressCardProps) {
  const completedCount = steps.filter((s) => s.status === "completed").length
  const runningStep = steps.find((s) => s.status === "running")

  if (isComplete && metrics) {
    return (
      <div className="my-2 rounded-xl border border-[hsl(var(--step-complete)/0.2)] bg-[hsl(var(--step-complete)/0.05)] p-4 space-y-3">
        <div className="flex items-center justify-between">
          <span className="text-sm font-semibold text-[hsl(var(--step-complete))]">Training Complete</span>
          {onViewReport && (
            <button
              onClick={onViewReport}
              className="text-xs font-medium text-primary hover:underline"
            >
              View Report
            </button>
          )}
        </div>
        <div className="grid grid-cols-3 gap-3">
          {metrics.accuracy != null && (
            <div>
              <div className="text-[10px] uppercase tracking-wider text-muted-foreground">Accuracy</div>
              <div className="text-lg font-bold font-mono">{(metrics.accuracy * 100).toFixed(1)}%</div>
            </div>
          )}
          {metrics.roc_auc != null && (
            <div>
              <div className="text-[10px] uppercase tracking-wider text-muted-foreground">ROC-AUC</div>
              <div className="text-lg font-bold font-mono">{metrics.roc_auc.toFixed(3)}</div>
            </div>
          )}
          {metrics.r2 != null && (
            <div>
              <div className="text-[10px] uppercase tracking-wider text-muted-foreground">R²</div>
              <div className="text-lg font-bold font-mono">{metrics.r2.toFixed(4)}</div>
            </div>
          )}
          {metrics.rmse != null && (
            <div>
              <div className="text-[10px] uppercase tracking-wider text-muted-foreground">RMSE</div>
              <div className="text-lg font-bold font-mono">{metrics.rmse.toFixed(2)}</div>
            </div>
          )}
          {metrics.iterations != null && (
            <div>
              <div className="text-[10px] uppercase tracking-wider text-muted-foreground">Iterations</div>
              <div className="text-lg font-bold font-mono">{metrics.iterations}</div>
            </div>
          )}
          {metrics.model_name && (
            <div>
              <div className="text-[10px] uppercase tracking-wider text-muted-foreground">Model</div>
              <div className="text-sm font-medium truncate">{metrics.model_name}</div>
            </div>
          )}
        </div>
      </div>
    )
  }

  return (
    <div className="my-2 rounded-xl border border-border/25 bg-muted/30 p-3 space-y-2.5">
      {/* Header */}
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
          Experiment checklist
        </span>
        <span className="text-xs font-mono text-muted-foreground">
          {completedCount}/{steps.length} done
        </span>
      </div>

      {/* Progress Bar */}
      <div className="h-1.5 rounded-full bg-muted/60 overflow-hidden">
        <div
          className="h-full rounded-full bg-primary transition-all duration-500 ease-out"
          style={{ width: `${progress}%` }}
        />
      </div>

      {/* Step List */}
      <div className="space-y-1">
        {steps.map((step) => (
          <div
            key={step.id}
            className={cn(
              "flex items-center gap-2 py-0.5 text-xs transition-opacity duration-300",
              step.status === "pending" ? "opacity-40" : "opacity-100"
            )}
          >
            <StepStatusIcon status={step.status} />
            <span className={cn(
              "flex-1",
              step.status === "running" && "text-[hsl(var(--step-active))] font-medium",
              step.status === "completed" && "text-muted-foreground",
            )}>
              {step.name}
            </span>
            {step.subtitle && (
              <span className="text-[10px] text-muted-foreground/70 truncate max-w-[140px]">
                {step.subtitle}
              </span>
            )}
          </div>
        ))}
      </div>

      {/* Running indicator */}
      {runningStep && (
        <div className="flex items-center gap-1.5 text-xs text-[hsl(var(--step-active))]">
          <div className="w-1.5 h-1.5 rounded-full bg-[hsl(var(--step-active))] animate-pulse" />
          {runningStep.name}...
        </div>
      )}
    </div>
  )
}
