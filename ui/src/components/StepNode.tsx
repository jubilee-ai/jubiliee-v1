import { useState } from "react"
import { cn } from "@/lib/utils"
import type { StepInfo } from "@/types/agent"
import {
  Check,
  Loader2,
  AlertCircle,
  ChevronDown,
  RotateCcw,
  Brain,
  Database,
  Sparkles,
  GitBranch,
  BarChart3,
  Wrench,
  ClipboardCheck,
  Zap,
  FileText,
  Clock,
} from "lucide-react"

const STEP_ICONS: Record<string, React.ComponentType<{ className?: string }>> = {
  select_model: Brain,
  data_collection: Database,
  cleaning: Sparkles,
  label_split_definition: GitBranch,
  feature_selection_specification: BarChart3,
  feature_engineering_executor: Wrench,
  training_approval: ClipboardCheck,
  training: Zap,
  generate_report: FileText,
}

interface StepNodeProps {
  step: StepInfo
  isActive: boolean
  onSelect?: () => void
}

export function StepNode({ step, isActive: _isActive, onSelect }: StepNodeProps) {
  const [expanded, setExpanded] = useState(false)

  const isCompleted = step.status === "completed"
  const isRunning = step.status === "running"
  const isPending = step.status === "pending" || step.status === "skipped"
  const isError = step.status === "error"
  const isStale = step.status === "stale"
  const isAwaiting = step.status === "awaiting_confirmation"

  const StepIcon = STEP_ICONS[step.id] || FileText
  const hasDetails = !!(step.details || step.subtitle)
  const hasBeenRun = (step.runCount || 0) > 0

  const handleClick = () => {
    if (hasDetails || hasBeenRun) {
      setExpanded(!expanded)
    }
    onSelect?.()
  }

  return (
    <div
      className={cn(
        "rounded-lg transition-all",
        isRunning && "bg-card shadow-sm ring-1 ring-primary/20",
        isAwaiting && "bg-amber-50/60 ring-1 ring-amber-200/60",
        isStale && "bg-amber-50/30",
        isCompleted && expanded && "bg-muted/50",
      )}
    >
      <div
        className={cn(
          "flex items-center gap-2.5 py-2 px-2.5 cursor-pointer rounded-lg transition-colors",
          "hover:bg-muted/50",
          isPending && "opacity-40",
        )}
        onClick={handleClick}
      >
        {/* Step-specific icon */}
        <div
          className={cn(
            "w-7 h-7 rounded-md flex items-center justify-center flex-shrink-0 transition-all",
            isCompleted && "bg-[hsl(var(--step-complete))]/8 text-[hsl(var(--step-complete))]",
            isRunning && "bg-primary/10 text-primary",
            isPending && "bg-muted text-muted-foreground/40",
            isError && "bg-destructive/10 text-destructive",
            isStale && "bg-amber-100 text-amber-600",
            isAwaiting && "bg-amber-100 text-amber-600",
          )}
        >
          <StepIcon className="h-3.5 w-3.5" />
        </div>

        {/* Name and subtitle */}
        <div className="flex-1 min-w-0">
          <div className={cn(
            "text-[13px] leading-tight",
            (isCompleted || isRunning || isAwaiting) && "font-medium",
            isPending && "text-muted-foreground",
            isStale && "text-amber-700",
          )}>
            {step.name}
          </div>
          {isRunning && (
            <div className="text-[11px] text-muted-foreground mt-0.5">Processing...</div>
          )}
          {isAwaiting && (
            <div className="text-[11px] text-amber-600 mt-0.5">Awaiting review</div>
          )}
          {isStale && (
            <div className="text-[11px] text-amber-600 mt-0.5">Needs re-run</div>
          )}
          {isCompleted && step.subtitle && !expanded && (
            <div className="text-[11px] text-muted-foreground mt-0.5 truncate">{step.subtitle}</div>
          )}
        </div>

        {/* Right side: badges and status */}
        <div className="flex items-center gap-1.5 flex-shrink-0">
          {(step.runCount || 0) > 1 && (
            <span className="text-[10px] font-medium px-1.5 py-0.5 rounded-full bg-muted text-muted-foreground">
              {step.runCount}x
            </span>
          )}

          {/* Status indicator */}
          <div
            className={cn(
              "w-5 h-5 rounded-full flex items-center justify-center transition-all",
              isCompleted && "text-[hsl(var(--step-complete))]",
              isRunning && "text-foreground",
              isError && "text-destructive",
              isStale && "text-amber-500",
              isAwaiting && "text-amber-500",
              isPending && "text-muted-foreground/30",
            )}
          >
            {isCompleted ? (
              <Check className="h-3.5 w-3.5" strokeWidth={2.5} />
            ) : isRunning ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : isError ? (
              <AlertCircle className="h-3.5 w-3.5" />
            ) : isStale ? (
              <RotateCcw className="h-3 w-3" />
            ) : isAwaiting ? (
              <Clock className="h-3.5 w-3.5" />
            ) : (
              <div className="w-1.5 h-1.5 rounded-full bg-current" />
            )}
          </div>

          {/* Expand chevron */}
          {(hasDetails || hasBeenRun) && (
            <ChevronDown
              className={cn(
                "h-3.5 w-3.5 text-muted-foreground/50 transition-transform duration-200",
                expanded && "rotate-180",
              )}
            />
          )}
        </div>
      </div>

      {/* Expandable details */}
      {expanded && hasDetails && (
        <div className="px-2.5 pb-2.5 pt-0">
          <div className="ml-[38px] text-[12px] text-muted-foreground leading-relaxed border-l-2 border-border/60 pl-3 py-1">
            {step.subtitle && (
              <div className="font-medium text-foreground/80 mb-1">{step.subtitle}</div>
            )}
            {step.details && (
              <div className="whitespace-pre-wrap opacity-80">
                {step.details.length > 300 ? step.details.slice(0, 300) + "..." : step.details}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
