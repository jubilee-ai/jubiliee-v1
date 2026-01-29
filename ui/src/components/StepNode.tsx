import { cn } from "@/lib/utils"
import type { StepInfo } from "@/types/agent"
import {
  Check,
  Loader2,
  AlertCircle,
} from "lucide-react"

interface StepNodeProps {
  step: StepInfo
  stepNumber: number
  isActive: boolean
  onSelect?: () => void
}

export function StepNode({ step, stepNumber, isActive: _isActive, onSelect }: StepNodeProps) {
  const isCompleted = step.status === "completed"
  const isRunning = step.status === "running"
  const isPending = step.status === "pending" || step.status === "skipped"
  const isError = step.status === "error"

  return (
    <div
      className={cn(
        "flex items-center gap-3 py-2.5 px-3 rounded-xl transition-all cursor-pointer hover:bg-muted/50",
        isRunning && "bg-background shadow-sm",
        isPending && "opacity-50",
        isCompleted && "hover:bg-foreground/5"
      )}
      onClick={onSelect}
    >
      {/* Step indicator */}
      <div
        className={cn(
          "w-6 h-6 rounded-full flex items-center justify-center text-xs font-medium transition-all",
          isCompleted && "bg-foreground text-background",
          isRunning && "bg-foreground text-background",
          isPending && "bg-muted text-muted-foreground",
          isError && "bg-destructive text-destructive-foreground"
        )}
      >
        {isCompleted ? (
          <Check className="h-3.5 w-3.5" />
        ) : isRunning ? (
          <Loader2 className="h-3.5 w-3.5 animate-spin" />
        ) : isError ? (
          <AlertCircle className="h-3.5 w-3.5" />
        ) : (
          stepNumber
        )}
      </div>

      {/* Step name */}
      <div className="flex-1 min-w-0">
        <div className={cn(
          "text-sm",
          (isCompleted || isRunning) && "font-medium",
          isPending && "text-muted-foreground"
        )}>
          {step.name}
        </div>
        {isRunning && (
          <div className="text-xs text-muted-foreground mt-0.5">Processing...</div>
        )}
      </div>
    </div>
  )
}
