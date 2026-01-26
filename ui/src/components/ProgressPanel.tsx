import { ScrollArea } from "@/components/ui/scroll-area"
import { StepNode } from "./StepNode"
import { Separator } from "@/components/ui/separator"
import type { StepInfo, TrainingAgentState } from "@/types/agent"
import { CheckCircle2 } from "lucide-react"

interface ProgressPanelProps {
  steps: StepInfo[]
  currentStepId: string | null
  agentState: TrainingAgentState
  isRunning: boolean
  onStepClick?: (stepId: string) => void
}

export function ProgressPanel({
  steps,
  currentStepId,
  agentState,
  isRunning: _isRunning,
  onStepClick,
}: ProgressPanelProps) {
  const completedSteps = steps.filter((s) => s.status === "completed").length
  const progress = Math.round((completedSteps / steps.length) * 100)

  return (
    <div className="h-full flex flex-col bg-muted/30">
      {/* Header */}
      <div className="p-5">
        <div className="flex items-baseline justify-between mb-1">
          <span className="text-sm font-medium">Progress</span>
          <span className="text-2xl font-semibold tracking-tight">{progress}%</span>
        </div>
        <div className="h-1.5 bg-border rounded-full overflow-hidden">
          <div 
            className="h-full bg-foreground rounded-full transition-all duration-500 ease-out"
            style={{ width: `${progress}%` }}
          />
        </div>

        {/* Context */}
        {agentState.goal && (
          <p className="mt-4 text-sm text-muted-foreground line-clamp-2 leading-relaxed">
            {agentState.goal}
          </p>
        )}
        {agentState.selected_model && (
          <p className="mt-2 text-sm font-medium">{agentState.selected_model}</p>
        )}
      </div>

      <Separator />

      {/* Steps List */}
      <ScrollArea className="flex-1">
        <div className="p-4 space-y-1">
          {steps.map((step, index) => (
            <StepNode
              key={step.id}
              step={step}
              stepNumber={index + 1}
              isActive={step.id === currentStepId || step.status === "awaiting_confirmation"}
              onSelect={() => onStepClick?.(step.id)}
            />
          ))}
        </div>
      </ScrollArea>

      {/* Final Metrics Summary */}
      {(agentState.training_metrics?.success || agentState.report_path) && (() => {
        const metrics = agentState.training_metrics
        const lastIteration = metrics?.iterations?.[metrics.iterations.length - 1]
        const testR2 = metrics?.test_r2 ?? lastIteration?.test_r2
        const testRmse = metrics?.test_rmse ?? lastIteration?.test_rmse
        
        const hasClassificationMetrics = metrics?.test_accuracy != null || metrics?.test_roc_auc != null
        const hasRegressionMetrics = testR2 != null || testRmse != null
        
        return (
          <>
            <Separator />
            <div className="p-5">
              <div className="flex items-center gap-2 mb-3">
                <CheckCircle2 className="h-4 w-4 text-green-600" />
                <span className="text-sm font-medium">Complete</span>
              </div>
              {hasClassificationMetrics ? (
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <div className="text-xs text-muted-foreground mb-1">Accuracy</div>
                    <div className="text-lg font-semibold">{((metrics?.test_accuracy || 0) * 100).toFixed(1)}%</div>
                  </div>
                  <div>
                    <div className="text-xs text-muted-foreground mb-1">ROC-AUC</div>
                    <div className="text-lg font-semibold">{(metrics?.test_roc_auc || 0).toFixed(3)}</div>
                  </div>
                </div>
              ) : hasRegressionMetrics ? (
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <div className="text-xs text-muted-foreground mb-1">R²</div>
                    <div className="text-lg font-semibold">{testR2?.toFixed(4)}</div>
                  </div>
                  <div>
                    <div className="text-xs text-muted-foreground mb-1">RMSE</div>
                    <div className="text-lg font-semibold">{testRmse?.toFixed(0)}</div>
                  </div>
                </div>
              ) : null}
            </div>
          </>
        )
      })()}
    </div>
  )
}
