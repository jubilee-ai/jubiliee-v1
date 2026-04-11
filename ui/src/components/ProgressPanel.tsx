import { ScrollArea } from "@/components/ui/scroll-area"
import { StepNode } from "./StepNode"
import { Separator } from "@/components/ui/separator"
import type { StepInfo, TrainingAgentState } from "@/types/agent"
import { CHECKLIST_EXCLUDE_IDS } from "@/lib/trainingSteps"
import { CheckCircle2 } from "lucide-react"

interface ProgressPanelProps {
  steps: StepInfo[]
  currentStepId: string | null
  agentState: TrainingAgentState
  isRunning: boolean
  onStepClick?: (stepId: string) => void
}

const STEP_GROUPS: Array<{ label: string; ids: string[]; mergeTraining?: boolean }> = [
  { label: "Setup", ids: ["data_collection"] },
  { label: "Preparation", ids: ["cleaning", "label_split_definition"] },
  { label: "Features", ids: ["feature_specification_and_engineering", "feature_selection_specification", "feature_engineering_executor"] },
  /** One checklist row: plan approval + model training share a single "Training" label */
  { label: "Training", ids: ["training_approval", "training"], mergeTraining: true },
]

function mergeTrainingSteps(
  approval: StepInfo | undefined,
  training: StepInfo | undefined,
): StepInfo | null {
  if (!approval && !training) return null
  const a = approval
  const t = training
  const status: StepInfo["status"] = (() => {
    if (!t && a) return a.status
    if (!a && t) return t.status
    if (!a || !t) return "pending"
    if (t.status === "completed") return "completed"
    if (t.status === "error" || a.status === "error") return "error"
    if (t.status === "stale" || a.status === "stale") return "stale"
    if (a.status === "awaiting_confirmation") return "awaiting_confirmation"
    if (t.status === "running" || a.status === "running") return "running"
    if (a.status === "completed" && (t.status === "pending" || t.status === "skipped")) return "pending"
    if (a.status === "pending") return "pending"
    return t.status
  })()
  const detailId =
    a && t && (a.status === "completed" || a.status === "skipped") ? t.id : a?.id ?? t?.id ?? "training"
  const base = (a && t && (a.status === "completed" || a.status === "skipped") ? t : a) ?? t ?? a!
  return {
    ...base,
    id: detailId,
    name: "Training",
    runCount: Math.max(a?.runCount ?? 0, t?.runCount ?? 0),
    status,
  }
}

export function ProgressPanel({
  steps,
  currentStepId,
  agentState,
  isRunning: _isRunning,
  onStepClick,
}: ProgressPanelProps) {
  const checklistSteps = steps.filter((s) => !CHECKLIST_EXCLUDE_IDS.has(s.id))
  const completedSteps = checklistSteps.filter((s) => s.status === "completed").length
  const totalSteps = checklistSteps.length
  const progress = totalSteps > 0 ? Math.round((completedSteps / totalSteps) * 100) : 0

    return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="p-5 pb-4">
        <div className="flex items-baseline justify-between mb-1.5">
          <span className="font-headline text-sm font-semibold">Experiment checklist</span>
          <span className="text-xs text-muted-foreground">
            {completedSteps}/{totalSteps} done
          </span>
        </div>
        <div className="h-1.5 bg-muted rounded-full overflow-hidden">
          <div 
            className="h-full bg-gradient-to-r from-primary to-accent rounded-full transition-all duration-500 ease-out"
            style={{ width: `${progress}%` }}
          />
        </div>

        {/* Goal */}
        {agentState.goal && (
          <p className="mt-3 text-ui text-muted-foreground line-clamp-2 leading-relaxed">
            {agentState.goal}
          </p>
        )}
      </div>

      <Separator />

      {/* Steps List - grouped */}
      <ScrollArea className="flex-1">
        <div className="p-3">
          {STEP_GROUPS.map((group, gi) => {
            if (group.mergeTraining) {
              const merged = mergeTrainingSteps(
                steps.find((s) => s.id === "training_approval"),
                steps.find((s) => s.id === "training"),
              )
              if (!merged) return null
              const active =
                merged.id === currentStepId ||
                currentStepId === "training_approval" ||
                currentStepId === "training" ||
                merged.status === "awaiting_confirmation"
              return (
                <div key={group.label} className={gi > 0 ? "mt-3" : ""}>
                  <div className="text-overline font-semibold uppercase tracking-widest text-muted-foreground/60 px-2.5 mb-1">
                    {group.label}
                  </div>
                  <div className="space-y-0.5">
                    <StepNode
                      key="training_merged"
                      step={merged}
                      isActive={active}
                      onSelect={() => onStepClick?.(merged.id)}
                    />
                  </div>
                </div>
              )
            }

            const groupSteps = group.ids
              .map((id) => steps.find((s) => s.id === id))
              .filter(Boolean) as StepInfo[]

            if (groupSteps.length === 0) return null

            return (
              <div key={group.label} className={gi > 0 ? "mt-3" : ""}>
                <div className="text-overline font-semibold uppercase tracking-widest text-muted-foreground/60 px-2.5 mb-1">
                  {group.label}
                </div>
                <div className="space-y-0.5">
                  {groupSteps.map((step) => (
                    <StepNode
                      key={step.id}
                      step={step}
                      isActive={step.id === currentStepId || step.status === "awaiting_confirmation"}
                      onSelect={() => onStepClick?.(step.id)}
                    />
                  ))}
                </div>
              </div>
            )
          })}
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
                <CheckCircle2 className="h-4 w-4 text-[hsl(var(--step-complete))]" />
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
