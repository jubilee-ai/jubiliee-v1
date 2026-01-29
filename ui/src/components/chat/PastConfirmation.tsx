import React from "react"
import ReactMarkdown from "react-markdown"
import { cn } from "@/lib/utils"
import { Bot, Check, ChevronRight } from "lucide-react"
import type { ConfirmationRequest, TrainingAgentState, StepInfo } from "@/types/agent"

export interface ResolvedConfirmation extends ConfirmationRequest {
  resolvedAction?: string
  afterMessageId?: string
  redoComment?: string
}

interface PastConfirmationProps {
  confirmation: ResolvedConfirmation
  agentState?: TrainingAgentState
  steps?: StepInfo[]
  onViewDetails: (stepId: string) => void
}

export function PastConfirmation({ 
  confirmation, 
  agentState, 
  steps, 
  onViewDetails 
}: PastConfirmationProps) {
  const isRedo = confirmation.resolvedAction === "redo"
  const redoComment = confirmation.redoComment

  return (
    <div className="flex gap-3">
      {/* Bot avatar */}
      <div className="flex-shrink-0 w-7 h-7 rounded-full flex items-center justify-center bg-muted">
        <Bot className="h-3.5 w-3.5 text-muted-foreground" />
      </div>
      
      <div className="flex-1 max-w-[92%] space-y-2">
        {/* Summary bubble - styling based on action */}
        <div className={cn(
          "rounded-2xl px-4 py-3 bg-muted/50 border",
          isRedo ? "border-amber-500/20" : "border-green-500/20"
        )}>
          <div className="flex items-center gap-2 mb-2">
            <span className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
              {confirmation.stepName}
            </span>
            {isRedo ? (
              <span className="flex items-center gap-1 text-xs text-amber-600 dark:text-amber-500">
                (Redo requested. Feedback: "{redoComment}")
              </span>
            ) : (
              <span className="flex items-center gap-1 text-xs text-green-600 dark:text-green-500">
                <Check className="h-3 w-3" />
                Accepted
              </span>
            )}
          </div>
          
          <div className="text-[15px] leading-relaxed prose prose-sm dark:prose-invert max-w-none prose-p:my-1 prose-headings:my-2">
            <ReactMarkdown>{confirmation.summary}</ReactMarkdown>
          </div>
          
          {/* Compact key details */}
          {confirmation.details && Object.keys(confirmation.details).length > 0 && (
            <ConfirmationDetails details={confirmation.details} />
          )}
          
          {/* View details link */}
          {agentState && steps && (
            <button
              onClick={() => onViewDetails(confirmation.step)}
              className="mt-2 flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
            >
              <span>View full details</span>
              <ChevronRight className="h-3 w-3" />
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

interface ConfirmationDetailsProps {
  details: Record<string, unknown>
}

export function ConfirmationDetails({ details }: ConfirmationDetailsProps) {
  return (
    <div className="mt-3 pt-2 border-t border-border/30 flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
      {Boolean(details.selected_model) && (
        <span>Model: <code className="text-foreground">{String(details.selected_model)}</code></span>
      )}
      {Boolean(details.collected_dataset_ref) && (
        <span>Dataset: <code className="text-foreground">{String(details.collected_dataset_ref)}</code></span>
      )}
      {Boolean(details.label_definition) && (
        <span>Target: <code className="text-foreground">{String((details.label_definition as Record<string, unknown>)?.target_column || "N/A")}</code></span>
      )}
      {Boolean(details.feature_spec) && (
        <span>Features: <code className="text-foreground">{((details.feature_spec as Record<string, unknown>)?.features as unknown[])?.length || 0}</code></span>
      )}
      {(details.training_metrics as Record<string, unknown>)?.test_accuracy != null && (
        <span>Accuracy: <code className="text-foreground">{(Number((details.training_metrics as Record<string, unknown>).test_accuracy) * 100).toFixed(1)}%</code></span>
      )}
      {(details.training_metrics as Record<string, unknown>)?.test_r2 != null && (
        <span>R²: <code className="text-foreground">{Number((details.training_metrics as Record<string, unknown>).test_r2).toFixed(4)}</code></span>
      )}
    </div>
  )
}
