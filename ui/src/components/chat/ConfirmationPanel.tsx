import { useState } from "react"
import ReactMarkdown from "react-markdown"
import { ShieldCheck, Check, FastForward, RotateCcw, ChevronRight } from "lucide-react"
import type { ConfirmationRequest, ConfirmationAction, TrainingAgentState, StepInfo } from "@/types/agent"
import { TRACE_EXCLUDE_IDS } from "@/lib/trainingSteps"
import { ConfirmationDetails } from "./PastConfirmation"

interface ConfirmationPanelProps {
  confirmationRequest: ConfirmationRequest
  onConfirmation: (action: ConfirmationAction, comment?: string) => void
  agentState?: TrainingAgentState
  steps?: StepInfo[]
  onViewDetails: (stepId: string) => void
}

export function ConfirmationPanel({
  confirmationRequest,
  onConfirmation,
  agentState,
  steps,
  onViewDetails,
}: ConfirmationPanelProps) {
  const [redoComment, setRedoComment] = useState("")

  const handleRedo = () => {
    if (redoComment.trim()) {
      onConfirmation("redo", redoComment)
      setRedoComment("")
    }
  }

  return (
    <div className="space-y-3">
      <div className="flex-1 space-y-3">
        {/* Summary bubble */}
        <div className="rounded-2xl px-4 py-3 bg-muted/50">
          <div className="flex items-center gap-2 mb-2">
            <ShieldCheck className="h-3.5 w-3.5 text-amber-500" />
            <span className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
              {confirmationRequest.stepName}
            </span>
            <span className="w-1.5 h-1.5 rounded-full bg-amber-500 animate-pulse" />
          </div>
          <p className="text-xs text-muted-foreground mb-2">
            {confirmationRequest.step === "planner"
              ? "Review the pipeline plan below. Approve to run it, or add feedback to adjust the plan."
              : "Review and approve to continue, or provide feedback to redo."}
          </p>
          
          <div className="text-[14px] leading-relaxed prose prose-sm dark:prose-invert max-w-none prose-p:my-1 prose-headings:my-2 max-h-48 overflow-y-auto">
            <ReactMarkdown>{confirmationRequest.summary}</ReactMarkdown>
          </div>
          
          {/* Compact key details */}
          {confirmationRequest.details && Object.keys(confirmationRequest.details).length > 0 && (
            <ConfirmationDetails details={confirmationRequest.details} />
          )}
          
          {/* View details link — planner HITL has no checklist row / step modal */}
          {agentState &&
            steps &&
            confirmationRequest.step !== "planner" &&
            !TRACE_EXCLUDE_IDS.has(confirmationRequest.step) && (
            <button
              onClick={() => onViewDetails(confirmationRequest.step)}
              className="mt-2 flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
            >
              <span>View full details</span>
              <ChevronRight className="h-3 w-3" />
            </button>
          )}
        </div>

        {/* Inline action bar */}
        <div className="flex items-center gap-2">
          {/* Accept button - primary action */}
          <button
            onClick={() => onConfirmation("accept")}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-foreground text-background text-sm font-medium hover:bg-foreground/90 transition-colors"
          >
            <Check className="h-3.5 w-3.5" />
            Accept
          </button>
          
          {/* Accept all - secondary */}
          <button
            onClick={() => onConfirmation("accept_all")}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full border border-border text-sm hover:bg-muted transition-colors"
          >
            <FastForward className="h-3.5 w-3.5" />
            Accept all
          </button>
          
          {/* Redo - inline input that expands */}
          <div className="flex-1 flex items-center gap-2">
            <div className="flex-1 relative">
              <input
                type="text"
                value={redoComment}
                onChange={(e) => setRedoComment(e.target.value)}
                placeholder="Redo with feedback..."
                className="w-full h-8 px-3 pr-9 rounded-full border border-border bg-background text-sm placeholder:text-muted-foreground/60 focus:outline-none focus:ring-1 focus:ring-foreground/20 transition-all"
                onKeyDown={(e) => {
                  if (e.key === "Enter" && redoComment.trim()) {
                    handleRedo()
                  }
                }}
              />
              {redoComment.trim() && (
                <button
                  onClick={handleRedo}
                  className="absolute right-1 top-1/2 -translate-y-1/2 w-6 h-6 rounded-full bg-foreground text-background flex items-center justify-center hover:bg-foreground/90 transition-colors"
                >
                  <RotateCcw className="h-3 w-3" />
                </button>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

