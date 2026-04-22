import { useRef, useMemo, forwardRef } from "react"
import type { Ref, MutableRefObject } from "react"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import { cn } from "@/lib/utils"
import { Box, FileText, ChevronRight, Check, RotateCcw, Database } from "lucide-react"
import type { Dataset as ApiDataset } from "@/lib/api"
import type { ChatMessage, ChatTaskPlanPayload, TaskPlanSummary, TrainingAgentState } from "@/types/agent"
import { FeatureAnalysisChatCard } from "@/components/chat/FeatureAnalysisChatCard"
import { AnalysisCard } from "@/components/chat/AnalysisCard"
import { looksLikeLeakedPlanJson, stripLeakedPlanJson } from "@/lib/planDisplay"
import { TaskPlanCard } from "@/components/TaskPlanCard"

/** Pipeline steps whose chat line should include the EDA digest when `key_stats` is present. */
const FEATURE_ANALYSIS_STEP_IDS = new Set([
  "feature_engineering_executor",
  "feature_specification_and_engineering",
  "feature_selection_specification",
])

const ANALYSIS_JSON_RE = /<ANALYSIS_JSON>[\s\S]*?<\/ANALYSIS_JSON>/g

function sanitizeAgentContent(content: string): string {
  const trimmed = stripLeakedPlanJson(content).replace(ANALYSIS_JSON_RE, "").trim()

  if (trimmed.startsWith("{") && trimmed.length > 10) {
    try {
      const obj = JSON.parse(trimmed) as Record<string, unknown>
      if (Array.isArray(obj.dataset_refs) && typeof obj.goal === "string") {
        return ""
      }
      if (obj.headline) return String(obj.headline)
      if (obj.summary && typeof obj.summary === "string") return obj.summary
      if (obj.message && typeof obj.message === "string") return obj.message
      if (obj.node) return `**${obj.node}** completed`
      return "Step completed"
    } catch {
      if (looksLikeLeakedPlanJson(trimmed)) return ""
    }
  }

  if (trimmed.startsWith("[") && trimmed.length > 20) {
    try {
      const obj = JSON.parse(trimmed)
      if (obj.headline) return obj.headline
      if (obj.summary && typeof obj.summary === "string") return obj.summary
      if (obj.message && typeof obj.message === "string") return obj.message
      if (obj.node) return `**${obj.node}** completed`
      return "Step completed"
    } catch {
      // ignore
    }
  }

  if (/^\s*\{?\s*'(plan|steps|status|goal|current_step|resolved_)/.test(trimmed)) {
    return ""
  }

  const lower = trimmed.toLowerCase()
  if (
    trimmed.length > 0 &&
    trimmed.length < 900 &&
    /i['']?ve set up|you can now run|you can start/i.test(lower) &&
    /training plan|dataset|background|step/i.test(lower)
  ) {
    return ""
  }

  return trimmed
}

interface MessageBubbleProps {
  message: ChatMessage
  isHighlighted?: boolean
  onViewReport?: () => void
  /** Shown beside View Report when a trained model can be opened on the Models page. */
  onViewModelInRegistry?: () => void
  stepId?: string | null
  isClickable?: boolean
  onStepClick?: () => void
  isRunning?: boolean
  onApproveTrainingPlanGuided?: (messageId: string, plan: TaskPlanSummary, refs: string[]) => void
  onApproveTrainingPlanBackground?: (messageId: string, plan: TaskPlanSummary, refs: string[]) => void
  datasets?: ApiDataset[]
  /** Latest experiment state — used to render the training metrics card on the training step message. */
  agentState?: TrainingAgentState | null
}

function assignRef<T>(r: Ref<T> | undefined, value: T | null) {
  if (typeof r === "function") r(value)
  else if (r && typeof r === "object" && "current" in r) {
    ;(r as MutableRefObject<T | null>).current = value
  }
}

export const MessageBubble = forwardRef<HTMLDivElement, MessageBubbleProps>(
  function MessageBubble(
    {
      message,
      isHighlighted,
      onViewReport,
      onViewModelInRegistry,
      stepId,
      isClickable,
      onStepClick,
      isRunning,
      onApproveTrainingPlanGuided,
      onApproveTrainingPlanBackground,
      datasets,
      agentState,
    },
    ref,
  ) {
    const rootRef = useRef<HTMLDivElement | null>(null) as MutableRefObject<HTMLDivElement | null>
    const isUser = message.role === "user"
    const isSystem = message.role === "system"
    const effectiveStepId = message.stepId ?? stepId ?? null
    const stepDetailsCtaLabel =
      effectiveStepId === "data_collection" ? "See preview" : "See details"
    const showFeatureAnalysisCard =
      !isUser &&
      !isSystem &&
      Boolean(agentState) &&
      effectiveStepId != null &&
      FEATURE_ANALYSIS_STEP_IDS.has(effectiveStepId)

    const showReportCta =
      !isUser &&
      !isSystem &&
      message.showReportButton === true &&
      Boolean(onViewReport)

    const hasTaskPlan = Boolean(message.taskPlan)
    const sanitizedAgent = sanitizeAgentContent(message.content)
    const agentBodyForMarkdown = hasTaskPlan ? "" : sanitizedAgent.trim()

    const analyses = message.analyses ?? []
    const proseAnalysisClass =
      analyses.length > 0
        ? "text-sm leading-snug prose prose-sm dark:prose-invert max-w-none prose-p:my-0.5 prose-headings:my-1.5 prose-headings:font-medium prose-ul:my-1 prose-li:my-0 prose-code:text-[11px] prose-code:leading-snug prose-strong:font-semibold"
        : "text-sm leading-relaxed prose prose-sm dark:prose-invert max-w-none prose-p:my-1 prose-p:leading-relaxed prose-headings:my-2 prose-headings:font-medium prose-ul:my-1 prose-ol:my-1 prose-li:my-0 prose-a:text-primary prose-a:no-underline hover:prose-a:underline prose-code:bg-primary/10 prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded-md prose-code:text-ui prose-code:font-normal prose-code:before:content-none prose-code:after:content-none prose-strong:font-semibold"

    const userLinkedKeys = message.linkedDatasetKeys?.filter(Boolean) ?? []
    const datasetChipLabel = useMemo(() => {
      const lookup = new Map<string, string>()
      for (const d of datasets ?? []) {
        if (d.file) lookup.set(d.file, d.name)
        if (d.id) lookup.set(d.id, d.name)
        if (d.name) lookup.set(d.name, d.name)
      }
      return (key: string) => lookup.get(key) ?? key.split("/").pop() ?? key
    }, [datasets])

    if (isSystem) {
      if (message.id === "__graph_thinking__") {
        return null
      }
      const isStepAccepted = message.content.includes("Step accepted")
      const isRedo = message.content.includes("Requested redo")
      return (
        <div
          ref={(node) => {
            rootRef.current = node
            assignRef(ref, node)
          }}
          className={cn(
          "flex items-center justify-center gap-1.5 text-xs py-2 select-none",
          isStepAccepted && "text-success",
          isRedo && "text-amber-600 dark:text-amber-400",
          !isStepAccepted && !isRedo && "text-muted-foreground/60"
        )}>
          {isStepAccepted && <Check className="h-3 w-3" />}
          {isRedo && <RotateCcw className="h-3 w-3" />}
          {message.content}
        </div>
      )
    }

    const handleBubbleClick = () => {
      if (isClickable && onStepClick) {
        onStepClick()
      }
    }

    const bubbleBase = "rounded-xl px-4 py-2.5 transition-all duration-300 w-full max-w-[85%]"
    const userBubble = "ml-auto bg-primary/[0.08] text-foreground rounded-br-sm"
    const agentBubble = "mr-auto text-foreground"

    return (
      <div 
        ref={(node) => {
          rootRef.current = node
          assignRef(ref, node)
        }}
        data-step-id={stepId}
        className={cn(
          "w-full flex transition-all duration-300 animate-message-in",
          isUser && !message.apiPayload && "justify-end",
          isHighlighted && "scale-[1.01]",
        )}
      >
        <div
          onClick={handleBubbleClick}
          className={cn(
            message.apiPayload ? "w-full max-w-4xl mx-auto px-1" : bubbleBase,
            isUser && !message.apiPayload ? userBubble : !isUser ? agentBubble : "",
            isHighlighted && "ring-2 ring-foreground/20 shadow-lg",
            isClickable && "cursor-pointer hover:bg-muted/50 hover:shadow-md group"
          )}
        >
          {isUser ? (
            message.apiPayload ? (
              <div className="rounded-xl border border-border/45 bg-muted/20 px-4 py-3 text-left w-full">
                <p className="text-overline font-semibold uppercase tracking-wider text-muted-foreground mb-1.5">
                  Topic
                </p>
                <p className="text-sm text-foreground leading-snug">{message.content}</p>
                {userLinkedKeys.length > 0 && (
                  <div className="flex flex-wrap gap-1.5 mt-2 pt-2 border-t border-border/25 justify-start">
                    {userLinkedKeys.map((key) => (
                      <span
                        key={key}
                        className="inline-flex items-center gap-1 rounded-full bg-background/60 border border-border/40 px-2.5 py-0.5 text-caption text-muted-foreground"
                      >
                        <Database className="h-3 w-3 shrink-0 opacity-70" aria-hidden />
                        <span className="truncate max-w-[220px]">{datasetChipLabel(key)}</span>
                      </span>
                    ))}
                  </div>
                )}
              </div>
            ) : (
              <>
                <p className="text-sm leading-6 whitespace-pre-wrap">{message.content}</p>
                {userLinkedKeys.length > 0 && (
                  <div className="flex flex-wrap gap-1.5 mt-2 justify-start w-full">
                    {userLinkedKeys.map((key) => (
                      <span
                        key={key}
                        className="inline-flex items-center gap-1 rounded-full bg-background/50 border border-border/35 px-2.5 py-0.5 text-caption text-muted-foreground"
                      >
                        <Database className="h-3 w-3 shrink-0 opacity-70" aria-hidden />
                        <span className="truncate max-w-[220px]">{datasetChipLabel(key)}</span>
                      </span>
                    ))}
                  </div>
                )}
              </>
            )
          ) : (
            <>
              {hasTaskPlan ? null : agentBodyForMarkdown ? (
                <div className={cn(proseAnalysisClass)}>
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{agentBodyForMarkdown}</ReactMarkdown>
                </div>
              ) : null}
              {analyses.length > 0 ? (
                <div className="mt-3 flex flex-col gap-3">
                  {analyses.map((insight, idx) => (
                    <AnalysisCard key={`${message.id}-a-${idx}-${insight.tool}-${insight.kind}`} insight={insight} />
                  ))}
                </div>
              ) : null}
              {showFeatureAnalysisCard && agentState ? <FeatureAnalysisChatCard agentState={agentState} /> : null}
              {message.taskPlan && onApproveTrainingPlanGuided && (
                <TaskPlanCard
                  payload={message.taskPlan as ChatTaskPlanPayload}
                  datasets={datasets}
                  resolved={message.taskPlanResolved}
                  isRunning={isRunning}
                  disabled={isRunning || Boolean(message.taskPlanResolved)}
                  onApproveGuided={() =>
                    onApproveTrainingPlanGuided(
                      message.id,
                      message.taskPlan!.plan,
                      message.taskPlan!.datasetRefs,
                    )
                  }
                  onApproveBackground={
                    onApproveTrainingPlanBackground
                      ? () =>
                          onApproveTrainingPlanBackground(
                            message.id,
                            message.taskPlan!.plan,
                            message.taskPlan!.datasetRefs,
                          )
                      : undefined
                  }
                />
              )}
            </>
          )}

          {isClickable && onStepClick && (
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation()
                onStepClick()
              }}
              className="mt-3 inline-flex items-center gap-1.5 text-xs font-medium text-muted-foreground hover:text-foreground transition-colors"
            >
              {stepDetailsCtaLabel}
              <ChevronRight className="h-3 w-3 group-hover:translate-x-0.5 transition-transform" />
            </button>
          )}
          
          {showReportCta && (
            <div className="mt-4 flex flex-wrap items-center gap-2">
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation()
                  onViewReport?.()
                }}
                className="w-full sm:w-auto inline-flex items-center justify-center gap-1.5 px-4 py-2 rounded-lg bg-primary-subtle text-primary-subtle-foreground text-sm font-medium hover:bg-primary-subtle/88 transition-colors"
              >
                <FileText className="h-3.5 w-3.5" />
                View Report
              </button>
              {onViewModelInRegistry ? (
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation()
                    onViewModelInRegistry()
                  }}
                  className="w-full sm:w-auto inline-flex items-center justify-center gap-1.5 px-4 py-2 rounded-lg border border-border bg-card text-foreground text-sm font-medium hover:bg-accent transition-colors"
                >
                  <Box className="h-3.5 w-3.5" />
                  View model
                </button>
              ) : null}
            </div>
          )}
        </div>
      </div>
    )
  }
)
