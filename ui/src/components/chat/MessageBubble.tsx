import { useState, useRef, useEffect, forwardRef } from "react"
import type { Ref, MutableRefObject } from "react"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import { cn } from "@/lib/utils"
import { FileText, ChevronRight, Check, RotateCcw, Info } from "lucide-react"
import type { Dataset as ApiDataset } from "@/lib/api"
import type { ChatMessage, ChatTaskPlanPayload, TaskPlanSummary } from "@/types/agent"
import { looksLikeLeakedPlanJson, stripLeakedPlanJson } from "@/lib/planDisplay"
import { TaskPlanCard } from "@/components/TaskPlanCard"

/** Long follow-up explanations from the model; keep readable without walls of text. */
const MAX_AGENT_BODY = 380

function sanitizeAgentContent(content: string): string {
  const trimmed = stripLeakedPlanJson(content).trim()

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
  stepId?: string | null
  isClickable?: boolean
  onStepClick?: () => void
  isRunning?: boolean
  onApproveTrainingPlan?: (messageId: string, plan: TaskPlanSummary, refs: string[]) => void
  datasets?: ApiDataset[]
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
      stepId,
      isClickable,
      onStepClick,
      isRunning,
      onApproveTrainingPlan,
      datasets,
    },
    ref,
  ) {
    const [isExpanded, setIsExpanded] = useState(false)
    const rootRef = useRef<HTMLDivElement | null>(null) as MutableRefObject<HTMLDivElement | null>
    const isUser = message.role === "user"
    const isSystem = message.role === "system"
    
    const showReportCta =
      !isUser &&
      !isSystem &&
      message.showReportButton === true &&
      Boolean(onViewReport)

    const hasTaskPlan = Boolean(message.taskPlan)
    const sanitizedAgent = sanitizeAgentContent(message.content)
    const agentBodyForMarkdown = hasTaskPlan ? "" : sanitizedAgent.trim()
    const shouldTruncateAgent = !hasTaskPlan && agentBodyForMarkdown.length > MAX_AGENT_BODY
    const displayAgentBody =
      shouldTruncateAgent && !isExpanded
        ? `${agentBodyForMarkdown.slice(0, MAX_AGENT_BODY).trim()}…`
        : agentBodyForMarkdown

    useEffect(() => {
      if (!isExpanded || !shouldTruncateAgent) return
      const el = rootRef.current
      if (!el) return
      let inner = 0
      const outer = requestAnimationFrame(() => {
        inner = requestAnimationFrame(() => {
          el.scrollIntoView({ behavior: "smooth", block: "end", inline: "nearest" })
        })
      })
      return () => {
        cancelAnimationFrame(outer)
        cancelAnimationFrame(inner)
      }
    }, [isExpanded, shouldTruncateAgent])

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
          {!isStepAccepted && !isRedo && <Info className="h-3 w-3" />}
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
            message.apiPayload ? "w-full max-w-3xl mx-auto px-1" : bubbleBase,
            isUser && !message.apiPayload ? userBubble : !isUser ? agentBubble : "",
            isHighlighted && "ring-2 ring-foreground/20 shadow-lg",
            isClickable && "cursor-pointer hover:bg-muted/50 hover:shadow-md group"
          )}
        >
          {isUser ? (
            message.apiPayload ? (
              <div className="rounded-xl border border-border/45 bg-muted/20 px-4 py-3 text-left w-full">
                <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground mb-1.5">
                  Topic
                </p>
                <p className="text-sm text-foreground leading-snug">{message.content}</p>
              </div>
            ) : (
              <p className="text-[14px] leading-6 whitespace-pre-wrap">{message.content}</p>
            )
          ) : (
            <>
              {hasTaskPlan ? null : displayAgentBody ? (
                <div className="text-[14px] leading-relaxed prose prose-sm dark:prose-invert max-w-none prose-p:my-1 prose-p:leading-relaxed prose-headings:my-2 prose-headings:font-medium prose-ul:my-1 prose-ol:my-1 prose-li:my-0 prose-a:text-primary prose-a:no-underline hover:prose-a:underline prose-code:bg-primary/10 prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded-md prose-code:text-[13px] prose-code:font-normal prose-code:before:content-none prose-code:after:content-none prose-strong:font-semibold">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{displayAgentBody}</ReactMarkdown>
                </div>
              ) : null}
              {message.taskPlan && onApproveTrainingPlan && (
                <TaskPlanCard
                  payload={message.taskPlan as ChatTaskPlanPayload}
                  datasets={datasets}
                  resolved={message.taskPlanResolved}
                  isRunning={isRunning}
                  disabled={isRunning || Boolean(message.taskPlanResolved)}
                  onApproveStart={() =>
                    onApproveTrainingPlan(message.id, message.taskPlan!.plan, message.taskPlan!.datasetRefs)
                  }
                />
              )}
            </>
          )}

          {shouldTruncateAgent && (
            <button
              onClick={(e) => {
                e.stopPropagation()
                setIsExpanded(!isExpanded)
              }}
              className="mt-2 text-sm text-muted-foreground hover:text-foreground transition-colors font-medium"
            >
              {isExpanded ? "Show less" : "Show more"}
            </button>
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
              See details
              <ChevronRight className="h-3 w-3 group-hover:translate-x-0.5 transition-transform" />
            </button>
          )}
          
          {showReportCta && (
            <button
              onClick={(e) => {
                e.stopPropagation()
                onViewReport?.()
              }}
              className="mt-4 w-full sm:w-auto inline-flex items-center justify-center gap-1.5 px-4 py-2 rounded-lg bg-primary-subtle text-primary-subtle-foreground text-sm font-medium hover:bg-primary-subtle/88 transition-colors"
            >
              <FileText className="h-3.5 w-3.5" />
              View Report
            </button>
          )}
        </div>
      </div>
    )
  }
)
