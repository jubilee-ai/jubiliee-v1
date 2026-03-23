import { useState, useRef, useEffect, forwardRef } from "react"
import type { Ref, MutableRefObject } from "react"
import ReactMarkdown from "react-markdown"
import { cn } from "@/lib/utils"
import { FileText, ChevronRight, Check, RotateCcw, Info } from "lucide-react"
import type { ChatMessage } from "@/types/agent"

const MAX_CONTENT_LENGTH = 400

interface MessageBubbleProps {
  message: ChatMessage
  isHighlighted?: boolean
  onViewReport?: () => void
  stepId?: string | null
  isClickable?: boolean
  onStepClick?: () => void
}

function assignRef<T>(r: Ref<T> | undefined, value: T | null) {
  if (typeof r === "function") r(value)
  else if (r && typeof r === "object" && "current" in r) {
    ;(r as MutableRefObject<T | null>).current = value
  }
}

export const MessageBubble = forwardRef<HTMLDivElement, MessageBubbleProps>(
  function MessageBubble({ message, isHighlighted, onViewReport, stepId, isClickable, onStepClick }, ref) {
    const [isExpanded, setIsExpanded] = useState(false)
    const rootRef = useRef<HTMLDivElement | null>(null) as MutableRefObject<HTMLDivElement | null>
    const isUser = message.role === "user"
    const isSystem = message.role === "system"
    
    const hasViewReport = !isUser && !isSystem && 
      (message.content.toLowerCase().includes("view report") || 
       message.content.toLowerCase().includes("training completed"))

    const shouldTruncate = message.content.length > MAX_CONTENT_LENGTH
    const displayContent = shouldTruncate && !isExpanded
      ? message.content.slice(0, MAX_CONTENT_LENGTH) + "..."
      : message.content

    useEffect(() => {
      if (!isExpanded || !shouldTruncate) return
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
    }, [isExpanded, shouldTruncate])

    if (isSystem) {
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
          isStepAccepted && "text-green-600 dark:text-green-400",
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

    const bubbleBase = "rounded-xl px-4 py-2.5 transition-all duration-300"
    const userBubble = "ml-auto bg-primary/[0.08] text-foreground rounded-br-sm max-w-[85%]"
    const agentBubble = "text-foreground"

    return (
      <div 
        ref={(node) => {
          rootRef.current = node
          assignRef(ref, node)
        }}
        data-step-id={stepId}
        className={cn(
          "w-full flex transition-all duration-300 animate-message-in",
          isUser && "justify-end",
          isHighlighted && "scale-[1.01]"
        )}
      >
        <div
          onClick={handleBubbleClick}
          className={cn(
            bubbleBase,
            isUser ? userBubble : agentBubble,
            isHighlighted && "ring-2 ring-foreground/20 shadow-lg",
            isClickable && "cursor-pointer hover:bg-muted/50 hover:shadow-md group"
          )}
        >
          {isUser ? (
            <p className="text-[14px] leading-6 whitespace-pre-wrap">{displayContent}</p>
          ) : (
            <div className="text-[14px] leading-relaxed prose prose-sm dark:prose-invert max-w-none prose-p:my-1 prose-p:leading-relaxed prose-headings:my-2 prose-headings:font-medium prose-ul:my-1 prose-ol:my-1 prose-li:my-0 prose-a:text-primary prose-a:no-underline hover:prose-a:underline prose-code:bg-primary/10 prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded-md prose-code:text-[13px] prose-code:font-normal prose-code:before:content-none prose-code:after:content-none prose-strong:font-semibold">
              <ReactMarkdown>{displayContent}</ReactMarkdown>
            </div>
          )}
          
          {shouldTruncate && (
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
          
          {isClickable && (
            <div className="mt-3 flex items-center gap-1.5 text-xs text-muted-foreground group-hover:text-foreground transition-colors">
              <span>Click for details</span>
              <ChevronRight className="h-3 w-3 group-hover:translate-x-0.5 transition-transform" />
            </div>
          )}
          
          {hasViewReport && onViewReport && (
            <button
              onClick={(e) => {
                e.stopPropagation()
                onViewReport()
              }}
              className="mt-3 inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-primary text-primary-foreground text-sm font-medium hover:bg-primary/90 transition-colors"
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
