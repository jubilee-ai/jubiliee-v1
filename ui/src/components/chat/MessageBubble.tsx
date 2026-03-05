import { useState, forwardRef } from "react"
import ReactMarkdown from "react-markdown"
import { cn } from "@/lib/utils"
import { User, Bot, FileText, ChevronRight } from "lucide-react"
import type { ChatMessage } from "@/types/agent"

const MAX_CONTENT_LENGTH = 400 // Characters before truncation

interface MessageBubbleProps {
  message: ChatMessage
  isHighlighted?: boolean
  onViewReport?: () => void
  stepId?: string | null
  isClickable?: boolean
  onStepClick?: () => void
}

export const MessageBubble = forwardRef<HTMLDivElement, MessageBubbleProps>(
  function MessageBubble({ message, isHighlighted, onViewReport, stepId, isClickable, onStepClick }, ref) {
    const [isExpanded, setIsExpanded] = useState(false)
    const isUser = message.role === "user"
    const isSystem = message.role === "system"
    
    // Check if this message mentions "View Report" - show a clickable button
    const hasViewReport = !isUser && !isSystem && 
      (message.content.toLowerCase().includes("view report") || 
       message.content.toLowerCase().includes("training completed"))

    const shouldTruncate = message.content.length > MAX_CONTENT_LENGTH
    const displayContent = shouldTruncate && !isExpanded
      ? message.content.slice(0, MAX_CONTENT_LENGTH) + "..."
      : message.content

    if (isSystem) {
      const isStepAccepted = message.content.includes("Step accepted")
      const isRedo = message.content.includes("Requested redo")
      return (
        <div ref={ref} className={cn(
          "text-xs text-center py-2",
          isStepAccepted && "text-green-600 dark:text-green-500",
          isRedo && "text-amber-600 dark:text-amber-500",
          !isStepAccepted && !isRedo && "text-muted-foreground/70"
        )}>
          {message.content}
        </div>
      )
    }

    const handleBubbleClick = () => {
      if (isClickable && onStepClick) {
        onStepClick()
      }
    }

    return (
      <div 
        ref={ref}
        data-step-id={stepId}
        className={cn(
          "flex gap-3 transition-all duration-300",
          isUser && "flex-row-reverse",
          isHighlighted && "scale-[1.02]"
        )}
      >
        <div
          className={cn(
            "flex-shrink-0 w-7 h-7 rounded-full flex items-center justify-center",
            isUser ? "bg-foreground text-background" : "bg-muted"
          )}
        >
          {isUser ? <User className="h-3.5 w-3.5" /> : <Bot className="h-3.5 w-3.5 text-muted-foreground" />}
        </div>
        <div
          onClick={handleBubbleClick}
          className={cn(
            "flex-1 max-w-[92%] rounded-2xl px-4 py-3 transition-all duration-300",
            isUser ? "bg-foreground text-background" : "bg-muted/50",
            isHighlighted && "ring-2 ring-foreground/20 shadow-lg",
            isClickable && "cursor-pointer hover:bg-muted/70 hover:shadow-md group"
          )}
        >
          <div className="text-[15px] leading-relaxed prose prose-sm dark:prose-invert max-w-none prose-p:my-1 prose-p:leading-relaxed prose-headings:my-2 prose-headings:font-medium prose-ul:my-1 prose-ol:my-1 prose-li:my-0 prose-code:bg-black/5 prose-code:dark:bg-white/10 prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded-md prose-code:text-[13px] prose-code:font-normal prose-code:before:content-none prose-code:after:content-none prose-strong:font-semibold">
            <ReactMarkdown>{displayContent}</ReactMarkdown>
          </div>
          
          {/* Show more/less button */}
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
          
          {/* View Details hint for clickable messages */}
          {isClickable && (
            <div className="mt-3 flex items-center gap-1.5 text-xs text-muted-foreground group-hover:text-foreground transition-colors">
              <span>Click for details</span>
              <ChevronRight className="h-3 w-3 group-hover:translate-x-0.5 transition-transform" />
            </div>
          )}
          
          {/* View Report button */}
          {hasViewReport && onViewReport && (
            <button
              onClick={(e) => {
                e.stopPropagation()
                onViewReport()
              }}
              className="mt-3 inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-foreground text-background text-sm font-medium hover:bg-foreground/90 transition-colors"
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
