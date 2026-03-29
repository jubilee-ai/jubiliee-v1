import { useState } from "react"
import { ChevronRight, Loader2 } from "lucide-react"
import { cn } from "@/lib/utils"

interface ThinkingBlockProps {
  content: string
  isActive?: boolean
  label?: string
}

export function ThinkingBlock({ content, isActive = false, label }: ThinkingBlockProps) {
  const [expanded, setExpanded] = useState(false)

  if (!content.trim()) return null

  const lines = content.split("\n").filter(Boolean)
  const hasStepLines = lines.some((l) => l.startsWith("⋯"))
  const summaryLabel = hasStepLines
    ? `Thought for ${lines.length} steps`
    : `Reasoned for ${Math.max(1, Math.ceil(content.length / 80))} lines`

  return (
    <div className="w-full my-1">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-1.5 text-xs text-muted-foreground/70 hover:text-muted-foreground transition-colors py-1 group"
      >
        {isActive ? (
          <Loader2 className="h-3 w-3 animate-spin text-primary/60" />
        ) : (
          <ChevronRight className={cn(
            "h-3 w-3 transition-transform duration-200",
            expanded && "rotate-90"
          )} />
        )}
        <span className="font-medium">
          {label || (isActive ? "Thinking..." : summaryLabel)}
        </span>
      </button>
      {expanded && (
        <div className="ml-5 mt-1 text-xs text-muted-foreground/60 border-l border-border/30 pl-3 py-1 max-h-[300px] overflow-y-auto">
          {hasStepLines ? (
            <div className="space-y-0.5">
              {lines.map((line, i) => (
                <div key={i} className="leading-relaxed">{line}</div>
              ))}
            </div>
          ) : (
            <pre className="whitespace-pre-wrap font-mono text-[11px] leading-relaxed">{content}</pre>
          )}
        </div>
      )}
    </div>
  )
}
