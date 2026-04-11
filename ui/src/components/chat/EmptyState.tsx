import { MessageSquare, Moon, ChevronRight } from "lucide-react"
import { cn } from "@/lib/utils"

interface EmptyStateProps {
  onAssignTask?: () => void
}

export function EmptyState({ onAssignTask }: EmptyStateProps) {
  return (
    <div className="py-10 px-2 sm:px-4 relative">
      <div className="absolute inset-0 flex items-center justify-center pointer-events-none" aria-hidden>
        <div className="w-80 h-80 bg-primary/[0.04] rounded-full blur-3xl" />
      </div>

      <div className="relative max-w-4xl mx-auto">
        <p className="text-center text-overline font-bold uppercase tracking-[0.2em] text-muted-foreground/45 mb-6">
          Experimental Lab
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div
            className={cn(
              "rounded-2xl border border-border/45 bg-card/70 backdrop-blur-sm p-6 text-left",
              "shadow-[0_2px_16px_-4px_rgba(0,0,0,0.06)]",
            )}
          >
            <div className="rounded-xl bg-muted/30 w-11 h-11 flex items-center justify-center mb-4">
              <MessageSquare className="h-5 w-5 text-primary/70" />
            </div>
            <h3 className="font-headline font-semibold text-base sm:text-lg tracking-tight text-foreground mb-1">Chat</h3>
            <p className="text-sm text-muted-foreground leading-relaxed">Use the chat below.</p>
          </div>

          <button
            type="button"
            disabled={!onAssignTask}
            onClick={() => onAssignTask?.()}
            className={cn(
              "rounded-2xl border border-primary/25 bg-card/85 backdrop-blur-sm p-6 text-left text-foreground",
              "shadow-[0_2px_20px_-4px_rgba(18,86,210,0.12)]",
              "transition-all hover:border-primary/45 hover:shadow-[0_8px_28px_-8px_rgba(18,86,210,0.18)]",
              "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
              "disabled:opacity-50 disabled:pointer-events-none active:scale-[0.99]",
            )}
          >
            <div className="rounded-xl bg-primary/10 w-11 h-11 flex items-center justify-center mb-4">
              <Moon className="h-5 w-5 text-primary" />
            </div>
            <h3 className="font-headline font-semibold text-base sm:text-lg tracking-tight mb-1">Assign a task</h3>
            <p className="text-sm text-muted-foreground leading-relaxed mb-4">
              Background training — Jubilee confirms the plan, then runs for you.
            </p>
            <span className="inline-flex items-center gap-1 text-sm font-medium text-primary">
              Get started
              <ChevronRight className="h-4 w-4" />
            </span>
          </button>
        </div>
      </div>
    </div>
  )
}
