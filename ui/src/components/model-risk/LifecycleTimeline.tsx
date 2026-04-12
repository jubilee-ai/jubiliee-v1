import { Check, Circle } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import type { ModelRiskLifecycleStage } from "@/lib/api"
import { cn } from "@/lib/utils"

function stageBadgeVariant(
  status: ModelRiskLifecycleStage["status"],
): "default" | "secondary" | "outline" | "destructive" {
  if (status === "current") return "default"
  if (status === "complete") return "secondary"
  if (status === "skipped") return "outline"
  return "outline"
}

interface LifecycleTimelineProps {
  stages: ModelRiskLifecycleStage[]
  className?: string
}

export function LifecycleTimeline({ stages, className }: LifecycleTimelineProps) {
  return (
    <div className={cn("relative pl-2", className)}>
      <ul className="space-y-0">
        {stages.map((s, i) => {
          const isLast = i === stages.length - 1
          const done = s.status === "complete" || s.status === "skipped"
          return (
            <li key={s.id} className="relative flex gap-3 pb-6 last:pb-0">
              {!isLast && (
                <span
                  className="absolute left-[11px] top-6 bottom-0 w-px bg-border"
                  aria-hidden
                />
              )}
              <div className="relative z-[1] flex h-6 w-6 shrink-0 items-center justify-center rounded-full border bg-card">
                {done ? (
                  <Check className="h-3.5 w-3.5 text-primary" aria-hidden />
                ) : (
                  <Circle
                    className={cn(
                      "h-3 w-3",
                      s.status === "current" ? "text-primary fill-primary/25" : "text-muted-foreground/40",
                    )}
                    aria-hidden
                  />
                )}
              </div>
              <div className="min-w-0 flex-1 pt-0.5">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-sm font-medium text-foreground">{s.label}</span>
                  <Badge variant={stageBadgeVariant(s.status)} className="text-overline capitalize">
                    {s.status}
                  </Badge>
                </div>
                {s.entered_at && (
                  <p className="mt-1 text-xs text-muted-foreground tabular-nums">
                    {new Date(s.entered_at).toLocaleString()}
                  </p>
                )}
                {s.notes && (
                  <p className="mt-1 text-xs text-muted-foreground">{s.notes}</p>
                )}
              </div>
            </li>
          )
        })}
      </ul>
    </div>
  )
}
