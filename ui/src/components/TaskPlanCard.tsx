import { Loader2, Moon } from "lucide-react"
import { Button } from "@/components/ui/button"
import { useMemo } from "react"
import type { ChatTaskPlanPayload } from "@/types/agent"
import type { Dataset as ApiDataset } from "@/lib/api"
import { resolveDatasetDisplayNames } from "@/lib/datasetDisplay"
import { truncateGoalForCard } from "@/lib/planDisplay"
import { cn } from "@/lib/utils"

export interface TaskPlanCardProps {
  payload: ChatTaskPlanPayload
  /** Workspace catalog; used to replace UUID refs with dataset names */
  datasets?: ApiDataset[]
  resolved?: boolean
  disabled?: boolean
  isRunning?: boolean
  onApproveStart: () => void
}

export function TaskPlanCard({
  payload,
  datasets,
  resolved = false,
  disabled = false,
  isRunning = false,
  onApproveStart,
}: TaskPlanCardProps) {
  const { plan } = payload
  const dataLine = useMemo(
    () =>
      plan.datasetLabels?.length
        ? resolveDatasetDisplayNames(plan.datasetLabels, payload.datasetRefs, datasets ?? []).join(", ")
        : "",
    [plan.datasetLabels, payload.datasetRefs, datasets],
  )

  return (
    <div
      className={cn(
        "mt-3 rounded-xl border border-border/50 bg-muted/20 px-4 py-3 text-left",
        resolved && "opacity-60",
      )}
    >
      <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground mb-2">
        Review plan
      </p>
      <p className="text-sm font-medium text-foreground leading-snug mb-2">
        {truncateGoalForCard(plan.goal)}
      </p>
      {dataLine ? (
        <p className="text-xs text-muted-foreground mb-3">
          Using data: <span className="text-foreground/85">{dataLine}</span>
        </p>
      ) : null}
      <p className="text-xs text-muted-foreground leading-relaxed mb-3">
        Approve to start the run, or reply in the chat if something should change first.
      </p>

      {resolved ? (
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <Loader2 className="h-3.5 w-3.5 animate-spin shrink-0 text-primary" aria-hidden />
          <span>Starting…</span>
        </div>
      ) : (
        <Button
          type="button"
          size="sm"
          className="gap-1.5 w-full sm:w-auto"
          disabled={disabled || isRunning}
          onClick={(e) => {
            e.stopPropagation()
            onApproveStart()
          }}
        >
          {isRunning ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Moon className="h-3.5 w-3.5" />}
          Approve &amp; start
        </Button>
      )}
    </div>
  )
}
