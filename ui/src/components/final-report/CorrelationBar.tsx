import { cn } from "@/lib/utils"

/**
 * Correlation bar visualization component
 * Shows a horizontal bar chart for feature correlations with the target variable
 */
export function CorrelationBar({
  feature,
  correlation,
  rank,
  compact,
}: {
  feature: string
  correlation: number
  rank: number
  /** Tighter layout for side-by-side or narrow contexts (e.g. chat preview). */
  compact?: boolean
}) {
  const absCorr = Math.abs(correlation)
  const isPositive = correlation >= 0
  const width = Math.min(100, absCorr * 100)

  return (
    <div className={cn("flex items-center gap-2 sm:gap-2.5 py-0.5", compact && "gap-1.5 py-px")}>
      <div
        className={cn(
          "text-muted-foreground text-right tabular-nums",
          compact ? "w-3.5 text-overline" : "w-5 text-xs",
        )}
      >
        {rank}
      </div>
      <div
        className={cn(
          "font-mono truncate min-w-0",
          compact ? "w-[4.5rem] text-overline" : "flex-1 max-w-[11rem] text-caption sm:text-xs",
        )}
        title={feature}
      >
        {feature}
      </div>
      <div className={cn("flex-1 flex items-center", compact ? "gap-1" : "gap-2")}>
        <div
        className={cn(
          "flex-1 bg-muted/30 rounded-full overflow-hidden relative",
          compact ? "h-2.5" : "h-2.5 sm:h-3",
        )}
        >
          {isPositive ? (
            <div
              className="absolute left-1/2 h-full bg-success/65 rounded-r-full transition-all"
              style={{ width: `${width / 2}%` }}
            />
          ) : (
            <div
              className="absolute right-1/2 h-full bg-red-500/70 rounded-l-full transition-all"
              style={{ width: `${width / 2}%` }}
            />
          )}
          {/* Center line */}
          <div className="absolute left-1/2 top-0 bottom-0 w-px bg-border" />
        </div>
        <span
          className={cn(
            "font-mono text-right shrink-0 tabular-nums",
            compact ? "w-10 text-overline" : "w-12 sm:w-14 text-caption sm:text-xs",
            isPositive ? "text-success" : "text-destructive",
          )}
        >
          {isPositive ? "+" : ""}
          {correlation.toFixed(3)}
        </span>
      </div>
    </div>
  )
}
