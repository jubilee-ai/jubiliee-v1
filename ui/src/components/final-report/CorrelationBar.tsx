/**
 * Correlation bar visualization component
 * Shows a horizontal bar chart for feature correlations with the target variable
 */
export function CorrelationBar({
  feature,
  correlation,
  rank,
}: {
  feature: string
  correlation: number
  rank: number
}) {
  const absCorr = Math.abs(correlation)
  const isPositive = correlation >= 0
  const width = Math.min(100, absCorr * 100)

  return (
    <div className="flex items-center gap-3 py-1.5">
      <div className="w-5 text-xs text-muted-foreground text-right">{rank}</div>
      <div className="w-28 font-mono text-xs truncate" title={feature}>
        {feature}
      </div>
      <div className="flex-1 flex items-center gap-2">
        <div className="flex-1 h-5 bg-muted/30 rounded-full overflow-hidden relative">
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
          className={`w-14 text-xs font-mono text-right ${isPositive ? "text-success" : "text-destructive"}`}
        >
          {isPositive ? "+" : ""}
          {correlation.toFixed(3)}
        </span>
      </div>
    </div>
  )
}
