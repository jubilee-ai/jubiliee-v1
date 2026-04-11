import type { TrainingAgentState, FeatureCorrelation, NumericSummary } from "@/types/agent"
import { cn } from "@/lib/utils"
import {
  getFeatureSelectionKeyStats,
  hasFeatureAnalysisContent,
} from "@/components/final-report/FeatureAnalysisPanels"
import { CorrelationBar } from "@/components/final-report/CorrelationBar"

type Props = {
  agentState: TrainingAgentState
}

const PREVIEW_CORR = 8
const PREVIEW_NUMERIC_ROWS = 8

/** Compact side-by-side preview; full analysis in step “See details”. */
export function FeatureAnalysisChatCard({ agentState }: Props) {
  const keyStats = getFeatureSelectionKeyStats(agentState)
  if (!hasFeatureAnalysisContent(keyStats)) return null

  const allCorrs = keyStats.feature_correlations ?? []
  const allNumeric = keyStats.numeric_summaries ?? []
  const corrs = allCorrs.slice(0, PREVIEW_CORR)
  const numericRows = allNumeric.slice(0, PREVIEW_NUMERIC_ROWS)

  const showCorr = corrs.length > 0
  const showNumeric = numericRows.length > 0
  const split = showCorr && showNumeric

  return (
    <div
      className="mt-3 rounded-xl border border-border/30 bg-gradient-to-b from-muted/15 to-muted/5 overflow-hidden shadow-sm"
      onClick={(e) => e.stopPropagation()}
    >
      <div className="p-2.5 sm:p-3">
        <div
          className={cn(
            split
              ? "grid grid-cols-1 sm:grid-cols-2 sm:gap-0 gap-4"
              : "grid grid-cols-1 gap-3",
          )}
        >
          {showCorr ? (
            <section
              className={cn(
                "min-w-0 space-y-1.5",
                split && "sm:pr-3 sm:border-r border-border/25",
              )}
            >
              <h4 className="text-overline font-medium text-muted-foreground tracking-wide">Correlation vs target</h4>
              <div className="rounded-lg bg-background/40 border border-border/20 px-1.5 py-1">
                {corrs.map((corr: FeatureCorrelation, i: number) => (
                  <CorrelationBar
                    key={`${corr.feature}-${i}`}
                    feature={corr.feature}
                    correlation={corr.correlation}
                    rank={i + 1}
                    compact
                  />
                ))}
              </div>
            </section>
          ) : null}

          {showNumeric ? (
            <section className={cn("min-w-0 space-y-1.5", split && "sm:pl-3")}>
              <h4 className="text-overline font-medium text-muted-foreground tracking-wide">Numeric snapshot</h4>
              <div className="rounded-lg border border-border/20 overflow-hidden bg-background/40">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="bg-muted/35 text-muted-foreground">
                      <th className="text-left font-medium py-1 px-1.5">Feature</th>
                      <th className="text-right font-medium py-1 px-1.5 w-[3.25rem]">μ</th>
                      <th className="text-right font-medium py-1 px-1.5 w-[3rem] sm:w-[3.25rem]">σ</th>
                    </tr>
                  </thead>
                  <tbody>
                    {numericRows.map((stat: NumericSummary, i: number) => (
                      <tr key={i} className="border-t border-border/15 first:border-0">
                        <td
                          className="py-1 px-1.5 font-mono text-foreground/90 max-w-[0] truncate"
                          title={stat.column}
                        >
                          {stat.column}
                        </td>
                        <td className="py-1 px-1.5 text-right tabular-nums text-muted-foreground">
                          {stat.mean != null ? Number(stat.mean).toLocaleString(undefined, { maximumFractionDigits: 2 }) : "—"}
                        </td>
                        <td className="py-1 px-1.5 text-right tabular-nums text-muted-foreground">
                          {stat.std != null ? Number(stat.std).toLocaleString(undefined, { maximumFractionDigits: 2 }) : "—"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          ) : null}
        </div>
      </div>
    </div>
  )
}
