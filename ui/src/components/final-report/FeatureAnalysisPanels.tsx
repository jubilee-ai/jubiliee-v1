import type { ReactNode } from "react"
import { AlertTriangle } from "lucide-react"
import type {
  TrainingAgentState,
  KeyStats,
  FeatureCorrelation,
  DistributionStat,
  NumericSummary,
  GroupSummary,
  ConcentrationStat,
  HistogramBin,
  LorenzPoint,
  StatisticalTestRow,
} from "@/types/agent"
import { Section, MetricBox } from "./shared"
import { CorrelationBar } from "./CorrelationBar"

export type FeatureAnalysisDensity = "full" | "compact"

export function getFeatureSelectionKeyStats(agentState: TrainingAgentState): KeyStats {
  const analysisTrace = agentState.analysis_trace || []
  const featureAnalysis = analysisTrace.find((t) => t.step === "feature_selection_specification")
  return (featureAnalysis?.key_stats || {}) as KeyStats
}

export function hasFeatureAnalysisContent(keyStats: KeyStats): boolean {
  const hasDatasetOverview = keyStats?.dataset_overview?.rows != null
  const hasNumericSummaries = (keyStats?.numeric_summaries?.length ?? 0) > 0
  const hasCorrelations = (keyStats?.feature_correlations?.length ?? 0) > 0
  const hasDistributionStats = (keyStats?.distribution_stats?.length ?? 0) > 0
  const hasGroupSummaries = (keyStats?.group_summaries?.length ?? 0) > 0
  const hasConcentrationAnalysis = (keyStats?.concentration_analysis?.length ?? 0) > 0
  const hasInferential = (keyStats?.statistical_tests?.length ?? 0) > 0
  return (
    hasDatasetOverview ||
    hasNumericSummaries ||
    hasCorrelations ||
    hasDistributionStats ||
    hasGroupSummaries ||
    hasConcentrationAnalysis ||
    hasInferential
  )
}

function sliceLimit<T>(arr: T[] | undefined, max: number): T[] {
  if (!arr?.length) return []
  if (!Number.isFinite(max) || max >= arr.length) return arr
  return arr.slice(0, max)
}

function PanelSection({
  title,
  density,
  children,
}: {
  title: string
  density: FeatureAnalysisDensity
  children: ReactNode
}) {
  if (density === "compact") {
    return (
      <div>
        <h3 className="text-caption font-semibold uppercase tracking-wider text-muted-foreground mb-2">
          {title}
        </h3>
        {children}
      </div>
    )
  }
  return <Section title={title}>{children}</Section>
}

export interface FeatureAnalysisPanelsProps {
  keyStats: KeyStats
  density?: FeatureAnalysisDensity
}

/**
 * Shared EDA panels (numeric stats, correlations, distributions, categorical rates, concentration).
 * Used by the Final Report Analysis tab and the in-chat digest.
 */
export function FeatureAnalysisPanels({ keyStats, density = "full" }: FeatureAnalysisPanelsProps) {
  const compact = density === "compact"
  const L = compact
    ? { nNum: 8, nCorr: 8, nDist: 3, nGroupCards: 4, nGroupRows: 5, nConc: 3 }
    : { nNum: Number.POSITIVE_INFINITY, nCorr: 15, nDist: 6, nGroupCards: Number.POSITIVE_INFINITY, nGroupRows: 6, nConc: Number.POSITIVE_INFINITY }

  const hasDatasetOverview = keyStats?.dataset_overview?.rows != null
  const hasNumericSummaries = keyStats?.numeric_summaries?.length > 0
  const hasCorrelations = keyStats?.feature_correlations?.length > 0
  const hasDistributionStats = keyStats?.distribution_stats?.length > 0
  const hasGroupSummaries = (keyStats?.group_summaries?.length ?? 0) > 0
  const hasConcentrationAnalysis = keyStats?.concentration_analysis?.length > 0
  const hasInferential = (keyStats?.statistical_tests?.length ?? 0) > 0

  const histClass = compact ? "h-10" : "h-10 sm:h-12"
  const distBoxClass = compact ? "rounded-lg p-3" : "rounded-lg p-3"
  const numericTableClass = compact ? "text-xs" : "text-xs"
  const summaryClamp = compact ? "line-clamp-5 text-xs leading-relaxed" : "text-xs leading-relaxed max-w-4xl"

  const spaceY = compact ? "space-y-4" : "space-y-6"

  return (
    <div className={spaceY}>
      {keyStats.summary_text ? (
        <div className={`bg-muted/30 ${compact ? "rounded-lg p-3 mb-1" : "rounded-lg p-3 mb-2"}`}>
          <p className={summaryClamp}>{keyStats.summary_text}</p>
        </div>
      ) : null}

      {hasDatasetOverview && (
        <PanelSection title="Dataset Overview" density={density}>
          <div className="flex flex-wrap gap-2 sm:gap-4 [&>div]:flex-[1_1_9rem] [&>div]:min-w-0 [&>div]:max-w-full">
            <MetricBox
              className={compact ? "!p-2.5" : undefined}
              label="Rows"
              value={keyStats.dataset_overview.rows?.toLocaleString() || "N/A"}
            />
            <MetricBox
              className={compact ? "!p-2.5" : undefined}
              label="Columns"
              value={String(keyStats.dataset_overview.columns || "N/A")}
            />
            <MetricBox
              className={compact ? "!p-2.5" : undefined}
              label="Numeric Features"
              value={String(keyStats.dataset_overview.numeric_columns || "N/A")}
            />
            <MetricBox
              className={compact ? "!p-2.5" : undefined}
              label="Categorical Features"
              value={String(keyStats.dataset_overview.categorical_columns || "N/A")}
            />
          </div>
        </PanelSection>
      )}

      {hasNumericSummaries && (
        <PanelSection title="Numeric Feature Statistics" density={density}>
          <div className="overflow-x-auto rounded-lg border border-border/50">
            <table className={`w-full ${numericTableClass}`}>
              <thead className="bg-muted/50">
                <tr>
                  <th className="text-left py-1.5 px-2 sm:py-2 sm:px-3 font-medium">Column</th>
                  <th className="text-right py-1.5 px-2 sm:py-2 sm:px-3 font-medium">Mean</th>
                  <th className="text-right py-1.5 px-2 sm:py-2 sm:px-3 font-medium">Std</th>
                  <th className="text-right py-1.5 px-2 sm:py-2 sm:px-3 font-medium">Min</th>
                  <th className="text-right py-1.5 px-2 sm:py-2 sm:px-3 font-medium">Max</th>
                  <th className="text-right py-1.5 px-2 sm:py-2 sm:px-3 font-medium">Median</th>
                  <th className="text-right py-1.5 px-2 sm:py-2 sm:px-3 font-medium">Skew</th>
                </tr>
              </thead>
              <tbody>
                {sliceLimit(keyStats.numeric_summaries, L.nNum).map((stat: NumericSummary, i: number) => (
                  <tr key={i} className="border-t border-border/50 hover:bg-muted/30">
                    <td className="py-1.5 px-2 sm:py-2 sm:px-3 font-mono text-xs font-medium">
                      {stat.column}
                    </td>
                    <td className="text-right py-1.5 px-2 sm:py-2 sm:px-3">{stat.mean?.toLocaleString() ?? "N/A"}</td>
                    <td className="text-right py-1.5 px-2 sm:py-2 sm:px-3">{stat.std?.toLocaleString() ?? "N/A"}</td>
                    <td className="text-right py-1.5 px-2 sm:py-2 sm:px-3">{stat.min?.toLocaleString() ?? "N/A"}</td>
                    <td className="text-right py-1.5 px-2 sm:py-2 sm:px-3">{stat.max?.toLocaleString() ?? "N/A"}</td>
                    <td className="text-right py-1.5 px-2 sm:py-2 sm:px-3">
                      {stat.median?.toLocaleString() ?? "N/A"}
                    </td>
                    <td
                      className={`text-right py-1.5 px-2 sm:py-2 sm:px-3 ${stat.skew && Math.abs(stat.skew) > 1 ? "text-yellow-600 font-medium" : ""}`}
                    >
                      {stat.skew?.toFixed(2) ?? "N/A"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </PanelSection>
      )}

      {hasCorrelations && (
        <PanelSection title="Feature Correlations with Target" density={density}>
          <div className="space-y-0.5">
            {sliceLimit(keyStats.feature_correlations, L.nCorr).map((corr: FeatureCorrelation, i: number) => (
              <CorrelationBar
                key={i}
                feature={corr.feature}
                correlation={corr.correlation}
                rank={i + 1}
              />
            ))}
          </div>
        </PanelSection>
      )}

      {hasInferential && (
        <PanelSection title="Inferential statistics (exploratory)" density={density}>
          <p className="text-xs text-muted-foreground mb-2">
            p-values are not adjusted for multiple comparisons. Use for screening, not final claims.
          </p>
          <div className="overflow-x-auto rounded-lg border border-border/50">
            <table className={`w-full ${numericTableClass}`}>
              <thead className="bg-muted/50">
                <tr>
                  <th className="text-left py-1.5 px-2 font-medium">Feature</th>
                  <th className="text-left py-1.5 px-2 font-medium">Test</th>
                  <th className="text-right py-1.5 px-2 font-medium">p</th>
                  <th className="text-right py-1.5 px-2 font-medium">Effect</th>
                </tr>
              </thead>
              <tbody>
                {sliceLimit(keyStats.statistical_tests, compact ? 8 : 20).map((row: StatisticalTestRow, i: number) => {
                  const p = row.p_value
                  const sig = p != null && p < 0.05
                  const effect =
                    row.cramers_v != null
                      ? `V=${row.cramers_v.toFixed(3)}`
                      : row.cohens_d != null
                        ? `d=${row.cohens_d.toFixed(3)}`
                        : "—"
                  return (
                    <tr key={i} className="border-t border-border/50 hover:bg-muted/30">
                      <td className="py-1.5 px-2 font-mono max-w-[140px] truncate">{row.feature ?? "—"}</td>
                      <td className="py-1.5 px-2 text-muted-foreground">{row.test ?? "—"}</td>
                      <td
                        className={`text-right py-1.5 px-2 ${sig ? "text-amber-700 dark:text-amber-400 font-medium" : ""}`}
                      >
                        {p != null ? p.toExponential(2) : "—"}
                      </td>
                      <td className="text-right py-1.5 px-2">{effect}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </PanelSection>
      )}

      {keyStats.leakage_warnings && keyStats.leakage_warnings.length > 0 && (
        <PanelSection title="Leakage Warnings" density={density}>
          <div className="bg-destructive/10 border border-destructive/20 rounded-lg p-3">
            <div className="flex items-center gap-2 text-destructive mb-2 sm:mb-3">
              <AlertTriangle className="h-4 w-4 shrink-0" />
              <span className="font-medium text-xs">Potential Data Leakage Detected</span>
            </div>
            <ul className="space-y-1">
              {keyStats.leakage_warnings.map((warning: string, i: number) => (
                <li key={i} className="text-xs text-destructive/80 flex gap-2">
                  <span>•</span>
                  <span>{warning}</span>
                </li>
              ))}
            </ul>
          </div>
        </PanelSection>
      )}

      {hasDistributionStats && (
        <PanelSection title="Distribution Analysis" density={density}>
          <div className={compact ? "space-y-3" : "space-y-6"}>
            {sliceLimit(keyStats.distribution_stats, L.nDist).map((stat: DistributionStat, i: number) => (
              <div key={i} className={`bg-muted/30 ${distBoxClass}`}>
                <div className="flex items-center justify-between mb-2 sm:mb-3">
                  <span className="font-mono text-xs font-medium">{stat.column}</span>
                  {stat.shape ? (
                    <span className="text-overline sm:text-xs text-muted-foreground">{stat.shape}</span>
                  ) : null}
                </div>

                <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-1.5 sm:gap-2 mb-3 sm:mb-4 text-overline sm:text-xs">
                  <div>
                    <span className="text-muted-foreground">Mean:</span>{" "}
                    <span className="font-medium">{stat.mean?.toLocaleString() ?? "N/A"}</span>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Std:</span>{" "}
                    <span className="font-medium">{stat.std?.toLocaleString() ?? "N/A"}</span>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Min:</span>{" "}
                    <span className="font-medium">{stat.min?.toLocaleString() ?? "N/A"}</span>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Max:</span>{" "}
                    <span className="font-medium">{stat.max?.toLocaleString() ?? "N/A"}</span>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Skew:</span>{" "}
                    <span
                      className={`font-medium ${stat.skewness && Math.abs(stat.skewness) > 1 ? "text-yellow-600" : ""}`}
                    >
                      {stat.skewness?.toFixed(2) ?? "N/A"}
                    </span>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Kurtosis:</span>{" "}
                    <span
                      className={`font-medium ${stat.kurtosis && Math.abs(stat.kurtosis) > 3 ? "text-yellow-600" : ""}`}
                    >
                      {stat.kurtosis?.toFixed(2) ?? "N/A"}
                    </span>
                  </div>
                </div>

                {stat.histogram && stat.histogram.length > 0 ? (
                  <div className={`flex items-end gap-0.5 max-w-2xl ${histClass}`}>
                    {stat.histogram.map((bin: HistogramBin, j: number) => {
                      const maxPct = Math.max(...stat.histogram!.map((b) => b.pct))
                      const height = (bin.pct / maxPct) * 100
                      return (
                        <div
                          key={j}
                          className="flex-1 bg-blue-500/60 rounded-t hover:bg-blue-500/80 transition-colors relative group"
                          style={{ height: `${height}%`, minHeight: "2px" }}
                          title={`${bin.range}: ${bin.count.toLocaleString()} (${bin.pct}%)`}
                        >
                          <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1 opacity-0 group-hover:opacity-100 transition-opacity bg-foreground text-background text-overline px-1.5 py-0.5 rounded whitespace-nowrap z-10">
                            {bin.pct}%
                          </div>
                        </div>
                      )
                    })}
                  </div>
                ) : null}
              </div>
            ))}
          </div>
        </PanelSection>
      )}

      {hasGroupSummaries && keyStats.group_summaries && (
        <PanelSection title="Categorical Feature Analysis (Target Rate by Group)" density={density}>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 sm:gap-6">
            {sliceLimit(keyStats.group_summaries, L.nGroupCards).map((group: GroupSummary, i: number) => (
              <div key={i} className={`bg-muted/30 ${distBoxClass}`}>
                <div className="flex items-center justify-between mb-2 sm:mb-3">
                  <span className="font-mono text-xs font-medium">{group.column}</span>
                  <span className="text-overline sm:text-xs text-muted-foreground">{group.n_groups} groups</span>
                </div>

                <div className="space-y-1.5 sm:space-y-2">
                  {sliceLimit(group.groups, L.nGroupRows).map((g, j) => {
                    const maxMean = Math.max(...group.groups.map((x) => x.mean || 0))
                    const width = maxMean > 0 ? ((g.mean || 0) / maxMean) * 100 : 0
                    return (
                      <div key={j} className="flex items-center gap-1.5 sm:gap-2">
                        <div className="w-20 sm:w-24 text-overline sm:text-xs truncate" title={String(g.value)}>
                          {g.value}
                        </div>
                        <div className="flex-1 h-3 sm:h-3.5 bg-muted/50 rounded overflow-hidden">
                          <div className="h-full bg-blue-500/70 rounded" style={{ width: `${width}%` }} />
                        </div>
                        <div className="w-12 sm:w-16 text-overline sm:text-xs text-right font-mono">
                          {((g.mean || 0) * 100).toFixed(1)}%
                        </div>
                      </div>
                    )
                  })}
                </div>
                {group.overall_mean != null ? (
                  <div className="mt-2 pt-2 border-t text-overline sm:text-xs text-muted-foreground">
                    Overall mean: {(group.overall_mean * 100).toFixed(1)}%
                  </div>
                ) : null}
              </div>
            ))}
          </div>
        </PanelSection>
      )}

      {hasConcentrationAnalysis && (
        <PanelSection title="Concentration Analysis" density={density}>
          <div
            className={
              compact ? "grid grid-cols-1 sm:grid-cols-2 gap-3" : "grid grid-cols-1 md:grid-cols-3 gap-3"
            }
          >
            {sliceLimit(keyStats.concentration_analysis, L.nConc).map((conc: ConcentrationStat, i: number) => (
              <div key={i} className={`bg-muted/30 ${distBoxClass}`}>
                <div className="font-mono text-xs font-medium mb-2 sm:mb-3">{conc.column}</div>

                {conc.gini != null ? (
                  <div className="mb-2 sm:mb-3">
                    <div className="flex justify-between text-xs">
                      <span>Gini Coefficient</span>
                      <span className="font-medium">{conc.gini.toFixed(3)}</span>
                    </div>
                    {conc.gini_interpretation ? (
                      <div className="text-overline sm:text-xs text-muted-foreground">{conc.gini_interpretation}</div>
                    ) : null}
                  </div>
                ) : null}

                <div className="space-y-1 sm:space-y-1.5 text-xs">
                  {conc.top_10pct_share != null ? (
                    <div className="flex justify-between gap-2">
                      <span className="text-muted-foreground">Top 10% owns</span>
                      <span className="font-medium">{conc.top_10pct_share}%</span>
                    </div>
                  ) : null}
                  {conc.top_50pct_share != null ? (
                    <div className="flex justify-between gap-2">
                      <span className="text-muted-foreground">Top 50% owns</span>
                      <span className="font-medium">{conc.top_50pct_share}%</span>
                    </div>
                  ) : null}
                  {conc.pareto_80pct != null ? (
                    <div className="flex justify-between gap-2">
                      <span className="text-muted-foreground">80% owned by</span>
                      <span className="font-medium">{conc.pareto_80pct}%</span>
                    </div>
                  ) : null}
                </div>

                {conc.lorenz_curve && conc.lorenz_curve.length > 0 ? (
                  <div className="mt-2 sm:mt-3 pt-2 sm:pt-3 border-t">
                    <div className="text-overline sm:text-xs text-muted-foreground mb-1 sm:mb-2">Lorenz Curve</div>
                    <div className={`${compact ? "h-9" : "h-11 sm:h-12"} flex items-end max-w-xs mx-auto w-full`}>
                      <svg viewBox="0 0 100 100" className="w-full h-full">
                        <line
                          x1="0"
                          y1="100"
                          x2="100"
                          y2="0"
                          stroke="currentColor"
                          strokeOpacity="0.2"
                          strokeDasharray="2"
                        />
                        <polyline
                          fill="none"
                          stroke="rgb(59, 130, 246)"
                          strokeWidth="1.5"
                          points={`0,100 ${conc.lorenz_curve.map((p: LorenzPoint) => `${p.pct_entities},${100 - p.pct_value}`).join(" ")}`}
                        />
                      </svg>
                    </div>
                  </div>
                ) : null}
              </div>
            ))}
          </div>
        </PanelSection>
      )}
    </div>
  )
}
