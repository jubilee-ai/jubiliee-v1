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
} from "@/types/agent"
import { Section, MetricBox } from "./shared"
import { CorrelationBar } from "./CorrelationBar"

interface AnalysisTabProps {
  agentState: TrainingAgentState
}

export function AnalysisTab({ agentState }: AnalysisTabProps) {
  // Extract key_stats from analysis_trace
  const analysisTrace = agentState.analysis_trace || []
  const featureAnalysis = analysisTrace.find((t) => t.step === "feature_selection_specification")
  const keyStats = (featureAnalysis?.key_stats || {}) as KeyStats

  const hasDatasetOverview = keyStats?.dataset_overview?.rows != null
  const hasNumericSummaries = keyStats?.numeric_summaries?.length > 0
  const hasCorrelations = keyStats?.feature_correlations?.length > 0
  const hasDistributionStats = keyStats?.distribution_stats?.length > 0
  const hasGroupSummaries = (keyStats?.group_summaries?.length ?? 0) > 0
  const hasConcentrationAnalysis = keyStats?.concentration_analysis?.length > 0

  const hasAnyData =
    hasDatasetOverview ||
    hasNumericSummaries ||
    hasCorrelations ||
    hasDistributionStats ||
    hasGroupSummaries ||
    hasConcentrationAnalysis

  if (!hasAnyData) {
    return (
      <div className="space-y-8">
        <Section title="Feature Analysis">
          <p className="text-muted-foreground">
            No feature analysis data available. This data is generated during the feature selection
            step.
          </p>
        </Section>
      </div>
    )
  }

  return (
    <div className="space-y-8">
      {/* Summary */}
      {keyStats.summary_text && (
        <div className="bg-muted/30 rounded-xl p-4 mb-4">
          <p className="text-sm leading-relaxed">{keyStats.summary_text}</p>
        </div>
      )}

      {/* Dataset Overview */}
      {hasDatasetOverview && (
        <Section title="Dataset Overview">
          <div className="flex flex-wrap gap-4 [&>div]:flex-[1_1_11rem] [&>div]:min-w-0 [&>div]:max-w-full">
            <MetricBox
              label="Rows"
              value={keyStats.dataset_overview.rows?.toLocaleString() || "N/A"}
            />
            <MetricBox
              label="Columns"
              value={String(keyStats.dataset_overview.columns || "N/A")}
            />
            <MetricBox
              label="Numeric Features"
              value={String(keyStats.dataset_overview.numeric_columns || "N/A")}
            />
            <MetricBox
              label="Categorical Features"
              value={String(keyStats.dataset_overview.categorical_columns || "N/A")}
            />
          </div>
        </Section>
      )}

      {/* Numeric Feature Summaries Table */}
      {hasNumericSummaries && (
        <Section title="Numeric Feature Statistics">
          <div className="overflow-x-auto rounded-lg border">
            <table className="w-full text-sm">
              <thead className="bg-muted/50">
                <tr>
                  <th className="text-left py-2 px-3 font-medium">Column</th>
                  <th className="text-right py-2 px-3 font-medium">Mean</th>
                  <th className="text-right py-2 px-3 font-medium">Std</th>
                  <th className="text-right py-2 px-3 font-medium">Min</th>
                  <th className="text-right py-2 px-3 font-medium">Max</th>
                  <th className="text-right py-2 px-3 font-medium">Median</th>
                  <th className="text-right py-2 px-3 font-medium">Skew</th>
                </tr>
              </thead>
              <tbody>
                {keyStats.numeric_summaries.map((stat: NumericSummary, i: number) => (
                  <tr key={i} className="border-t border-border/50 hover:bg-muted/30">
                    <td className="py-2 px-3 font-mono text-xs font-medium">{stat.column}</td>
                    <td className="text-right py-2 px-3">{stat.mean?.toLocaleString() ?? "N/A"}</td>
                    <td className="text-right py-2 px-3">{stat.std?.toLocaleString() ?? "N/A"}</td>
                    <td className="text-right py-2 px-3">{stat.min?.toLocaleString() ?? "N/A"}</td>
                    <td className="text-right py-2 px-3">{stat.max?.toLocaleString() ?? "N/A"}</td>
                    <td className="text-right py-2 px-3">
                      {stat.median?.toLocaleString() ?? "N/A"}
                    </td>
                    <td
                      className={`text-right py-2 px-3 ${stat.skew && Math.abs(stat.skew) > 1 ? "text-yellow-600 font-medium" : ""}`}
                    >
                      {stat.skew?.toFixed(2) ?? "N/A"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Section>
      )}

      {/* Feature Correlations Chart */}
      {hasCorrelations && (
        <Section title="Feature Correlations with Target">
          <div className="space-y-1">
            {keyStats.feature_correlations
              .slice(0, 15)
              .map((corr: FeatureCorrelation, i: number) => (
                <CorrelationBar
                  key={i}
                  feature={corr.feature}
                  correlation={corr.correlation}
                  rank={i + 1}
                />
              ))}
          </div>
        </Section>
      )}

      {/* Leakage Warnings */}
      {keyStats.leakage_warnings && keyStats.leakage_warnings.length > 0 && (
        <Section title="Leakage Warnings">
          <div className="bg-destructive/10 border border-destructive/20 rounded-xl p-4">
            <div className="flex items-center gap-2 text-destructive mb-3">
              <AlertTriangle className="h-4 w-4" />
              <span className="font-medium">Potential Data Leakage Detected</span>
            </div>
            <ul className="space-y-1">
              {keyStats.leakage_warnings.map((warning: string, i: number) => (
                <li key={i} className="text-sm text-destructive/80 flex gap-2">
                  <span>•</span>
                  <span>{warning}</span>
                </li>
              ))}
            </ul>
          </div>
        </Section>
      )}

      {/* Distribution Statistics with Histograms */}
      {hasDistributionStats && (
        <Section title="Distribution Analysis">
          <div className="space-y-6">
            {keyStats.distribution_stats.slice(0, 6).map((stat: DistributionStat, i: number) => (
              <div key={i} className="bg-muted/30 rounded-xl p-4">
                <div className="flex items-center justify-between mb-3">
                  <span className="font-mono text-sm font-medium">{stat.column}</span>
                  {stat.shape && <span className="text-xs text-muted-foreground">{stat.shape}</span>}
                </div>

                {/* Stats row */}
                <div className="grid grid-cols-2 md:grid-cols-6 gap-2 mb-4 text-xs">
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

                {/* Histogram */}
                {stat.histogram && stat.histogram.length > 0 && (
                  <div className="flex items-end gap-1 h-20">
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
                          <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1 opacity-0 group-hover:opacity-100 transition-opacity bg-foreground text-background text-xs px-2 py-1 rounded whitespace-nowrap z-10">
                            {bin.pct}%
                          </div>
                        </div>
                      )
                    })}
                  </div>
                )}
              </div>
            ))}
          </div>
        </Section>
      )}

      {/* Group Summaries with Bar Charts */}
      {hasGroupSummaries && (
        <Section title="Categorical Feature Analysis (Target Rate by Group)">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {keyStats.group_summaries!.map((group: GroupSummary, i: number) => (
              <div key={i} className="bg-muted/30 rounded-xl p-4">
                <div className="flex items-center justify-between mb-3">
                  <span className="font-mono text-sm font-medium">{group.column}</span>
                  <span className="text-xs text-muted-foreground">{group.n_groups} groups</span>
                </div>

                {/* Bar chart showing target rate per group */}
                <div className="space-y-2">
                  {group.groups.slice(0, 6).map((g, j) => {
                    const maxMean = Math.max(...group.groups.map((x) => x.mean || 0))
                    const width = maxMean > 0 ? ((g.mean || 0) / maxMean) * 100 : 0
                    return (
                      <div key={j} className="flex items-center gap-2">
                        <div className="w-24 text-xs truncate" title={String(g.value)}>
                          {g.value}
                        </div>
                        <div className="flex-1 h-5 bg-muted/50 rounded overflow-hidden">
                          <div
                            className="h-full bg-blue-500/70 rounded"
                            style={{ width: `${width}%` }}
                          />
                        </div>
                        <div className="w-16 text-xs text-right font-mono">
                          {((g.mean || 0) * 100).toFixed(1)}%
                        </div>
                      </div>
                    )
                  })}
                </div>
                {group.overall_mean != null && (
                  <div className="mt-2 pt-2 border-t text-xs text-muted-foreground">
                    Overall mean: {(group.overall_mean * 100).toFixed(1)}%
                  </div>
                )}
              </div>
            ))}
          </div>
        </Section>
      )}

      {/* Concentration Analysis with Lorenz Curves */}
      {hasConcentrationAnalysis && (
        <Section title="Concentration Analysis">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {keyStats.concentration_analysis.map((conc: ConcentrationStat, i: number) => (
              <div key={i} className="bg-muted/30 rounded-xl p-4">
                <div className="font-mono text-sm font-medium mb-3">{conc.column}</div>

                {/* Gini and interpretation */}
                {conc.gini != null && (
                  <div className="mb-3">
                    <div className="flex justify-between text-sm">
                      <span>Gini Coefficient</span>
                      <span className="font-medium">{conc.gini.toFixed(3)}</span>
                    </div>
                    {conc.gini_interpretation && (
                      <div className="text-xs text-muted-foreground">{conc.gini_interpretation}</div>
                    )}
                  </div>
                )}

                {/* Top N shares */}
                <div className="space-y-1.5 text-sm">
                  {conc.top_10pct_share != null && (
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Top 10% owns</span>
                      <span className="font-medium">{conc.top_10pct_share}%</span>
                    </div>
                  )}
                  {conc.top_50pct_share != null && (
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">Top 50% owns</span>
                      <span className="font-medium">{conc.top_50pct_share}%</span>
                    </div>
                  )}
                  {conc.pareto_80pct != null && (
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">80% owned by</span>
                      <span className="font-medium">{conc.pareto_80pct}%</span>
                    </div>
                  )}
                </div>

                {/* Mini Lorenz curve */}
                {conc.lorenz_curve && conc.lorenz_curve.length > 0 && (
                  <div className="mt-3 pt-3 border-t">
                    <div className="text-xs text-muted-foreground mb-2">Lorenz Curve</div>
                    <div className="h-16 flex items-end">
                      <svg viewBox="0 0 100 100" className="w-full h-full">
                        {/* Perfect equality line */}
                        <line
                          x1="0"
                          y1="100"
                          x2="100"
                          y2="0"
                          stroke="currentColor"
                          strokeOpacity="0.2"
                          strokeDasharray="2"
                        />
                        {/* Lorenz curve */}
                        <polyline
                          fill="none"
                          stroke="rgb(59, 130, 246)"
                          strokeWidth="2"
                          points={`0,100 ${conc.lorenz_curve.map((p: LorenzPoint) => `${p.pct_entities},${100 - p.pct_value}`).join(" ")}`}
                        />
                      </svg>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </Section>
      )}
    </div>
  )
}
