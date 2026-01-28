import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ScrollArea } from "@/components/ui/scroll-area"
import type { TrainingAgentState, StepInfo, KeyStats, FeatureCorrelation, DistributionStat, NumericSummary, GroupSummary, ConcentrationStat, HistogramBin, LorenzPoint } from "@/types/agent"
import { formatNumber, formatPercent } from "@/lib/utils"
import {
  X,
  Download,
  Copy,
  Check,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  AlertTriangle,
  TrendingUp,
  BarChart3,
} from "lucide-react"

interface FinalReportProps {
  agentState: TrainingAgentState
  steps: StepInfo[]
  onClose: () => void
}

export function FinalReport({ agentState, steps, onClose }: FinalReportProps) {
  const [activeTab, setActiveTab] = useState("summary")
  const [copied, setCopied] = useState(false)
  const metrics = agentState.training_metrics

  const hasResults = metrics?.success || 
    agentState.model_weights_path || 
    agentState.report_path ||
    (agentState.audit_trace && agentState.audit_trace.length > 0)

  const copyReport = () => {
    const report = generateTextReport(agentState, steps)
    navigator.clipboard.writeText(report)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const downloadReport = () => {
    const report = generateJsonReport(agentState, steps)
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = `${metrics?.model_name || "model"}_report.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  if (!hasResults) {
    return (
      <div className="fixed inset-0 z-50 bg-background/80 backdrop-blur-sm flex items-center justify-center p-6">
        <div className="bg-background rounded-2xl border shadow-2xl p-6 max-w-md">
          <h2 className="text-xl font-semibold mb-2">Training Report</h2>
          <p className="text-muted-foreground mb-4">Training has not completed yet or no results available.</p>
          <p className="text-sm text-muted-foreground mb-2">Current step: {agentState.current_step || "unknown"}</p>
          {agentState.error && <p className="text-sm text-destructive mb-4">Error: {agentState.error}</p>}
          <Button onClick={onClose}>Close</Button>
        </div>
      </div>
    )
  }

  return (
    <div className="fixed inset-0 z-50 bg-background/80 backdrop-blur-sm flex items-center justify-center p-6">
      <div className="w-full max-w-4xl h-[85vh] bg-background rounded-2xl border shadow-2xl flex flex-col overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b">
          <div>
            <h2 className="text-xl font-semibold tracking-tight">Training Report</h2>
            <p className="text-sm text-muted-foreground mt-0.5">
              {metrics?.model_name || agentState.model_weights_path || "Model training complete"}
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="ghost" size="sm" onClick={copyReport} className="h-8 gap-2">
              {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
              <span className="hidden sm:inline">{copied ? "Copied" : "Copy"}</span>
            </Button>
            <Button variant="ghost" size="sm" onClick={downloadReport} className="h-8 gap-2">
              <Download className="h-4 w-4" />
              <span className="hidden sm:inline">Download</span>
            </Button>
            <Button variant="ghost" size="sm" onClick={onClose} className="h-8 w-8 p-0">
              <X className="h-4 w-4" />
            </Button>
          </div>
        </div>

        {/* Tabs */}
        <Tabs value={activeTab} onValueChange={setActiveTab} className="flex-1 flex flex-col min-h-0">
          <div className="px-6 pt-2 border-b">
            <TabsList className="bg-transparent p-0 h-auto gap-6">
              <TabsTrigger value="summary" className="bg-transparent px-0 pb-3 pt-0 rounded-none border-b-2 border-transparent data-[state=active]:border-foreground data-[state=active]:bg-transparent data-[state=active]:shadow-none">
                Summary
              </TabsTrigger>
              <TabsTrigger value="metrics" className="bg-transparent px-0 pb-3 pt-0 rounded-none border-b-2 border-transparent data-[state=active]:border-foreground data-[state=active]:bg-transparent data-[state=active]:shadow-none">
                Metrics
              </TabsTrigger>
              <TabsTrigger value="analysis" className="bg-transparent px-0 pb-3 pt-0 rounded-none border-b-2 border-transparent data-[state=active]:border-foreground data-[state=active]:bg-transparent data-[state=active]:shadow-none">
                Analysis
              </TabsTrigger>
              <TabsTrigger value="features" className="bg-transparent px-0 pb-3 pt-0 rounded-none border-b-2 border-transparent data-[state=active]:border-foreground data-[state=active]:bg-transparent data-[state=active]:shadow-none">
                Features
              </TabsTrigger>
              <TabsTrigger value="trace" className="bg-transparent px-0 pb-3 pt-0 rounded-none border-b-2 border-transparent data-[state=active]:border-foreground data-[state=active]:bg-transparent data-[state=active]:shadow-none">
                Trace
              </TabsTrigger>
            </TabsList>
          </div>

          <ScrollArea className="flex-1">
            <div className="p-6">
              <TabsContent value="summary" className="mt-0">
                <SummaryTab agentState={agentState} />
              </TabsContent>
              <TabsContent value="metrics" className="mt-0">
                <MetricsTab agentState={agentState} />
              </TabsContent>
              <TabsContent value="analysis" className="mt-0">
                <AnalysisTab agentState={agentState} />
              </TabsContent>
              <TabsContent value="features" className="mt-0">
                <FeaturesTab agentState={agentState} />
              </TabsContent>
              <TabsContent value="trace" className="mt-0">
                <TraceTab steps={steps} agentState={agentState} />
              </TabsContent>
            </div>
          </ScrollArea>
        </Tabs>
      </div>
    </div>
  )
}

function SummaryTab({ agentState }: { agentState: TrainingAgentState }) {
  const metrics = agentState.training_metrics
  const dataStep = agentState.audit_trace?.find(t => t.step === "data_collection") as Record<string, unknown> | undefined
  const featureStep = agentState.audit_trace?.find(t => t.step === "feature_engineering_executor") as Record<string, unknown> | undefined
  
  const lastIteration = metrics?.iterations?.[metrics.iterations.length - 1]
  const testR2 = metrics?.test_r2 ?? lastIteration?.test_r2
  const testRmse = metrics?.test_rmse ?? lastIteration?.test_rmse
  const testMae = metrics?.test_mae ?? lastIteration?.test_mae
  const valR2 = metrics?.val_r2 ?? lastIteration?.val_r2
  
  const hasClassificationMetrics = metrics?.test_accuracy != null || metrics?.test_roc_auc != null
  const hasRegressionMetrics = testR2 != null || testRmse != null || testMae != null

  return (
    <div className="space-y-8">
      {/* Hero metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {hasClassificationMetrics ? (
          <>
            <MetricBox label="Test Accuracy" value={formatPercent(metrics?.test_accuracy)} highlight />
            <MetricBox label="Test ROC-AUC" value={formatNumber(metrics?.test_roc_auc, 3)} highlight />
          </>
        ) : hasRegressionMetrics ? (
          <>
            <MetricBox label="Test R²" value={formatNumber(testR2, 4)} highlight />
            <MetricBox label="Test RMSE" value={formatNumber(testRmse, 2)} highlight />
          </>
        ) : (
          <>
            <MetricBox label="Status" value={metrics?.success ? "Success" : "Complete"} highlight />
            <MetricBox label="Model" value={metrics?.model_type || agentState.selected_model || "N/A"} />
          </>
        )}
        <MetricBox label="Iterations" value={String(metrics?.num_iterations || 1)} />
        <MetricBox label="Model Type" value={metrics?.model_type || agentState.selected_model || "N/A"} />
      </div>

      {/* Overview */}
      <Section title="Overview">
        <div className="space-y-3">
          <InfoRow label="Goal" value={agentState.goal} />
          <InfoRow label="Dataset" value={agentState.collected_dataset_ref} />
          <InfoRow label="Target Column" value={agentState.label_definition?.target_column} />
          <InfoRow label="Split Strategy" value={agentState.label_definition?.split_strategy} />
        </div>
      </Section>

      {/* Key Insights */}
      <Section title="Key Insights">
        <ul className="space-y-2 text-sm text-muted-foreground">
          {hasClassificationMetrics ? (
            <li className="flex gap-2">
              <span className="text-muted-foreground/50">•</span>
              Model achieved {formatPercent(metrics?.test_accuracy)} test accuracy with ROC-AUC of {formatNumber(metrics?.test_roc_auc, 3)}
            </li>
          ) : hasRegressionMetrics ? (
            <li className="flex gap-2">
              <span className="text-muted-foreground/50">•</span>
              Model achieved Test R² of {formatNumber(testR2, 4)} with RMSE of {formatNumber(testRmse, 2)} and MAE of {formatNumber(testMae, 2)}
            </li>
          ) : (
            <li className="flex gap-2">
              <span className="text-muted-foreground/50">•</span>
              Model training completed successfully using {metrics?.model_type || agentState.selected_model}
            </li>
          )}
          {hasRegressionMetrics && valR2 != null && (
            <li className="flex gap-2">
              <span className="text-muted-foreground/50">•</span>
              Validation R² was {formatNumber(valR2, 4)}, indicating good generalization
            </li>
          )}
          <li className="flex gap-2">
            <span className="text-muted-foreground/50">•</span>
            Training converged after {metrics?.num_iterations || 1} iteration(s)
          </li>
          <li className="flex gap-2">
            <span className="text-muted-foreground/50">•</span>
            {agentState.feature_spec?.features.length || (featureStep?.features_created as unknown[])?.length || 0} features were engineered from the original dataset
          </li>
          <li className="flex gap-2">
            <span className="text-muted-foreground/50">•</span>
            Data was split using {agentState.label_definition?.split_strategy || "random"} strategy (70/15/15)
          </li>
          {dataStep?.rows != null && (
            <li className="flex gap-2">
              <span className="text-muted-foreground/50">•</span>
              Original dataset had {String(dataStep.rows)} rows
            </li>
          )}
        </ul>
      </Section>

      {/* Training Summary */}
      {metrics?.summary && (
        <Section title="Training Summary">
          <p className="text-sm leading-relaxed">{metrics.summary}</p>
        </Section>
      )}

      {/* Recommendations */}
      {metrics?.recommendations && (
        <Section title="Recommendations">
          <p className="text-sm leading-relaxed">{metrics.recommendations}</p>
        </Section>
      )}

      {/* Model Details */}
      <Section title="Model Details">
        <div className="space-y-3">
          <div>
            <div className="text-sm text-muted-foreground mb-1">Model Name</div>
            <code className="text-sm bg-muted/50 px-3 py-2 rounded-lg block break-all">
              {metrics?.model_name || agentState.model_weights_path || "N/A"}
            </code>
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <div className="text-sm text-muted-foreground mb-1">Model Type</div>
              <code className="text-sm bg-muted/50 px-3 py-2 rounded-lg block">
                {metrics?.model_type || agentState.selected_model || "N/A"}
              </code>
            </div>
            <div>
              <div className="text-sm text-muted-foreground mb-1">Report Path</div>
              <code className="text-sm bg-muted/50 px-3 py-2 rounded-lg block truncate">
                {agentState.report_path || "N/A"}
              </code>
            </div>
          </div>
          {agentState.model_explanation && (
            <div>
              <div className="text-sm text-muted-foreground mb-1">Model Explanation</div>
              <p className="text-sm leading-relaxed bg-muted/30 px-3 py-2 rounded-lg">
                {agentState.model_explanation}
              </p>
            </div>
          )}
        </div>
      </Section>
    </div>
  )
}

function MetricsTab({ agentState }: { agentState: TrainingAgentState }) {
  const metrics = agentState.training_metrics
  const lastIteration = metrics?.iterations?.[metrics.iterations.length - 1]
  const testR2 = metrics?.test_r2 ?? lastIteration?.test_r2
  const testRmse = metrics?.test_rmse ?? lastIteration?.test_rmse
  const testMae = metrics?.test_mae ?? lastIteration?.test_mae
  const valR2 = metrics?.val_r2 ?? lastIteration?.val_r2
  const valRmse = metrics?.val_rmse ?? lastIteration?.val_rmse
  const valMae = metrics?.val_mae ?? lastIteration?.val_mae
  
  const hasClassificationMetrics = metrics?.test_accuracy != null || metrics?.test_roc_auc != null
  const hasRegressionMetrics = testR2 != null || valR2 != null

  return (
    <div className="space-y-8">
      {/* Validation vs Test comparison */}
      <div className="grid md:grid-cols-2 gap-6">
        <Section title="Validation Metrics">
          <div className="space-y-4">
            {hasClassificationMetrics ? (
              <>
                <MetricRow label="Accuracy" value={formatPercent(metrics?.val_accuracy)} />
                <MetricRow label="ROC-AUC" value={formatNumber(metrics?.val_roc_auc, 3)} />
              </>
            ) : hasRegressionMetrics ? (
              <>
                <MetricRow label="R² Score" value={formatNumber(valR2, 4)} />
                <MetricRow label="RMSE" value={formatNumber(valRmse, 2)} />
                <MetricRow label="MAE" value={formatNumber(valMae, 2)} />
              </>
            ) : (
              <p className="text-sm text-muted-foreground">No validation metrics available</p>
            )}
          </div>
        </Section>

        <Section title="Test Metrics">
          <div className="space-y-4">
            {hasClassificationMetrics ? (
              <>
                <MetricRow label="Accuracy" value={formatPercent(metrics?.test_accuracy)} highlight />
                <MetricRow label="ROC-AUC" value={formatNumber(metrics?.test_roc_auc, 3)} highlight />
              </>
            ) : hasRegressionMetrics ? (
              <>
                <MetricRow label="R² Score" value={formatNumber(testR2, 4)} highlight />
                <MetricRow label="RMSE" value={formatNumber(testRmse, 2)} highlight />
                <MetricRow label="MAE" value={formatNumber(testMae, 2)} highlight />
              </>
            ) : (
              <div className="space-y-2">
                <MetricRow label="Status" value={metrics?.success ? "Success" : "Completed"} highlight />
                <MetricRow label="Model" value={metrics?.model_name || "N/A"} />
              </div>
            )}
          </div>
        </Section>
      </div>

      {/* Iteration History */}
      <Section title="Training Iterations">
        <div className="space-y-2">
          {metrics?.iterations && metrics.iterations.length > 0 ? (
            metrics.iterations.map((iter, i) => (
              <div
                key={i}
                className={`flex items-center justify-between p-4 rounded-xl ${
                  i === (metrics.iterations?.length || 1) - 1
                    ? "bg-foreground/5 ring-1 ring-foreground/10"
                    : "bg-muted/30"
                }`}
              >
                <div className="flex items-center gap-3">
                  <span className="font-medium">Iteration {i + 1}</span>
                  {i === (metrics.iterations?.length || 1) - 1 && (
                    <Badge variant="secondary" className="text-xs">Best</Badge>
                  )}
                </div>
                <div className="flex gap-6 text-sm">
                  {hasClassificationMetrics ? (
                    <>
                      <span className="text-muted-foreground">
                        Accuracy: <span className="text-foreground font-medium">{formatPercent(iter.val_accuracy)}</span>
                      </span>
                      <span className="text-muted-foreground">
                        AUC: <span className="text-foreground font-medium">{formatNumber(iter.val_roc_auc, 3)}</span>
                      </span>
                    </>
                  ) : iter.val_r2 != null ? (
                    <>
                      <span className="text-muted-foreground">
                        R²: <span className="text-foreground font-medium">{formatNumber(iter.val_r2, 4)}</span>
                      </span>
                      <span className="text-muted-foreground">
                        RMSE: <span className="text-foreground font-medium">{formatNumber(iter.val_rmse, 2)}</span>
                      </span>
                    </>
                  ) : (
                    <span className="text-muted-foreground">Completed</span>
                  )}
                </div>
              </div>
            ))
          ) : (
            <div className="p-4 rounded-xl bg-foreground/5 ring-1 ring-foreground/10">
              <div className="flex items-center gap-3">
                <span className="font-medium">Iteration 1</span>
                <Badge variant="secondary" className="text-xs">Best</Badge>
              </div>
              <p className="text-sm text-muted-foreground mt-2">Training completed in 1 iteration</p>
            </div>
          )}
        </div>
      </Section>
    </div>
  )
}

function AnalysisTab({ agentState }: { agentState: TrainingAgentState }) {
  // Extract key_stats from analysis_trace
  const analysisTrace = agentState.analysis_trace || []
  const featureAnalysis = analysisTrace.find(t => t.step === "feature_selection_specification")
  const keyStats = (featureAnalysis?.key_stats || {}) as KeyStats
  
  const hasDatasetOverview = keyStats?.dataset_overview?.rows != null
  const hasNumericSummaries = keyStats?.numeric_summaries?.length > 0
  const hasCorrelations = keyStats?.feature_correlations?.length > 0
  const hasDistributionStats = keyStats?.distribution_stats?.length > 0
  const hasGroupSummaries = keyStats?.group_summaries?.length > 0
  const hasConcentrationAnalysis = keyStats?.concentration_analysis?.length > 0
  
  const hasAnyData = hasDatasetOverview || hasNumericSummaries || hasCorrelations || 
    hasDistributionStats || hasGroupSummaries || hasConcentrationAnalysis

  if (!hasAnyData) {
    return (
      <div className="space-y-8">
        <Section title="Feature Analysis">
          <p className="text-muted-foreground">No feature analysis data available. This data is generated during the feature selection step.</p>
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
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <MetricBox label="Rows" value={keyStats.dataset_overview.rows?.toLocaleString() || "N/A"} />
            <MetricBox label="Columns" value={String(keyStats.dataset_overview.columns || "N/A")} />
            <MetricBox label="Numeric Features" value={String(keyStats.dataset_overview.numeric_columns || "N/A")} />
            <MetricBox label="Categorical Features" value={String(keyStats.dataset_overview.categorical_columns || "N/A")} />
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
                    <td className="text-right py-2 px-3">{stat.mean?.toLocaleString() ?? 'N/A'}</td>
                    <td className="text-right py-2 px-3">{stat.std?.toLocaleString() ?? 'N/A'}</td>
                    <td className="text-right py-2 px-3">{stat.min?.toLocaleString() ?? 'N/A'}</td>
                    <td className="text-right py-2 px-3">{stat.max?.toLocaleString() ?? 'N/A'}</td>
                    <td className="text-right py-2 px-3">{stat.median?.toLocaleString() ?? 'N/A'}</td>
                    <td className={`text-right py-2 px-3 ${stat.skew && Math.abs(stat.skew) > 1 ? 'text-yellow-600 font-medium' : ''}`}>
                      {stat.skew?.toFixed(2) ?? 'N/A'}
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
            {keyStats.feature_correlations.slice(0, 15).map((corr: FeatureCorrelation, i: number) => (
              <CorrelationBar key={i} feature={corr.feature} correlation={corr.correlation} rank={i + 1} />
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
                  {stat.shape && (
                    <span className="text-xs text-muted-foreground">{stat.shape}</span>
                  )}
                </div>
                
                {/* Stats row */}
                <div className="grid grid-cols-2 md:grid-cols-6 gap-2 mb-4 text-xs">
                  <div><span className="text-muted-foreground">Mean:</span> <span className="font-medium">{stat.mean?.toLocaleString() ?? 'N/A'}</span></div>
                  <div><span className="text-muted-foreground">Std:</span> <span className="font-medium">{stat.std?.toLocaleString() ?? 'N/A'}</span></div>
                  <div><span className="text-muted-foreground">Min:</span> <span className="font-medium">{stat.min?.toLocaleString() ?? 'N/A'}</span></div>
                  <div><span className="text-muted-foreground">Max:</span> <span className="font-medium">{stat.max?.toLocaleString() ?? 'N/A'}</span></div>
                  <div><span className="text-muted-foreground">Skew:</span> <span className={`font-medium ${stat.skewness && Math.abs(stat.skewness) > 1 ? 'text-yellow-600' : ''}`}>{stat.skewness?.toFixed(2) ?? 'N/A'}</span></div>
                  <div><span className="text-muted-foreground">Kurtosis:</span> <span className={`font-medium ${stat.kurtosis && Math.abs(stat.kurtosis) > 3 ? 'text-yellow-600' : ''}`}>{stat.kurtosis?.toFixed(2) ?? 'N/A'}</span></div>
                </div>
                
                {/* Histogram */}
                {stat.histogram && stat.histogram.length > 0 && (
                  <div className="flex items-end gap-1 h-20">
                    {stat.histogram.map((bin: HistogramBin, j: number) => {
                      const maxPct = Math.max(...stat.histogram!.map(b => b.pct))
                      const height = (bin.pct / maxPct) * 100
                      return (
                        <div 
                          key={j} 
                          className="flex-1 bg-blue-500/60 rounded-t hover:bg-blue-500/80 transition-colors relative group"
                          style={{ height: `${height}%`, minHeight: '2px' }}
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
                    const maxMean = Math.max(...group.groups.map(x => x.mean || 0))
                    const width = maxMean > 0 ? ((g.mean || 0) / maxMean) * 100 : 0
                    return (
                      <div key={j} className="flex items-center gap-2">
                        <div className="w-24 text-xs truncate" title={String(g.value)}>{g.value}</div>
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
                        <line x1="0" y1="100" x2="100" y2="0" stroke="currentColor" strokeOpacity="0.2" strokeDasharray="2" />
                        {/* Lorenz curve */}
                        <polyline
                          fill="none"
                          stroke="rgb(59, 130, 246)"
                          strokeWidth="2"
                          points={`0,100 ${conc.lorenz_curve.map((p: LorenzPoint) => `${p.pct_entities},${100 - p.pct_value}`).join(' ')}`}
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

// Correlation bar component for visualization
function CorrelationBar({ feature, correlation, rank }: { feature: string; correlation: number; rank: number }) {
  const absCorr = Math.abs(correlation)
  const isPositive = correlation >= 0
  const width = Math.min(100, absCorr * 100)
  
  return (
    <div className="flex items-center gap-3 py-1.5">
      <div className="w-5 text-xs text-muted-foreground text-right">{rank}</div>
      <div className="w-28 font-mono text-xs truncate" title={feature}>{feature}</div>
      <div className="flex-1 flex items-center gap-2">
        <div className="flex-1 h-5 bg-muted/30 rounded-full overflow-hidden relative">
          {isPositive ? (
            <div
              className="absolute left-1/2 h-full bg-green-500/70 rounded-r-full transition-all"
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
        <span className={`w-14 text-xs font-mono text-right ${isPositive ? 'text-green-600' : 'text-red-600'}`}>
          {isPositive ? '+' : ''}{correlation.toFixed(3)}
        </span>
      </div>
    </div>
  )
}

function FeaturesTab({ agentState }: { agentState: TrainingAgentState }) {
  const features = agentState.feature_spec?.features || []

  return (
    <div className="space-y-8">
      {/* Features List */}
      <Section title={`Engineered Features (${features.length})`}>
        <div className="space-y-2">
          {features.length > 0 ? (
            features.map((feature, i) => (
              <div key={i} className="p-4 rounded-xl bg-muted/30">
                <div className="flex items-center gap-2 flex-wrap mb-2">
                  <span className="font-medium">{renderValue(feature.name)}</span>
                  <Badge variant="outline" className="text-xs font-normal">{renderValue(feature.encoding)}</Badge>
                </div>
                {feature.formula && (
                  <code className="text-xs text-muted-foreground font-mono block break-all">
                    {renderValue(feature.formula)}
                  </code>
                )}
              </div>
            ))
          ) : (
            <p className="text-sm text-muted-foreground">No features specified</p>
          )}
        </div>
      </Section>

      {/* Data Pipeline */}
      <Section title="Data Pipeline">
        <div className="space-y-1 rounded-xl bg-muted/30 overflow-hidden">
          <PipelineRow label="Raw Dataset" value={agentState.collected_dataset_ref} />
          <PipelineRow label="Cleaned Dataset" value={agentState.cleaned_dataset_ref} />
          <PipelineRow label="Train Set" value={agentState.transformed_train_ref} />
          <PipelineRow label="Validation Set" value={agentState.transformed_val_ref} />
          <PipelineRow label="Test Set" value={agentState.transformed_test_ref} last />
        </div>
      </Section>

      {/* Label Definition */}
      {agentState.label_definition && (
        <Section title="Label Definition">
          <div className="grid grid-cols-2 gap-4">
            <InfoBox label="Target Column" value={agentState.label_definition.target_column} />
            <InfoBox label="Split Strategy" value={agentState.label_definition.split_strategy} />
            <InfoBox label="Grain" value={agentState.label_definition.grain || "N/A"} />
            {agentState.label_definition.forbidden_columns && agentState.label_definition.forbidden_columns.length > 0 && (
              <div className="col-span-2">
                <div className="text-sm text-muted-foreground mb-1">Forbidden Columns</div>
                <div className="flex flex-wrap gap-2">
                  {agentState.label_definition.forbidden_columns.map((col, i) => (
                    <Badge key={i} variant="secondary" className="text-xs font-normal">{col}</Badge>
                  ))}
                </div>
              </div>
            )}
          </div>
        </Section>
      )}

      {/* Cleaning Transformations */}
      {agentState.cleaning_transformations && agentState.cleaning_transformations.length > 0 && (
        <Section title={`Cleaning Transformations (${agentState.cleaning_transformations.length})`}>
          <div className="space-y-1.5 max-h-[250px] overflow-y-auto">
            {agentState.cleaning_transformations.map((t, i) => {
              if (typeof t === "object" && t !== null) {
                const transform = t as Record<string, unknown>
                const toolName = transform.tool || transform.op || transform.operation || "transform"
                const args = transform.args as Record<string, unknown> | undefined
                const result = transform.result as string | undefined
                
                // Extract columns from args
                let columns = ""
                let extraInfo = ""
                if (args) {
                  if (args.columns) {
                    columns = Array.isArray(args.columns) ? (args.columns as string[]).join(", ") : String(args.columns)
                  } else if (args.column) {
                    columns = String(args.column)
                  }
                  if (args.value !== undefined) extraInfo = `= ${args.value}`
                  if (args.strategy) extraInfo = `(${args.strategy})`
                }
                
                // Extract result info
                let resultInfo = ""
                if (result) {
                  const match = String(result).match(/→\s*`([^`]+)`\s*(.*)/)
                  if (match) {
                    resultInfo = match[2] || ""
                  }
                }
                
                return (
                  <div key={i} className="flex items-start gap-3 py-2 px-3 bg-muted/30 rounded-lg">
                    <span className="text-xs font-mono bg-foreground/10 px-2 py-0.5 rounded font-medium shrink-0">
                      {String(toolName).replace(/_tool$/, "")}
                    </span>
                    <div className="flex-1 text-sm min-w-0">
                      {columns && <span className="font-medium">{columns}</span>}
                      {extraInfo && <span className="text-muted-foreground ml-2">{extraInfo}</span>}
                      {resultInfo && <span className="text-muted-foreground ml-2">{resultInfo}</span>}
                    </div>
                  </div>
                )
              }
              return (
                <code key={i} className="text-xs bg-muted/50 rounded-lg px-3 py-2 block font-mono">
                  {String(t)}
                </code>
              )
            })}
          </div>
        </Section>
      )}
    </div>
  )
}

function TraceTab({ steps, agentState }: { steps: StepInfo[]; agentState: TrainingAgentState }) {
  const [expanded, setExpanded] = useState<string | null>(null)
  const auditTrace = agentState.audit_trace || []
  const completedSteps = steps.filter((s) => s.status === "completed")

  const toggle = (id: string) => setExpanded(expanded === id ? null : id)

  const getAuditEntry = (stepId: string) => {
    const stepMapping: Record<string, string> = {
      "select_model": "select_model",
      "data_collection": "data_collection",
      "cleaning": "cleaning_and_standardization",
      "label_split_definition": "label_split_definition",
      "feature_selection_specification": "feature_selection_specification",
      "feature_engineering_executor": "feature_engineering_executor",
      "human_confirmation": "human_confirmation",
      "training": "training",
      "generate_report": "generate_report",
    }
    const auditStepName = stepMapping[stepId] || stepId
    return auditTrace.find((t) => t.step === auditStepName)
  }

  const renderStepContent = (stepId: string) => {
    const audit = getAuditEntry(stepId)

    if (stepId === "select_model") {
      return (
        <div className="space-y-3">
          <InfoBox label="Selected Model" value={agentState.selected_model} highlight />
          {agentState.model_explanation && (
            <div>
              <div className="text-sm text-muted-foreground mb-1">Explanation</div>
              <p className="text-sm leading-relaxed">{agentState.model_explanation}</p>
            </div>
          )}
          {audit && (
            <div className="grid grid-cols-2 gap-3">
              {audit.confidence != null && <InfoBox label="Confidence" value={String(audit.confidence)} />}
              {Array.isArray(audit.alternatives) && audit.alternatives.length > 0 && (
                <InfoBox label="Alternatives" value={(audit.alternatives as string[]).join(", ")} />
              )}
            </div>
          )}
        </div>
      )
    }

    if (stepId === "data_collection") {
      return (
        <div className="space-y-3">
          <InfoBox label="Dataset Reference" value={agentState.collected_dataset_ref} mono />
          {audit && (
            <div className="grid grid-cols-2 gap-3">
              {audit.rows != null && <InfoBox label="Rows" value={String(audit.rows)} />}
              {audit.source != null && <InfoBox label="Source" value={String(audit.source)} />}
            </div>
          )}
          {Array.isArray(audit?.columns) && audit.columns.length > 0 && (
            <div>
              <div className="text-sm text-muted-foreground mb-1">Columns ({(audit.columns as string[]).length})</div>
              <p className="text-sm">{(audit.columns as string[]).join(", ")}</p>
            </div>
          )}
        </div>
      )
    }

    if (stepId === "cleaning") {
      const transformations = agentState.cleaning_transformations || []
      const cleaningSummary = agentState.cleaning_summary
      
      // Helper to format a single transformation (new format with tool, args, result)
      const formatTransformation = (t: unknown, index: number) => {
        if (typeof t === "object" && t !== null) {
          const transform = t as Record<string, unknown>
          const toolName = transform.tool || transform.op || transform.operation || "transform"
          const args = transform.args as Record<string, unknown> | undefined
          const result = transform.result as string | undefined
          
          // Extract columns from args
          let columns = ""
          let extraInfo = ""
          if (args) {
            if (args.columns) {
              columns = Array.isArray(args.columns) ? (args.columns as string[]).join(", ") : String(args.columns)
            } else if (args.column) {
              columns = String(args.column)
            }
            if (args.value !== undefined) extraInfo = `= ${args.value}`
            if (args.strategy) extraInfo = `(${args.strategy})`
          }
          
          // Extract result info
          let resultInfo = ""
          if (result) {
            const match = String(result).match(/→\s*`([^`]+)`\s*(.*)/)
            if (match) {
              resultInfo = match[2] || ""
            }
          }
          
          return (
            <div key={index} className="flex items-start gap-3 py-2 px-3 bg-muted/30 rounded-lg">
              <span className="text-xs font-mono bg-foreground/10 px-2 py-0.5 rounded font-medium shrink-0">
                {String(toolName).replace(/_tool$/, "")}
              </span>
              <div className="flex-1 text-sm min-w-0">
                {columns && <span className="font-medium">{columns}</span>}
                {extraInfo && <span className="text-muted-foreground ml-2">{extraInfo}</span>}
                {resultInfo && <span className="text-muted-foreground ml-2">{resultInfo}</span>}
              </div>
            </div>
          )
        }
        return (
          <code key={index} className="text-xs bg-muted/50 rounded px-2 py-1 block font-mono">
            {String(t)}
          </code>
        )
      }
      
      return (
        <div className="space-y-4">
          {/* Cleaning Summary from Agent */}
          {cleaningSummary && (
            <div className="bg-muted/30 rounded-lg p-4">
              <div className="text-sm font-medium mb-2">Cleaning Summary</div>
              <p className="text-sm text-muted-foreground whitespace-pre-wrap">
                {cleaningSummary.replace(/^✅\s*CLEANING COMPLETE\n?/i, "")}
              </p>
            </div>
          )}
          
          <InfoBox label="Cleaned Dataset" value={agentState.cleaned_dataset_ref} mono />
          
          <div>
            <div className="text-sm font-medium mb-2">Transformations Applied ({transformations.length})</div>
            {transformations.length > 0 ? (
              <div className="space-y-1.5 max-h-[400px] overflow-y-auto">
                {transformations.map((t, i) => formatTransformation(t, i))}
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">No transformations needed - data was already clean</p>
            )}
          </div>
        </div>
      )
    }

    if (stepId === "label_split_definition") {
      const labelDef = agentState.label_definition
      return (
        <div className="space-y-3">
          {labelDef && (
            <div className="grid grid-cols-2 gap-3">
              <InfoBox label="Target Column" value={labelDef.target_column} />
              <InfoBox label="Split Strategy" value={labelDef.split_strategy} />
              <InfoBox label="Grain" value={labelDef.grain || "N/A"} />
            </div>
          )}
          {labelDef?.forbidden_columns && labelDef.forbidden_columns.length > 0 && (
            <div>
              <div className="text-sm text-muted-foreground mb-1">Forbidden Columns</div>
              <p className="text-sm">{labelDef.forbidden_columns.join(", ")}</p>
            </div>
          )}
          <div>
            <div className="text-sm text-muted-foreground mb-2">Dataset Splits</div>
            <div className="grid grid-cols-3 gap-2">
              <InfoBox label="Train" value={agentState.train_dataset_ref} mono small />
              <InfoBox label="Validation" value={agentState.val_dataset_ref} mono small />
              <InfoBox label="Test" value={agentState.test_dataset_ref} mono small />
            </div>
          </div>
        </div>
      )
    }

    if (stepId === "feature_selection_specification") {
      const features = agentState.feature_spec?.features || []
      const analysisTrace = agentState.analysis_trace || []
      const featureAnalysis = analysisTrace.find(t => t.step === "feature_selection_specification")
      const keyStats = (featureAnalysis?.key_stats || {}) as KeyStats
      
      return (
        <div className="space-y-4">
          {/* Summary */}
          {keyStats.summary_text && (
            <div className="bg-muted/30 rounded-lg p-3">
              <div className="text-sm">{keyStats.summary_text}</div>
            </div>
          )}
          
          {/* Quick Stats */}
          {keyStats.dataset_overview?.rows && (
            <div className="grid grid-cols-4 gap-2">
              <InfoBox label="Rows" value={keyStats.dataset_overview.rows.toLocaleString()} small />
              <InfoBox label="Columns" value={String(keyStats.dataset_overview.columns)} small />
              <InfoBox label="Numeric" value={String(keyStats.dataset_overview.numeric_columns)} small />
              <InfoBox label="Categorical" value={String(keyStats.dataset_overview.categorical_columns)} small />
            </div>
          )}
          
          {/* Top Correlations */}
          {keyStats.feature_correlations && keyStats.feature_correlations.length > 0 && (
            <div>
              <div className="text-sm text-muted-foreground mb-2">Top Correlations with Target</div>
              <div className="space-y-1">
                {keyStats.feature_correlations.slice(0, 5).map((corr, i) => (
                  <div key={i} className="flex justify-between text-sm bg-muted/30 rounded px-3 py-1.5">
                    <span className="font-mono text-xs">{corr.feature}</span>
                    <span className={`font-medium ${corr.correlation >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                      {corr.correlation >= 0 ? '+' : ''}{corr.correlation.toFixed(4)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
          
          {/* Leakage Warnings */}
          {keyStats.leakage_warnings && keyStats.leakage_warnings.length > 0 && (
            <div className="bg-destructive/10 rounded-lg p-3">
              <div className="text-sm font-medium text-destructive mb-1">⚠️ Leakage Warnings</div>
              <ul className="text-sm text-destructive/80 space-y-0.5">
                {keyStats.leakage_warnings.map((w, i) => (
                  <li key={i}>• {w}</li>
                ))}
              </ul>
            </div>
          )}
          
          {/* Features Specified */}
          <div>
            <div className="text-sm text-muted-foreground mb-2">Features Specified ({features.length})</div>
            <div className="space-y-1 max-h-[200px] overflow-y-auto">
              {features.map((f, i) => (
                <div key={i} className="flex items-center gap-2 text-sm bg-muted/30 rounded px-3 py-2">
                  <span className="font-medium">{renderValue(f.name)}</span>
                  <Badge variant="outline" className="text-xs font-normal">{renderValue(f.encoding)}</Badge>
                  {f.formula && <span className="text-muted-foreground text-xs font-mono truncate">{renderValue(f.formula)}</span>}
                </div>
              ))}
            </div>
          </div>
        </div>
      )
    }

    if (stepId === "feature_engineering_executor") {
      return (
        <div className="space-y-3">
          <InfoBox 
            label="Validation" 
            value={agentState.feature_validation_passed ? "Passed" : "Issues Found"} 
            highlight={agentState.feature_validation_passed}
          />
          {audit?.features_created != null && Array.isArray(audit.features_created) ? (
            <div>
              <div className="text-sm text-muted-foreground mb-1">Features Created ({(audit.features_created as string[]).length})</div>
              <p className="text-sm">{(audit.features_created as string[]).join(", ")}</p>
            </div>
          ) : null}
          {audit?.shapes != null && typeof audit.shapes === "object" ? (
            <div>
              <div className="text-sm text-muted-foreground mb-2">Dataset Shapes</div>
              <div className="grid grid-cols-3 gap-2">
                {Object.entries(audit.shapes as Record<string, number[]>).map(([key, shape]) => (
                  <InfoBox key={key} label={key} value={Array.isArray(shape) ? `${shape[0]} × ${shape[1]}` : String(shape)} small />
                ))}
              </div>
            </div>
          ) : null}
          {audit?.errors != null && Array.isArray(audit.errors) && (audit.errors as string[]).length > 0 ? (
            <div className="bg-destructive/10 rounded-lg p-3">
              <div className="text-sm font-medium text-destructive mb-1">Errors</div>
              <p className="text-sm text-destructive/80">{(audit.errors as string[]).join(", ")}</p>
            </div>
          ) : null}
          <div>
            <div className="text-sm text-muted-foreground mb-2">Transformed Datasets</div>
            <div className="grid grid-cols-3 gap-2">
              <InfoBox label="Train" value={agentState.transformed_train_ref} mono small />
              <InfoBox label="Validation" value={agentState.transformed_val_ref} mono small />
              <InfoBox label="Test" value={agentState.transformed_test_ref} mono small />
            </div>
          </div>
        </div>
      )
    }

    if (stepId === "human_confirmation") {
      return (
        <div className="space-y-3">
          <InfoBox 
            label="Status" 
            value={agentState.human_confirmed ? "Confirmed" : "Pending"} 
            highlight={agentState.human_confirmed}
          />
          {audit?.mode != null && <InfoBox label="Mode" value={String(audit.mode)} />}
        </div>
      )
    }

    if (stepId === "training") {
      const metrics = agentState.training_metrics
      const hasClassification = metrics?.test_accuracy != null || metrics?.test_roc_auc != null
      const hasRegression = metrics?.test_r2 != null || metrics?.val_r2 != null || metrics?.iterations?.[0]?.test_r2 != null

      return (
        <div className="space-y-3">
          <div className="grid grid-cols-2 gap-3">
            <InfoBox label="Status" value={metrics?.success ? "Success" : "Failed"} highlight={metrics?.success} />
            <InfoBox label="Iterations" value={String(metrics?.num_iterations || 1)} />
          </div>
          <InfoBox label="Model" value={metrics?.model_name || agentState.model_weights_path} mono />

          {hasClassification && (
            <div>
              <div className="text-sm text-muted-foreground mb-2">Classification Metrics</div>
              <div className="grid grid-cols-2 gap-2 text-sm">
                {metrics?.val_accuracy != null && <span>Val Accuracy: <strong>{(metrics.val_accuracy * 100).toFixed(2)}%</strong></span>}
                {metrics?.test_accuracy != null && <span>Test Accuracy: <strong>{(metrics.test_accuracy * 100).toFixed(2)}%</strong></span>}
                {metrics?.val_roc_auc != null && <span>Val ROC-AUC: <strong>{metrics.val_roc_auc.toFixed(4)}</strong></span>}
                {metrics?.test_roc_auc != null && <span>Test ROC-AUC: <strong>{metrics.test_roc_auc.toFixed(4)}</strong></span>}
              </div>
            </div>
          )}

          {hasRegression && (
            <div>
              <div className="text-sm text-muted-foreground mb-2">Regression Metrics</div>
              <div className="grid grid-cols-2 gap-2 text-sm">
                {(metrics?.val_r2 ?? metrics?.iterations?.[0]?.val_r2) != null && (
                  <span>Val R²: <strong>{(metrics?.val_r2 ?? metrics?.iterations?.[0]?.val_r2)?.toFixed(4)}</strong></span>
                )}
                {(metrics?.test_r2 ?? metrics?.iterations?.[0]?.test_r2) != null && (
                  <span>Test R²: <strong>{(metrics?.test_r2 ?? metrics?.iterations?.[0]?.test_r2)?.toFixed(4)}</strong></span>
                )}
                {(metrics?.val_rmse ?? metrics?.iterations?.[0]?.val_rmse) != null && (
                  <span>Val RMSE: <strong>{(metrics?.val_rmse ?? metrics?.iterations?.[0]?.val_rmse)?.toFixed(2)}</strong></span>
                )}
                {(metrics?.test_rmse ?? metrics?.iterations?.[0]?.test_rmse) != null && (
                  <span>Test RMSE: <strong>{(metrics?.test_rmse ?? metrics?.iterations?.[0]?.test_rmse)?.toFixed(2)}</strong></span>
                )}
                {(metrics?.test_mae ?? metrics?.iterations?.[0]?.test_mae) != null && (
                  <span>Test MAE: <strong>{(metrics?.test_mae ?? metrics?.iterations?.[0]?.test_mae)?.toFixed(2)}</strong></span>
                )}
              </div>
            </div>
          )}

          {metrics?.summary && (
            <div className="bg-muted/30 rounded-lg p-3">
              <div className="text-sm font-medium mb-1">Summary</div>
              <p className="text-sm">{metrics.summary}</p>
            </div>
          )}

          {metrics?.recommendations && (
            <div className="bg-muted/30 rounded-lg p-3">
              <div className="text-sm font-medium mb-1">Recommendations</div>
              <p className="text-sm">{metrics.recommendations}</p>
            </div>
          )}
        </div>
      )
    }

    if (stepId === "generate_report") {
      return (
        <div className="space-y-3">
          <InfoBox label="Report Path" value={agentState.report_path} mono />
          <InfoBox label="Model Weights" value={agentState.model_weights_path} mono />
          <InfoBox label="Audit Entries" value={String(auditTrace.length)} />
        </div>
      )
    }

    if (audit) {
      return (
        <pre className="text-xs bg-muted/30 rounded-lg p-3 overflow-x-auto">
          {JSON.stringify(audit, null, 2)}
        </pre>
      )
    }

    return <p className="text-sm text-muted-foreground">No details available</p>
  }

  return (
    <div className="space-y-8">
      <Section title="Execution Trace">
        <div className="space-y-1 rounded-xl overflow-hidden">
          {completedSteps.map((step) => (
            <div key={step.id}>
              <button
                onClick={() => toggle(step.id)}
                className="w-full flex items-center justify-between p-4 hover:bg-muted/30 transition-colors text-left"
              >
                <div className="flex items-center gap-3">
                  <CheckCircle2 className="h-4 w-4 text-green-600" />
                  <span className="font-medium">{step.name}</span>
                </div>
                <div className="flex items-center gap-3">
                  {step.endTime && step.startTime && (
                    <span className="text-xs text-muted-foreground">
                      {((step.endTime - step.startTime) / 1000).toFixed(1)}s
                    </span>
                  )}
                  {expanded === step.id ? (
                    <ChevronDown className="h-4 w-4 text-muted-foreground" />
                  ) : (
                    <ChevronRight className="h-4 w-4 text-muted-foreground" />
                  )}
                </div>
              </button>
              {expanded === step.id && (
                <div className="px-4 pb-4">
                  <div className="bg-muted/30 rounded-xl p-4">
                    {renderStepContent(step.id)}
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      </Section>

      <Section title="Raw Audit Trace">
        <pre className="text-xs bg-muted/30 rounded-xl p-4 overflow-x-auto max-h-[400px] overflow-y-auto">
          {JSON.stringify(agentState.audit_trace, null, 2)}
        </pre>
      </Section>
    </div>
  )
}

// Utility components
function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <h3 className="text-sm font-medium mb-4">{title}</h3>
      {children}
    </div>
  )
}

function MetricBox({ label, value, highlight }: { label: string; value: string; highlight?: boolean }) {
  return (
    <div className={`rounded-xl p-4 ${highlight ? "bg-foreground/5" : "bg-muted/30"}`}>
      <div className="text-xs text-muted-foreground mb-1">{label}</div>
      <div className="text-xl font-semibold">{value}</div>
    </div>
  )
}

function MetricRow({ label, value, highlight }: { label: string; value: string; highlight?: boolean }) {
  return (
    <div className="flex justify-between items-center">
      <span className="text-muted-foreground">{label}</span>
      <span className={`text-xl font-semibold ${highlight ? "text-green-600" : ""}`}>{value}</span>
    </div>
  )
}

function InfoRow({ label, value }: { label: string; value?: string | null }) {
  return (
    <div className="flex justify-between items-start gap-4 py-2 border-b border-border/50 last:border-0">
      <span className="text-muted-foreground shrink-0">{label}</span>
      <span className="text-right">{value || "N/A"}</span>
    </div>
  )
}

function InfoBox({ label, value, mono, highlight, small }: { label: string; value?: string | null; mono?: boolean; highlight?: boolean; small?: boolean }) {
  return (
    <div className={`bg-muted/50 rounded-lg ${small ? "px-2 py-1.5" : "px-3 py-2"}`}>
      <div className={`text-muted-foreground ${small ? "text-xs" : "text-sm"} mb-0.5`}>{label}</div>
      <div className={`${mono ? "font-mono text-xs" : small ? "text-sm" : ""} ${highlight ? "text-green-600 font-medium" : ""} break-all`}>
        {value || "N/A"}
      </div>
    </div>
  )
}

function PipelineRow({ label, value, last }: { label: string; value?: string | null; last?: boolean }) {
  return (
    <div className={`flex justify-between items-center px-4 py-3 ${!last ? "border-b border-border/30" : ""}`}>
      <span className="text-muted-foreground">{label}</span>
      <code className="text-xs font-mono truncate max-w-[60%]">{value || "N/A"}</code>
    </div>
  )
}

function renderValue(value: unknown): string {
  if (value === null || value === undefined) return "N/A"
  if (typeof value === "string") return value
  if (typeof value === "number" || typeof value === "boolean") return String(value)
  if (typeof value === "object") {
    try { return JSON.stringify(value) } catch { return String(value) }
  }
  return String(value)
}

function generateTextReport(state: TrainingAgentState, steps: StepInfo[]): string {
  const metrics = state.training_metrics
  const completedSteps = steps.filter(s => s.status === "completed").length
  return `
TRAINING REPORT
===============

Goal: ${state.goal}
Model: ${metrics?.model_name} (${metrics?.model_type})
Report Path: ${state.report_path}

METRICS
-------
Test Accuracy: ${formatPercent(metrics?.test_accuracy)}
Test ROC-AUC: ${formatNumber(metrics?.test_roc_auc, 3)}
Validation Accuracy: ${formatPercent(metrics?.val_accuracy)}
Validation ROC-AUC: ${formatNumber(metrics?.val_roc_auc, 3)}

PIPELINE
--------
Steps Completed: ${completedSteps}/${steps.length}

DATA
----
Dataset: ${state.collected_dataset_ref}
Target Column: ${state.label_definition?.target_column}
Split Strategy: ${state.label_definition?.split_strategy}

FEATURES (${state.feature_spec?.features.length || 0})
${state.feature_spec?.features.map((f) => `- ${f.name}: ${f.formula}`).join("\n") || "None"}

TRAINING ITERATIONS
-------------------
${metrics?.iterations?.map((it) => `Iteration ${it.iteration}: Accuracy=${formatPercent(it.val_accuracy)}, AUC=${formatNumber(it.val_roc_auc, 3)}`).join("\n") || "None"}
Best Iteration: ${metrics?.best_iteration}

Generated: ${new Date().toISOString()}
`.trim()
}

function generateJsonReport(state: TrainingAgentState, steps: StepInfo[]): object {
  return {
    generated_at: new Date().toISOString(),
    goal: state.goal,
    model: {
      name: state.training_metrics?.model_name,
      type: state.training_metrics?.model_type,
      explanation: state.model_explanation,
    },
    metrics: {
      test: {
        accuracy: state.training_metrics?.test_accuracy,
        roc_auc: state.training_metrics?.test_roc_auc,
        r2: state.training_metrics?.test_r2,
        rmse: state.training_metrics?.test_rmse,
        mae: state.training_metrics?.test_mae,
      },
      validation: {
        accuracy: state.training_metrics?.val_accuracy,
        roc_auc: state.training_metrics?.val_roc_auc,
        r2: state.training_metrics?.val_r2,
        rmse: state.training_metrics?.val_rmse,
        mae: state.training_metrics?.val_mae,
      },
      iterations: state.training_metrics?.iterations,
      best_iteration: state.training_metrics?.best_iteration,
      summary: state.training_metrics?.summary,
      recommendations: state.training_metrics?.recommendations,
    },
    data: {
      collected_dataset: state.collected_dataset_ref,
      cleaned_dataset: state.cleaned_dataset_ref,
      train_dataset: state.transformed_train_ref,
      val_dataset: state.transformed_val_ref,
      test_dataset: state.transformed_test_ref,
    },
    label_definition: state.label_definition,
    feature_spec: state.feature_spec,
    cleaning_transformations: state.cleaning_transformations,
    audit_trace: state.audit_trace,
    steps: steps.map((s) => ({
      id: s.id,
      name: s.name,
      status: s.status,
      duration_ms: s.endTime && s.startTime ? s.endTime - s.startTime : null,
    })),
  }
}
