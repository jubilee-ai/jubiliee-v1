import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import type { TrainingAgentState, StepInfo, KeyStats, FeatureCorrelation } from "@/types/agent"
import { X, CheckCircle2, AlertTriangle } from "lucide-react"
import { formatNumber, formatPercent } from "@/lib/utils"

interface StepDetailModalProps {
  stepId: string
  agentState: TrainingAgentState
  steps: StepInfo[]
  onClose: () => void
}

export function StepDetailModal({ stepId, agentState, steps, onClose }: StepDetailModalProps) {
  const step = steps.find(s => s.id === stepId)
  const auditTrace = agentState.audit_trace || []
  
  const stepMapping: Record<string, string> = {
    "select_model": "select_model",
    "data_collection": "data_collection",
    "cleaning": "cleaning_and_standardization",
    "label_split_definition": "label_split_definition",
    "feature_selection_specification": "feature_selection_specification",
    "feature_specification_and_engineering": "feature_engineering_executor",
    "feature_engineering_executor": "feature_engineering_executor",
    "training_approval": "training_approval",
    "training": "training",
    "generate_report": "generate_report",
  }
  
  const auditStepName = stepMapping[stepId] || stepId
  const audit = auditTrace.find((t) => t.step === auditStepName)

  const getStepTitle = () => {
    switch (stepId) {
      case "select_model": return "Model Selection"
      case "data_collection": return "Data Collection"
      case "cleaning": return "Data Cleaning & Standardization"
      case "label_split_definition": return "Label & Split Definition"
      case "feature_selection_specification": return "Feature Selection"
      case "feature_specification_and_engineering": return "Features (spec + build)"
      case "feature_engineering_executor": return "Feature Engineering"
      case "training_approval": return "Training Configuration"
      case "training": return "Model Training"
      case "generate_report": return "Report Generation"
      default: return step?.name || stepId
    }
  }

  const renderContent = () => {
    switch (stepId) {
      case "select_model":
        return <ModelSelectionDetail agentState={agentState} audit={audit} />
      case "data_collection":
        return <DataCollectionDetail agentState={agentState} audit={audit} />
      case "cleaning":
        return <CleaningDetail agentState={agentState} audit={audit} />
      case "label_split_definition":
        return <LabelSplitDetail agentState={agentState} audit={audit} />
      case "feature_selection_specification":
        return <FeatureSelectionDetail agentState={agentState} />
      case "feature_specification_and_engineering":
        return (
          <>
            <FeatureSelectionDetail agentState={agentState} />
            <FeatureEngineeringDetail agentState={agentState} audit={audit} />
          </>
        )
      case "feature_engineering_executor":
        return <FeatureEngineeringDetail agentState={agentState} audit={audit} />
      case "training_approval":
        return <TrainingConfigDetail agentState={agentState} audit={audit} />
      case "training":
        return <TrainingDetail agentState={agentState} />
      case "generate_report":
        return <ReportDetail agentState={agentState} />
      default:
        return (
          <div className="text-sm text-muted-foreground">
            {audit ? (
              <pre className="bg-muted/30 rounded-lg p-3 overflow-x-auto text-xs">
                {JSON.stringify(audit, null, 2)}
              </pre>
            ) : (
              "No details available for this step."
            )}
          </div>
        )
    }
  }

  return (
    <div className="fixed inset-0 z-50 bg-background/80 backdrop-blur-sm flex items-center justify-center p-4" onClick={onClose}>
      <div 
        className="w-full max-w-2xl bg-background rounded-2xl border shadow-2xl flex flex-col"
        style={{ maxHeight: "80vh" }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex-shrink-0 flex items-center justify-between px-6 py-4 border-b">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-full bg-success/10 dark:bg-success/15 flex items-center justify-center">
              <CheckCircle2 className="h-4 w-4 text-success" />
            </div>
            <div>
              <h2 className="text-lg font-semibold tracking-tight">{getStepTitle()}</h2>
              {step?.endTime && step?.startTime && (
                <p className="text-xs text-muted-foreground">
                  Completed in {((step.endTime - step.startTime) / 1000).toFixed(1)}s
                </p>
              )}
            </div>
          </div>
          <Button variant="ghost" size="sm" onClick={onClose} className="h-8 w-8 p-0">
            <X className="h-4 w-4" />
          </Button>
        </div>

        {/* Content - scrollable */}
        <div className="flex-1 overflow-y-auto p-6 min-h-0">
          {renderContent()}
        </div>

        {/* Footer with action hint */}
        <div className="flex-shrink-0 px-6 py-4 border-t bg-muted/30">
          <p className="text-xs text-muted-foreground text-center">
            Review the details above to decide if you should proceed or request changes.
          </p>
        </div>
      </div>
    </div>
  )
}

// Model Selection Detail
function ModelSelectionDetail({ agentState, audit }: { agentState: TrainingAgentState; audit?: Record<string, unknown> }) {
  return (
    <div className="space-y-6">
      {/* Selected Model */}
      <div className="bg-foreground/5 rounded-xl p-4">
        <div className="text-xs text-muted-foreground mb-1">Selected Model</div>
        <div className="text-2xl font-semibold">{agentState.selected_model || "N/A"}</div>
      </div>

      {/* Explanation */}
      {agentState.model_explanation && (
        <div>
          <div className="text-sm font-medium mb-2">Why this model?</div>
          <p className="text-sm leading-relaxed text-muted-foreground bg-muted/30 rounded-lg p-4">
            {agentState.model_explanation}
          </p>
        </div>
      )}

      {/* Alternatives if available */}
      {Array.isArray(audit?.alternatives) && (audit.alternatives as string[]).length > 0 && (
        <div>
          <div className="text-sm font-medium mb-2">Alternative Models Considered</div>
          <div className="flex flex-wrap gap-2">
            {(audit.alternatives as string[]).map((alt, i) => (
              <Badge key={i} variant="outline" className="text-sm">{alt}</Badge>
            ))}
          </div>
        </div>
      )}

      {/* Key Decision Factors */}
      <div>
        <div className="text-sm font-medium mb-2">Key Decision Factors</div>
        <ul className="space-y-2 text-sm text-muted-foreground">
          <li className="flex gap-2">
            <span className="text-success">✓</span>
            Matches the task type (classification/regression)
          </li>
          <li className="flex gap-2">
            <span className="text-success">✓</span>
            Interpretability requirements considered
          </li>
          <li className="flex gap-2">
            <span className="text-success">✓</span>
            Dataset size and feature characteristics evaluated
          </li>
        </ul>
      </div>
    </div>
  )
}

// Data Collection Detail
function DataCollectionDetail({ agentState, audit }: { agentState: TrainingAgentState; audit?: Record<string, unknown> }) {
  return (
    <div className="space-y-6">
      {/* Dataset Reference */}
      <div className="bg-foreground/5 rounded-xl p-4">
        <div className="text-xs text-muted-foreground mb-1">Dataset Loaded</div>
        <code className="text-sm font-mono">{agentState.collected_dataset_ref || "N/A"}</code>
      </div>

      {/* Stats */}
      {audit && (
        <div className="grid grid-cols-2 gap-4">
          {audit.rows != null && (
            <div className="bg-muted/30 rounded-lg p-3">
              <div className="text-xs text-muted-foreground">Rows</div>
              <div className="text-xl font-semibold">{(audit.rows as number).toLocaleString()}</div>
            </div>
          )}
          {Array.isArray(audit.columns) && (
            <div className="bg-muted/30 rounded-lg p-3">
              <div className="text-xs text-muted-foreground">Columns</div>
              <div className="text-xl font-semibold">{(audit.columns as string[]).length}</div>
            </div>
          )}
        </div>
      )}

      {/* Columns List */}
      {Array.isArray(audit?.columns) && (audit.columns as string[]).length > 0 && (
        <div>
          <div className="text-sm font-medium mb-2">Available Columns ({(audit.columns as string[]).length})</div>
          <div className="flex flex-wrap gap-1.5 max-h-[150px] overflow-y-auto">
            {(audit.columns as string[]).map((col, i) => (
              <Badge key={i} variant="secondary" className="text-xs font-mono font-normal">{col}</Badge>
            ))}
          </div>
        </div>
      )}

      {/* What to check */}
      <div className="bg-blue-50 dark:bg-blue-950/30 rounded-lg p-4">
        <div className="text-sm font-medium text-blue-700 dark:text-blue-300 mb-2">What to Review</div>
        <ul className="text-sm text-blue-600 dark:text-blue-400 space-y-1">
          <li>• Is this the correct dataset for your goal?</li>
          <li>• Are all expected columns present?</li>
          <li>• Does the row count match expectations?</li>
        </ul>
      </div>
    </div>
  )
}

// Cleaning Detail
function CleaningDetail({ agentState }: { agentState: TrainingAgentState; audit?: Record<string, unknown> }) {
  const transformations = agentState.cleaning_transformations || []
  
  return (
    <div className="space-y-6">
      {/* Summary */}
      {agentState.cleaning_summary && (
        <div className="bg-muted/30 rounded-lg p-4">
          <div className="text-sm font-medium mb-2">Cleaning Summary</div>
          <p className="text-sm text-muted-foreground whitespace-pre-wrap">
            {agentState.cleaning_summary.replace(/^✅\s*CLEANING COMPLETE\n?/i, "")}
          </p>
        </div>
      )}

      {/* Cleaned Dataset */}
      <div className="bg-foreground/5 rounded-xl p-4">
        <div className="text-xs text-muted-foreground mb-1">Cleaned Dataset</div>
        <code className="text-sm font-mono">{agentState.cleaned_dataset_ref || "N/A"}</code>
      </div>

      {/* Transformations */}
      <div>
        <div className="text-sm font-medium mb-2">Transformations Applied ({transformations.length})</div>
        {transformations.length > 0 ? (
          <div className="space-y-1.5 max-h-[250px] overflow-y-auto">
            {transformations.map((t, i) => {
              const transform = t as Record<string, unknown>
              const toolName = transform.tool || transform.op || "transform"
              const args = transform.args as Record<string, unknown> | undefined
              
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
              
              return (
                <div key={i} className="flex items-start gap-3 py-2 px-3 bg-muted/30 rounded-lg">
                  <span className="text-xs font-mono bg-foreground/10 px-2 py-0.5 rounded font-medium shrink-0">
                    {String(toolName).replace(/_tool$/, "")}
                  </span>
                  <div className="flex-1 text-sm min-w-0">
                    {columns && <span className="font-medium">{columns}</span>}
                    {extraInfo && <span className="text-muted-foreground ml-2">{extraInfo}</span>}
                  </div>
                </div>
              )
            })}
          </div>
        ) : (
          <p className="text-sm text-muted-foreground">No transformations needed - data was already clean</p>
        )}
      </div>

      {/* What to check */}
      <div className="bg-blue-50 dark:bg-blue-950/30 rounded-lg p-4">
        <div className="text-sm font-medium text-blue-700 dark:text-blue-300 mb-2">What to Review</div>
        <ul className="text-sm text-blue-600 dark:text-blue-400 space-y-1">
          <li>• Are the cleaning operations appropriate for your data?</li>
          <li>• Were any important values incorrectly treated as missing?</li>
          <li>• Do the transformations preserve data integrity?</li>
        </ul>
      </div>
    </div>
  )
}

// Label & Split Detail
function LabelSplitDetail({ agentState }: { agentState: TrainingAgentState; audit?: Record<string, unknown> }) {
  const labelDef = agentState.label_definition
  
  return (
    <div className="space-y-6">
      {/* Target Column */}
      <div className="bg-foreground/5 rounded-xl p-4">
        <div className="text-xs text-muted-foreground mb-1">Target Column</div>
        <div className="text-2xl font-semibold">{labelDef?.target_column || "N/A"}</div>
      </div>

      {/* Split Configuration */}
      <div className="grid grid-cols-2 gap-4">
        <div className="bg-muted/30 rounded-lg p-3">
          <div className="text-xs text-muted-foreground">Split Strategy</div>
          <div className="text-lg font-semibold capitalize">{labelDef?.split_strategy || "random"}</div>
        </div>
        <div className="bg-muted/30 rounded-lg p-3">
          <div className="text-xs text-muted-foreground">Grain</div>
          <div className="text-lg font-semibold">{labelDef?.grain || "row"}</div>
        </div>
      </div>

      {/* Split Breakdown */}
      <div>
        <div className="text-sm font-medium mb-2">Data Split (70/15/15)</div>
        <div className="space-y-2">
          <div className="flex items-center gap-3">
            <div className="w-20 text-sm text-muted-foreground">Train</div>
            <div className="flex-1 h-6 bg-muted/30 rounded-full overflow-hidden">
              <div className="h-full bg-blue-500/70 rounded-full" style={{ width: "70%" }} />
            </div>
            <span className="text-sm font-medium w-12 text-right">70%</span>
          </div>
          <div className="flex items-center gap-3">
            <div className="w-20 text-sm text-muted-foreground">Validation</div>
            <div className="flex-1 h-6 bg-muted/30 rounded-full overflow-hidden">
              <div className="h-full bg-success/65 rounded-full" style={{ width: "15%" }} />
            </div>
            <span className="text-sm font-medium w-12 text-right">15%</span>
          </div>
          <div className="flex items-center gap-3">
            <div className="w-20 text-sm text-muted-foreground">Test</div>
            <div className="flex-1 h-6 bg-muted/30 rounded-full overflow-hidden">
              <div className="h-full bg-primary/45 rounded-full" style={{ width: "15%" }} />
            </div>
            <span className="text-sm font-medium w-12 text-right">15%</span>
          </div>
        </div>
      </div>

      {/* Forbidden Columns */}
      {labelDef?.forbidden_columns && labelDef.forbidden_columns.length > 0 && (
        <div>
          <div className="text-sm font-medium mb-2">Excluded from Training</div>
          <div className="flex flex-wrap gap-1.5">
            {labelDef.forbidden_columns.map((col, i) => (
              <Badge key={i} variant="destructive" className="text-xs font-mono font-normal">{col}</Badge>
            ))}
          </div>
        </div>
      )}

      {/* What to check */}
      <div className="bg-blue-50 dark:bg-blue-950/30 rounded-lg p-4">
        <div className="text-sm font-medium text-blue-700 dark:text-blue-300 mb-2">What to Review</div>
        <ul className="text-sm text-blue-600 dark:text-blue-400 space-y-1">
          <li>• Is the target column correctly identified?</li>
          <li>• Is the split strategy appropriate for your data?</li>
          <li>• Are there any columns that should be excluded?</li>
        </ul>
      </div>
    </div>
  )
}

// Feature Selection Detail
function FeatureSelectionDetail({ agentState }: { agentState: TrainingAgentState }) {
  const [showAllCorrelations, setShowAllCorrelations] = useState(false)
  const [showAllFeatures, setShowAllFeatures] = useState(false)
  
  const features = agentState.feature_spec?.features || []
  const analysisTrace = agentState.analysis_trace || []
  const featureAnalysis = analysisTrace.find(t => t.step === "feature_selection_specification")
  const keyStats = (featureAnalysis?.key_stats || {}) as KeyStats
  
  const correlations = keyStats.feature_correlations || []
  const displayedCorrelations = showAllCorrelations ? correlations : correlations.slice(0, 10)
  const displayedFeatures = showAllFeatures ? features : features.slice(0, 10)
  
  return (
    <div className="space-y-6">
      {/* Summary */}
      {keyStats.summary_text && (
        <div className="bg-muted/30 rounded-lg p-4">
          <p className="text-sm">{keyStats.summary_text}</p>
        </div>
      )}

      {/* Dataset Stats */}
      {keyStats.dataset_overview?.rows && (
        <div className="grid grid-cols-4 gap-3">
          <div className="bg-muted/30 rounded-lg p-3 text-center">
            <div className="text-xs text-muted-foreground">Rows</div>
            <div className="text-lg font-semibold">{keyStats.dataset_overview.rows.toLocaleString()}</div>
          </div>
          <div className="bg-muted/30 rounded-lg p-3 text-center">
            <div className="text-xs text-muted-foreground">Columns</div>
            <div className="text-lg font-semibold">{keyStats.dataset_overview.columns}</div>
          </div>
          <div className="bg-muted/30 rounded-lg p-3 text-center">
            <div className="text-xs text-muted-foreground">Numeric</div>
            <div className="text-lg font-semibold">{keyStats.dataset_overview.numeric_columns}</div>
          </div>
          <div className="bg-muted/30 rounded-lg p-3 text-center">
            <div className="text-xs text-muted-foreground">Categorical</div>
            <div className="text-lg font-semibold">{keyStats.dataset_overview.categorical_columns}</div>
          </div>
        </div>
      )}

      {/* Target Analysis */}
      {keyStats.target_analysis && (
        <div>
          <div className="text-sm font-medium mb-2">Target Variable Analysis</div>
          <div className="bg-muted/30 rounded-lg p-4">
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="text-muted-foreground">Column:</span>{" "}
                <span className="font-medium">{keyStats.target_analysis.column || "N/A"}</span>
              </div>
              <div>
                <span className="text-muted-foreground">Type:</span>{" "}
                <span className="font-medium capitalize">{keyStats.target_analysis.type || "N/A"}</span>
              </div>
              {keyStats.target_analysis.type === "numeric" ? (
                <>
                  <div>
                    <span className="text-muted-foreground">Mean:</span>{" "}
                    <span className="font-medium">{keyStats.target_analysis.mean?.toFixed(4) ?? "N/A"}</span>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Std:</span>{" "}
                    <span className="font-medium">{keyStats.target_analysis.std?.toFixed(4) ?? "N/A"}</span>
                  </div>
                </>
              ) : (
                <>
                  <div>
                    <span className="text-muted-foreground">Unique Values:</span>{" "}
                    <span className="font-medium">{keyStats.target_analysis.unique_values ?? "N/A"}</span>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Top Value:</span>{" "}
                    <span className="font-medium">{keyStats.target_analysis.top_value ?? "N/A"}</span>
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
      )}

      {/* All Correlations with Target */}
      {correlations.length > 0 && (
        <div>
          <div className="text-sm font-medium mb-2">Feature Correlations with Target ({correlations.length})</div>
          <div className="space-y-1">
            {displayedCorrelations.map((corr: FeatureCorrelation, i: number) => {
              const corrValue = typeof corr.correlation === 'number' ? corr.correlation : 0
              const absCorr = Math.abs(corrValue)
              const isPositive = corrValue >= 0
              const featureName = renderValue(corr.feature)
              return (
                <div key={i} className="flex items-center gap-3 py-1.5">
                  <div className="w-5 text-xs text-muted-foreground text-right">{i + 1}</div>
                  <div className="w-36 font-mono text-xs truncate" title={featureName}>{featureName}</div>
                  <div className="flex-1 h-4 bg-muted/30 rounded-full overflow-hidden relative">
                    {isPositive ? (
                      <div
                        className="absolute left-1/2 h-full bg-success/65 rounded-r-full"
                        style={{ width: `${(absCorr * 50)}%` }}
                      />
                    ) : (
                      <div
                        className="absolute right-1/2 h-full bg-red-500/70 rounded-l-full"
                        style={{ width: `${(absCorr * 50)}%` }}
                      />
                    )}
                    <div className="absolute left-1/2 top-0 bottom-0 w-px bg-border" />
                  </div>
                  <span className={`w-14 text-xs font-mono text-right ${isPositive ? 'text-success' : 'text-destructive'}`}>
                    {isPositive ? '+' : ''}{corrValue.toFixed(3)}
                  </span>
                </div>
              )
            })}
          </div>
          {correlations.length > 10 && (
            <button
              onClick={() => setShowAllCorrelations(!showAllCorrelations)}
              className="mt-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
            >
              {showAllCorrelations ? "Show less" : `Show all ${correlations.length} correlations`}
            </button>
          )}
        </div>
      )}

      {/* High Correlation Pairs (multicollinearity) */}
      {keyStats.high_correlation_pairs && keyStats.high_correlation_pairs.length > 0 && (
        <div>
          <div className="text-sm font-medium mb-2">Highly Correlated Feature Pairs</div>
          <div className="bg-yellow-50 dark:bg-yellow-950/30 rounded-lg p-4">
            <p className="text-xs text-yellow-700 dark:text-yellow-300 mb-2">
              These feature pairs are highly correlated with each other, which may cause multicollinearity issues.
            </p>
            <div className="space-y-1">
              {keyStats.high_correlation_pairs.slice(0, 5).map((pair, i) => (
                <div key={i} className="flex items-center gap-2 text-sm">
                  <span className="font-mono text-xs">{renderValue(pair.feature1)}</span>
                  <span className="text-muted-foreground">↔</span>
                  <span className="font-mono text-xs">{renderValue(pair.feature2)}</span>
                  <span className="text-yellow-600 font-medium ml-auto">
                    {typeof pair.correlation === 'number' ? pair.correlation.toFixed(3) : 'N/A'}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Leakage Warnings */}
      {keyStats.leakage_warnings && keyStats.leakage_warnings.length > 0 && (
        <div className="bg-destructive/10 border border-destructive/20 rounded-lg p-4">
          <div className="flex items-center gap-2 text-destructive mb-2">
            <AlertTriangle className="h-4 w-4" />
            <span className="font-medium">Potential Data Leakage Detected</span>
          </div>
          <ul className="text-sm text-destructive/80 space-y-1">
            {keyStats.leakage_warnings.map((w, i) => (
              <li key={i}>• {w}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Numeric Feature Statistics */}
      {keyStats.numeric_summaries && keyStats.numeric_summaries.length > 0 && (
        <div>
          <div className="text-sm font-medium mb-2">Numeric Feature Statistics</div>
          <div className="overflow-x-auto rounded-lg border">
            <table className="w-full text-xs">
              <thead className="bg-muted/50">
                <tr>
                  <th className="text-left py-2 px-2 font-medium">Feature</th>
                  <th className="text-right py-2 px-2 font-medium">Mean</th>
                  <th className="text-right py-2 px-2 font-medium">Std</th>
                  <th className="text-right py-2 px-2 font-medium">Min</th>
                  <th className="text-right py-2 px-2 font-medium">Max</th>
                  <th className="text-right py-2 px-2 font-medium">Null%</th>
                </tr>
              </thead>
              <tbody>
                {keyStats.numeric_summaries.slice(0, 10).map((stat, i) => (
                  <tr key={i} className="border-t border-border/50">
                    <td className="py-1.5 px-2 font-mono">{renderValue(stat.column)}</td>
                    <td className="text-right py-1.5 px-2">{typeof stat.mean === 'number' ? stat.mean.toFixed(2) : 'N/A'}</td>
                    <td className="text-right py-1.5 px-2">{typeof stat.std === 'number' ? stat.std.toFixed(2) : 'N/A'}</td>
                    <td className="text-right py-1.5 px-2">{typeof stat.min === 'number' ? stat.min.toFixed(2) : 'N/A'}</td>
                    <td className="text-right py-1.5 px-2">{typeof stat.max === 'number' ? stat.max.toFixed(2) : 'N/A'}</td>
                    <td className="text-right py-1.5 px-2">{typeof stat.outliers_pct === 'number' ? stat.outliers_pct.toFixed(1) : '0'}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Categorical Summaries */}
      {keyStats.categorical_summaries && keyStats.categorical_summaries.length > 0 && (
        <div>
          <div className="text-sm font-medium mb-2">Categorical Features</div>
          <div className="grid grid-cols-2 gap-3">
            {keyStats.categorical_summaries.slice(0, 6).map((cat, i) => (
              <div key={i} className="bg-muted/30 rounded-lg p-3">
                <div className="font-mono text-xs font-medium mb-1">{renderValue(cat.column)}</div>
                <div className="text-xs text-muted-foreground">{cat.group_count} unique values</div>
                {cat.groups && cat.groups.length > 0 && (
                  <div className="text-xs text-muted-foreground mt-1">
                    Top: {cat.groups.slice(0, 3).map(g => renderValue(g)).join(", ")}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Features List */}
      <div>
        <div className="text-sm font-medium mb-2">Features Selected ({features.length})</div>
        <div className="space-y-1">
          {displayedFeatures.map((f, i) => {
            const encodingStr = renderValue(f.encoding)
            const formulaStr = renderValue(f.formula)
            return (
              <div key={i} className="flex items-center gap-2 text-sm bg-muted/30 rounded px-3 py-2">
                <span className="font-medium flex-1">{renderValue(f.name)}</span>
                {encodingStr && encodingStr !== "N/A" && (
                  <Badge variant="outline" className="text-xs font-normal">{encodingStr}</Badge>
                )}
                {formulaStr && formulaStr !== "N/A" && (
                  <span className="text-xs text-muted-foreground font-mono truncate max-w-[200px]" title={formulaStr}>
                    {formulaStr}
                  </span>
                )}
              </div>
            )
          })}
        </div>
        {features.length > 10 && (
          <button
            onClick={() => setShowAllFeatures(!showAllFeatures)}
            className="mt-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
          >
            {showAllFeatures ? "Show less" : `Show all ${features.length} features`}
          </button>
        )}
      </div>
    </div>
  )
}

// Feature Engineering Detail
function FeatureEngineeringDetail({ agentState, audit }: { agentState: TrainingAgentState; audit?: Record<string, unknown> }) {
  return (
    <div className="space-y-6">
      {/* Validation Status */}
      <div className={`rounded-xl p-4 ${agentState.feature_validation_passed ? 'bg-success/8 dark:bg-success/12' : 'bg-yellow-50 dark:bg-yellow-950/30'}`}>
        <div className="flex items-center gap-2">
          <CheckCircle2 className={`h-5 w-5 ${agentState.feature_validation_passed ? 'text-success' : 'text-yellow-600'}`} />
          <span className={`font-medium ${agentState.feature_validation_passed ? 'text-foreground dark:text-success' : 'text-yellow-700 dark:text-yellow-300'}`}>
            {agentState.feature_validation_passed ? "Feature Validation Passed" : "Validation Issues Found"}
          </span>
        </div>
      </div>

      {/* Features Created */}
      {audit?.features_created != null && Array.isArray(audit.features_created) && (
        <div>
          <div className="text-sm font-medium mb-2">Features Created ({(audit.features_created as string[]).length})</div>
          <div className="flex flex-wrap gap-1.5">
            {(audit.features_created as string[]).map((f, i) => (
              <Badge key={i} variant="secondary" className="text-xs font-mono font-normal">{f}</Badge>
            ))}
          </div>
        </div>
      )}

      {/* Dataset Shapes */}
      {audit?.shapes != null && typeof audit.shapes === "object" && (
        <div>
          <div className="text-sm font-medium mb-2">Dataset Shapes</div>
          <div className="grid grid-cols-3 gap-3">
            {Object.entries(audit.shapes as Record<string, unknown>).map(([key, shape]) => {
              let label: string
              if (Array.isArray(shape) && shape.length >= 2) {
                label = `${Number(shape[0]).toLocaleString()} × ${shape[1]}`
              } else if (shape != null) {
                label = String(shape)
              } else {
                return null
              }
              return (
                <div key={key} className="bg-muted/30 rounded-lg p-3 text-center">
                  <div className="text-xs text-muted-foreground capitalize">{key}</div>
                  <div className="text-lg font-semibold">{label}</div>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Transformed Datasets */}
      <div>
        <div className="text-sm font-medium mb-2">Transformed Dataset References</div>
        <div className="space-y-2">
          {agentState.transformed_train_ref && (
            <div className="flex items-center gap-3 bg-muted/30 rounded-lg p-2">
              <Badge variant="outline" className="text-xs">Train</Badge>
              <code className="text-xs font-mono truncate flex-1">{agentState.transformed_train_ref}</code>
            </div>
          )}
          {agentState.transformed_val_ref && (
            <div className="flex items-center gap-3 bg-muted/30 rounded-lg p-2">
              <Badge variant="outline" className="text-xs">Val</Badge>
              <code className="text-xs font-mono truncate flex-1">{agentState.transformed_val_ref}</code>
            </div>
          )}
          {agentState.transformed_test_ref && (
            <div className="flex items-center gap-3 bg-muted/30 rounded-lg p-2">
              <Badge variant="outline" className="text-xs">Test</Badge>
              <code className="text-xs font-mono truncate flex-1">{agentState.transformed_test_ref}</code>
            </div>
          )}
        </div>
      </div>

      {/* Errors */}
      {audit?.errors != null && Array.isArray(audit.errors) && (audit.errors as string[]).length > 0 && (
        <div className="bg-destructive/10 rounded-lg p-4">
          <div className="text-sm font-medium text-destructive mb-1">Errors Encountered</div>
          <ul className="text-sm text-destructive/80 space-y-1">
            {(audit.errors as string[]).map((e, i) => (
              <li key={i}>• {e}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}

// Training Config Detail (training_approval step)
function TrainingConfigDetail({ agentState, audit }: { agentState: TrainingAgentState; audit?: Record<string, unknown> }) {
  const [expandedSection, setExpandedSection] = useState<number | null>(null)

  const plan = (audit as Record<string, unknown> | undefined) || {}
  const tp = (plan.training_plan as Record<string, unknown>) ||
    (agentState.training_params as Record<string, unknown>) || {}
  const hp = (tp.hyperparameters || plan.hyperparameters || {}) as Record<string, unknown>
  const strategy = (tp.strategy_notes || plan.strategy_notes || []) as string[] | string
  const dataSummary = (tp.data_summary || plan.data_summary || {}) as Record<string, unknown>
  const modelType = String(tp.model_type || plan.model_type || "Unknown")
  const taskType = String(tp.task_type || plan.task_type || "")
  const classWeight = tp.class_weight || plan.class_weight
  const maxIter = tp.max_iterations || plan.max_iterations

  const strategyNotes = Array.isArray(strategy) ? strategy : strategy ? [String(strategy)] : []

  const sectionTitles = [
    "Objective & Data",
    "Imbalance Handling",
    "Hyperparameter Rationale",
    "Training Iterations",
    "Feature & Preprocessing",
    "Validation Protocol",
    "Deployment & Interpretability",
    "Performance Tuning",
  ]

  const guessSectionTitle = (text: string, index: number): string => {
    const lower = text.toLowerCase()
    for (const title of sectionTitles) {
      if (lower.startsWith(title.toLowerCase())) return title
    }
    const colonIdx = text.indexOf(":")
    if (colonIdx > 0 && colonIdx < 50) return text.slice(0, colonIdx).trim()
    return `Section ${index + 1}`
  }

  return (
    <div className="space-y-6">
      {/* Model & Task header */}
      <div className="flex gap-4">
        <div className="flex-1 bg-foreground/5 rounded-xl p-4">
          <div className="text-xs text-muted-foreground mb-1">Model</div>
          <div className="text-xl font-semibold">{modelType}</div>
        </div>
        {taskType && (
          <div className="flex-1 bg-foreground/5 rounded-xl p-4">
            <div className="text-xs text-muted-foreground mb-1">Task</div>
            <div className="text-xl font-semibold">{taskType}</div>
          </div>
        )}
      </div>

      {/* Hyperparameters grid */}
      {Object.keys(hp).length > 0 ? (
        <div>
          <div className="text-sm font-medium mb-3">Hyperparameters</div>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
            {Object.entries(hp).map(([key, value]) => (
              <div key={key} className="bg-muted/30 rounded-lg px-3 py-2">
                <div className="text-[11px] text-muted-foreground font-mono truncate">{key}</div>
                <div className="text-sm font-semibold font-mono mt-0.5">{String(value)}</div>
              </div>
            ))}
          </div>
        </div>
      ) : null}

      {/* Class weight & max iterations */}
      {(classWeight || maxIter) ? (
        <div className="flex gap-4">
          {classWeight ? (
            <div className="bg-blue-50 dark:bg-blue-950/30 rounded-lg px-3 py-2">
              <div className="text-[11px] text-blue-600 dark:text-blue-400">Class Weight</div>
              <div className="text-sm font-medium">{String(classWeight)}</div>
            </div>
          ) : null}
          {maxIter ? (
            <div className="bg-blue-50 dark:bg-blue-950/30 rounded-lg px-3 py-2">
              <div className="text-[11px] text-blue-600 dark:text-blue-400">Max Iterations</div>
              <div className="text-sm font-medium">{String(maxIter)}</div>
            </div>
          ) : null}
        </div>
      ) : null}

      {/* Data Summary */}
      {Object.keys(dataSummary).length > 0 && (
        <div>
          <div className="text-sm font-medium mb-2">Data Summary</div>
          <div className="grid grid-cols-2 gap-2 text-sm">
            {Object.entries(dataSummary).map(([key, value]) => (
              <div key={key} className="flex justify-between bg-muted/20 rounded-lg px-3 py-2">
                <span className="text-muted-foreground">{key.replace(/_/g, " ")}</span>
                <span className="font-medium">{typeof value === "number" ? value.toLocaleString() : String(value)}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Strategy sections */}
      {strategyNotes.length > 0 && (
        <div>
          <div className="text-sm font-medium mb-3">Training Strategy</div>
          <div className="space-y-2">
            {strategyNotes.map((note, i) => {
              const title = guessSectionTitle(note, i)
              const isExpanded = expandedSection === i
              const body = note.startsWith(title) ? note.slice(title.length).replace(/^[\s:–—-]+/, "") : note

              return (
                <div key={i} className="border border-border/50 rounded-lg overflow-hidden">
                  <button
                    onClick={() => setExpandedSection(isExpanded ? null : i)}
                    className="w-full flex items-center justify-between px-4 py-2.5 text-left hover:bg-muted/30 transition-colors"
                  >
                    <div className="flex items-center gap-2">
                      <span className="w-5 h-5 rounded-full bg-foreground/10 flex items-center justify-center text-[11px] font-medium text-muted-foreground">
                        {i + 1}
                      </span>
                      <span className="text-sm font-medium">{title}</span>
                    </div>
                    <span className={`text-muted-foreground transition-transform ${isExpanded ? "rotate-180" : ""}`}>
                      ▾
                    </span>
                  </button>
                  {isExpanded && (
                    <div className="px-4 pb-3 pt-0">
                      <p className="text-sm leading-relaxed text-muted-foreground">{body}</p>
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}

// Training Detail
function TrainingDetail({ agentState }: { agentState: TrainingAgentState }) {
  const [showAllIterations, setShowAllIterations] = useState(false)
  
  const metrics = agentState.training_metrics
  const iterations = metrics?.iterations || []
  const hasClassification = metrics?.test_accuracy != null || metrics?.test_roc_auc != null
  const hasRegression = metrics?.test_r2 != null || metrics?.val_r2 != null || iterations[0]?.test_r2 != null
  
  const bestIter = metrics?.best_iteration as Record<string, number | string | null | undefined> | undefined
  const testR2 = metrics?.test_r2 ?? (bestIter?.test_r2 as number | undefined)
  const testRmse = metrics?.test_rmse ?? (bestIter?.test_rmse as number | undefined)
  const testMae = metrics?.test_mae ?? (bestIter?.test_mae as number | undefined)
  const valR2 = metrics?.val_r2 ?? (bestIter?.val_r2 as number | undefined)
  const valRmse = metrics?.val_rmse ?? (bestIter?.val_rmse as number | undefined)
  const valMae = metrics?.val_mae ?? (bestIter?.val_mae as number | undefined)
  const trainR2 = metrics?.train_r2 ?? (bestIter?.train_r2 as number | undefined)
  const trainRmse = metrics?.train_rmse ?? (bestIter?.train_rmse as number | undefined)

  const displayedIterations = showAllIterations ? iterations : iterations.slice(0, 3)

  return (
    <div className="space-y-6">
      {/* Status */}
      <div className={`rounded-xl p-4 ${metrics?.success ? 'bg-success/8 dark:bg-success/12' : 'bg-red-50 dark:bg-red-950/30'}`}>
        <div className="flex items-center gap-2">
          <CheckCircle2 className={`h-5 w-5 ${metrics?.success ? 'text-success' : 'text-destructive'}`} />
          <span className={`font-medium ${metrics?.success ? 'text-foreground dark:text-success' : 'text-red-700 dark:text-red-300'}`}>
            {metrics?.success ? "Training Successful" : "Training Completed"}
          </span>
        </div>
      </div>

      {/* Model Information */}
      <div>
        <div className="text-sm font-medium mb-2">Model Information</div>
        <div className="grid grid-cols-2 gap-3">
          <div className="bg-muted/30 rounded-lg p-3">
            <div className="text-xs text-muted-foreground">Model Type</div>
            <div className="font-semibold">{metrics?.model_type || agentState.selected_model || "N/A"}</div>
          </div>
          <div className="bg-muted/30 rounded-lg p-3">
            <div className="text-xs text-muted-foreground">Iterations</div>
            <div className="font-semibold">{metrics?.num_iterations || iterations.length || 1}</div>
          </div>
          {metrics?.model_name && (
            <div className="bg-muted/30 rounded-lg p-3 col-span-2">
              <div className="text-xs text-muted-foreground">Model Name</div>
              <code className="text-sm font-mono">{metrics.model_name}</code>
            </div>
          )}
        </div>
      </div>

      {/* Key Metrics - Hero */}
      <div className="grid grid-cols-2 gap-4">
        {hasClassification ? (
          <>
            <div className="bg-foreground/5 rounded-xl p-4">
              <div className="text-xs text-muted-foreground mb-1">Test Accuracy</div>
              <div className="text-2xl font-semibold">{formatPercent(metrics?.test_accuracy)}</div>
            </div>
            <div className="bg-foreground/5 rounded-xl p-4">
              <div className="text-xs text-muted-foreground mb-1">Test ROC-AUC</div>
              <div className="text-2xl font-semibold">{formatNumber(metrics?.test_roc_auc, 3)}</div>
            </div>
          </>
        ) : hasRegression ? (
          <>
            <div className="bg-foreground/5 rounded-xl p-4">
              <div className="text-xs text-muted-foreground mb-1">Test R²</div>
              <div className="text-2xl font-semibold">{formatNumber(testR2, 4)}</div>
            </div>
            <div className="bg-foreground/5 rounded-xl p-4">
              <div className="text-xs text-muted-foreground mb-1">Test RMSE</div>
              <div className="text-2xl font-semibold">{formatNumber(testRmse, 2)}</div>
            </div>
          </>
        ) : (
          <>
            <div className="bg-foreground/5 rounded-xl p-4">
              <div className="text-xs text-muted-foreground mb-1">Status</div>
              <div className="text-2xl font-semibold">Complete</div>
            </div>
            <div className="bg-foreground/5 rounded-xl p-4">
              <div className="text-xs text-muted-foreground mb-1">Iterations</div>
              <div className="text-2xl font-semibold">{metrics?.num_iterations || 1}</div>
            </div>
          </>
        )}
      </div>

      {/* Full Metrics Table */}
      <div>
        <div className="text-sm font-medium mb-2">All Metrics</div>
        <div className="overflow-x-auto rounded-lg border">
          <table className="w-full text-sm">
            <thead className="bg-muted/50">
              <tr>
                <th className="text-left py-2 px-3 font-medium">Metric</th>
                <th className="text-right py-2 px-3 font-medium">Train</th>
                <th className="text-right py-2 px-3 font-medium">Validation</th>
                <th className="text-right py-2 px-3 font-medium">Test</th>
              </tr>
            </thead>
            <tbody>
              {hasClassification ? (
                <>
                  <tr className="border-t border-border/50">
                    <td className="py-2 px-3">Accuracy</td>
                    <td className="text-right py-2 px-3 font-mono">{formatPercent(bestIter?.train_accuracy as number | undefined)}</td>
                    <td className="text-right py-2 px-3 font-mono">{formatPercent(metrics?.val_accuracy)}</td>
                    <td className="text-right py-2 px-3 font-mono font-semibold">{formatPercent(metrics?.test_accuracy)}</td>
                  </tr>
                  <tr className="border-t border-border/50">
                    <td className="py-2 px-3">ROC-AUC</td>
                    <td className="text-right py-2 px-3 font-mono">-</td>
                    <td className="text-right py-2 px-3 font-mono">{formatNumber(metrics?.val_roc_auc, 4)}</td>
                    <td className="text-right py-2 px-3 font-mono font-semibold">{formatNumber(metrics?.test_roc_auc, 4)}</td>
                  </tr>
                </>
              ) : hasRegression ? (
                <>
                  <tr className="border-t border-border/50">
                    <td className="py-2 px-3">R² Score</td>
                    <td className="text-right py-2 px-3 font-mono">{formatNumber(trainR2, 4)}</td>
                    <td className="text-right py-2 px-3 font-mono">{formatNumber(valR2, 4)}</td>
                    <td className="text-right py-2 px-3 font-mono font-semibold">{formatNumber(testR2, 4)}</td>
                  </tr>
                  <tr className="border-t border-border/50">
                    <td className="py-2 px-3">RMSE</td>
                    <td className="text-right py-2 px-3 font-mono">{formatNumber(trainRmse, 2)}</td>
                    <td className="text-right py-2 px-3 font-mono">{formatNumber(valRmse, 2)}</td>
                    <td className="text-right py-2 px-3 font-mono font-semibold">{formatNumber(testRmse, 2)}</td>
                  </tr>
                  <tr className="border-t border-border/50">
                    <td className="py-2 px-3">MAE</td>
                    <td className="text-right py-2 px-3 font-mono">-</td>
                    <td className="text-right py-2 px-3 font-mono">{formatNumber(valMae, 2)}</td>
                    <td className="text-right py-2 px-3 font-mono font-semibold">{formatNumber(testMae, 2)}</td>
                  </tr>
                </>
              ) : (
                <tr className="border-t border-border/50">
                  <td className="py-2 px-3 text-muted-foreground" colSpan={4}>No detailed metrics available</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Training Iterations */}
      {iterations.length > 0 && (
        <div>
          <div className="text-sm font-medium mb-2">Training Iterations ({iterations.length})</div>
          <div className="space-y-3">
            {displayedIterations.map((iter, i) => {
              const iterNum = iter.iteration ?? i + 1
              const bestIterName = bestIter?.model_name as string | undefined
              const isBest = bestIterName
                ? iter.model_name === bestIterName
                : i === iterations.length - 1
              
              return (
                <div 
                  key={i} 
                  className={`rounded-lg p-4 ${isBest ? 'bg-foreground/5 ring-1 ring-foreground/10' : 'bg-muted/30'}`}
                >
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <span className="font-medium">Iteration {iterNum}</span>
                      {isBest && <Badge variant="secondary" className="text-xs">Best</Badge>}
                      {iter.success === false && <Badge variant="destructive" className="text-xs">Failed</Badge>}
                    </div>
                    {iter.tool && (
                      <span className="text-xs text-muted-foreground font-mono">{iter.tool}</span>
                    )}
                  </div>
                  
                  {/* Iteration Metrics */}
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-sm mb-3">
                    {hasClassification ? (
                      <>
                        {iter.val_accuracy != null && (
                          <div>
                            <span className="text-muted-foreground">Val Acc:</span>{" "}
                            <span className="font-medium">{(iter.val_accuracy * 100).toFixed(2)}%</span>
                          </div>
                        )}
                        {iter.val_roc_auc != null && (
                          <div>
                            <span className="text-muted-foreground">Val AUC:</span>{" "}
                            <span className="font-medium">{iter.val_roc_auc.toFixed(4)}</span>
                          </div>
                        )}
                        {iter.train_accuracy != null && (
                          <div>
                            <span className="text-muted-foreground">Train Acc:</span>{" "}
                            <span className="font-medium">{(iter.train_accuracy * 100).toFixed(2)}%</span>
                          </div>
                        )}
                      </>
                    ) : (
                      <>
                        {iter.val_r2 != null && (
                          <div>
                            <span className="text-muted-foreground">Val R²:</span>{" "}
                            <span className="font-medium">{iter.val_r2.toFixed(4)}</span>
                          </div>
                        )}
                        {iter.val_rmse != null && (
                          <div>
                            <span className="text-muted-foreground">Val RMSE:</span>{" "}
                            <span className="font-medium">{iter.val_rmse.toFixed(2)}</span>
                          </div>
                        )}
                        {iter.test_r2 != null && (
                          <div>
                            <span className="text-muted-foreground">Test R²:</span>{" "}
                            <span className="font-medium">{iter.test_r2.toFixed(4)}</span>
                          </div>
                        )}
                        {iter.test_rmse != null && (
                          <div>
                            <span className="text-muted-foreground">Test RMSE:</span>{" "}
                            <span className="font-medium">{iter.test_rmse.toFixed(2)}</span>
                          </div>
                        )}
                      </>
                    )}
                  </div>
                  
                  {/* Hyperparameters */}
                  {iter.hyperparams && Object.keys(iter.hyperparams).length > 0 && (
                    <div className="mt-2 pt-2 border-t border-border/30">
                      <div className="text-xs text-muted-foreground mb-1">Hyperparameters</div>
                      <div className="flex flex-wrap gap-1.5">
                        {Object.entries(iter.hyperparams).slice(0, 6).map(([key, value]) => (
                          <span key={key} className="text-xs bg-muted/50 px-2 py-0.5 rounded font-mono">
                            {key}: {typeof value === 'number' ? value.toFixed(4) : String(value)}
                          </span>
                        ))}
                        {Object.keys(iter.hyperparams).length > 6 && (
                          <span className="text-xs text-muted-foreground">
                            +{Object.keys(iter.hyperparams).length - 6} more
                          </span>
                        )}
                      </div>
                    </div>
                  )}
                  
                  {/* Params/Config */}
                  {iter.params && Object.keys(iter.params).length > 0 && (
                    <div className="mt-2 pt-2 border-t border-border/30">
                      <div className="text-xs text-muted-foreground mb-1">Configuration</div>
                      <div className="flex flex-wrap gap-1.5">
                        {Object.entries(iter.params).slice(0, 6).map(([key, value]) => (
                          <span key={key} className="text-xs bg-muted/50 px-2 py-0.5 rounded font-mono">
                            {key}: {renderValue(value)}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                  
                  {/* Error if any */}
                  {iter.error && (
                    <div className="mt-2 pt-2 border-t border-border/30">
                      <div className="text-xs text-destructive">{iter.error}</div>
                    </div>
                  )}
                </div>
              )
            })}
          </div>
          {iterations.length > 3 && (
            <button
              onClick={() => setShowAllIterations(!showAllIterations)}
              className="mt-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
            >
              {showAllIterations ? "Show less" : `Show all ${iterations.length} iterations`}
            </button>
          )}
        </div>
      )}

      {/* Model Path */}
      {agentState.model_weights_path && (
        <div>
          <div className="text-sm font-medium mb-2">Model Saved To</div>
          <code className="text-xs font-mono bg-muted/30 px-3 py-2 rounded-lg block break-all">
            {agentState.model_weights_path}
          </code>
        </div>
      )}

      {/* Feature Redo Info */}
      {agentState.feature_redo_requested && (
        <div className="bg-yellow-50 dark:bg-yellow-950/30 rounded-lg p-4">
          <div className="text-sm font-medium text-yellow-700 dark:text-yellow-300 mb-2">Feature Redo Requested</div>
          {agentState.feature_redo_reason && (
            <p className="text-sm text-yellow-600 dark:text-yellow-400 mb-1">
              <strong>Reason:</strong> {agentState.feature_redo_reason}
            </p>
          )}
          {agentState.feature_redo_recommendation && (
            <p className="text-sm text-yellow-600 dark:text-yellow-400">
              <strong>Recommendation:</strong> {agentState.feature_redo_recommendation}
            </p>
          )}
          <p className="text-xs text-yellow-600 dark:text-yellow-400 mt-2">
            Redo iteration: {agentState.feature_redo_iteration}
          </p>
        </div>
      )}

      {/* Summary */}
      {metrics?.summary && (
        <div className="bg-muted/30 rounded-lg p-4">
          <div className="text-sm font-medium mb-2">Training Summary</div>
          <p className="text-sm text-muted-foreground whitespace-pre-wrap">{metrics.summary}</p>
        </div>
      )}

      {/* Recommendations */}
      {metrics?.recommendations && (
        <div className="bg-primary/6 dark:bg-primary/12 rounded-lg p-4 border border-border/50">
          <div className="text-sm font-medium text-foreground mb-2">Recommendations</div>
          <p className="text-sm text-primary">{metrics.recommendations}</p>
        </div>
      )}
    </div>
  )
}

// Report Detail
function ReportDetail({ agentState }: { agentState: TrainingAgentState }) {
  return (
    <div className="space-y-6">
      {/* Success Message */}
      <div className="bg-success/8 dark:bg-success/12 rounded-xl p-4">
        <div className="flex items-center gap-2">
          <CheckCircle2 className="h-5 w-5 text-success" />
          <span className="font-medium text-foreground dark:text-success">Experiment complete</span>
        </div>
      </div>

      {/* Report and Model Paths */}
      <div className="space-y-3">
        {agentState.report_path && (
          <div>
            <div className="text-sm font-medium mb-1">Report Path</div>
            <code className="text-xs font-mono bg-muted/30 px-3 py-2 rounded-lg block break-all">
              {agentState.report_path}
            </code>
          </div>
        )}
        {agentState.model_weights_path && (
          <div>
            <div className="text-sm font-medium mb-1">Model Weights</div>
            <code className="text-xs font-mono bg-muted/30 px-3 py-2 rounded-lg block break-all">
              {agentState.model_weights_path}
            </code>
          </div>
        )}
      </div>

      {/* What's included */}
      <div>
        <div className="text-sm font-medium mb-2">Report Includes</div>
        <ul className="text-sm text-muted-foreground space-y-1">
          <li className="flex gap-2"><span className="text-success">✓</span> Model performance metrics</li>
          <li className="flex gap-2"><span className="text-success">✓</span> Feature importance rankings</li>
          <li className="flex gap-2"><span className="text-success">✓</span> Data pipeline documentation</li>
          <li className="flex gap-2"><span className="text-success">✓</span> Training configuration</li>
          <li className="flex gap-2"><span className="text-success">✓</span> Audit trace of all steps</li>
        </ul>
      </div>

      {/* Next steps */}
      <div className="bg-primary/6 dark:bg-primary/12 rounded-lg p-4 border border-border/50">
        <div className="text-sm font-medium text-foreground mb-2">Next Steps</div>
        <ul className="text-sm text-muted-foreground space-y-1">
          <li>• Review the full report for detailed insights</li>
          <li>• Deploy the model for inference</li>
          <li>• Monitor model performance in production</li>
        </ul>
      </div>
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
