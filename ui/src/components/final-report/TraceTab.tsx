import { useState } from "react"
import { Badge } from "@/components/ui/badge"
import { CheckCircle2, ChevronDown, ChevronRight } from "lucide-react"
import type { TrainingAgentState, StepInfo, KeyStats } from "@/types/agent"
import { Section, InfoBox } from "./shared"
import { getIterationMetrics, renderValue } from "./utils"

interface TraceTabProps {
  steps: StepInfo[]
  agentState: TrainingAgentState
}

export function TraceTab({ steps, agentState }: TraceTabProps) {
  const [expanded, setExpanded] = useState<string | null>(null)
  const auditTrace = agentState.audit_trace || []
  const completedSteps = steps.filter((s) => s.status === "completed")

  const toggle = (id: string) => setExpanded(expanded === id ? null : id)

  const getAuditEntry = (stepId: string) => {
    const stepMapping: Record<string, string> = {
      select_model: "select_model",
      data_collection: "data_collection",
      cleaning: "cleaning_and_standardization",
      label_split_definition: "label_split_definition",
      feature_selection_specification: "feature_selection_specification",
      feature_engineering_executor: "feature_engineering_executor",
      training: "training",
      generate_report: "generate_report",
    }
    const auditStepName = stepMapping[stepId] || stepId
    return auditTrace.find((t) => t.step === auditStepName)
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
                  <CheckCircle2 className="h-4 w-4 text-success" />
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
                    <StepContent
                      stepId={step.id}
                      agentState={agentState}
                      audit={getAuditEntry(step.id)}
                      auditTrace={auditTrace}
                    />
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

/**
 * Renders content for a specific step in the trace
 */
function StepContent({
  stepId,
  agentState,
  audit,
  auditTrace,
}: {
  stepId: string
  agentState: TrainingAgentState
  audit: Record<string, unknown> | undefined
  auditTrace: Record<string, unknown>[]
}) {
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
            {audit.confidence != null && (
              <InfoBox label="Confidence" value={String(audit.confidence)} />
            )}
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
            <div className="text-sm text-muted-foreground mb-1">
              Columns ({(audit.columns as string[]).length})
            </div>
            <p className="text-sm">{(audit.columns as string[]).join(", ")}</p>
          </div>
        )}
      </div>
    )
  }

  if (stepId === "cleaning") {
    return <CleaningStepContent agentState={agentState} />
  }

  if (stepId === "label_split_definition") {
    return <LabelSplitStepContent agentState={agentState} />
  }

  if (stepId === "feature_selection_specification") {
    return <FeatureSelectionStepContent agentState={agentState} />
  }

  if (stepId === "feature_engineering_executor") {
    return <FeatureEngineeringStepContent agentState={agentState} audit={audit} />
  }

  if (stepId === "training") {
    return <TrainingStepContent agentState={agentState} />
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

/**
 * Content for the cleaning step
 */
function CleaningStepContent({ agentState }: { agentState: TrainingAgentState }) {
  const transformations = agentState.cleaning_transformations || []
  const cleaningSummary = agentState.cleaning_summary

  const formatTransformation = (t: unknown, index: number) => {
    if (typeof t === "object" && t !== null) {
      const transform = t as Record<string, unknown>
      const toolName = transform.tool || transform.op || transform.operation || "transform"
      const args = transform.args as Record<string, unknown> | undefined
      const result = transform.result as string | undefined

      let columns = ""
      let extraInfo = ""
      if (args) {
        if (args.columns) {
          columns = Array.isArray(args.columns)
            ? (args.columns as string[]).join(", ")
            : String(args.columns)
        } else if (args.column) {
          columns = String(args.column)
        }
        if (args.value !== undefined) extraInfo = `= ${args.value}`
        if (args.strategy) extraInfo = `(${args.strategy})`
      }

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
        <div className="text-sm font-medium mb-2">
          Transformations Applied ({transformations.length})
        </div>
        {transformations.length > 0 ? (
          <div className="space-y-1.5 max-h-[400px] overflow-y-auto">
            {transformations.map((t, i) => formatTransformation(t, i))}
          </div>
        ) : (
          <p className="text-sm text-muted-foreground">
            No transformations needed - data was already clean
          </p>
        )}
      </div>
    </div>
  )
}

/**
 * Content for the label split definition step
 */
function LabelSplitStepContent({ agentState }: { agentState: TrainingAgentState }) {
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

/**
 * Content for the feature selection step
 */
function FeatureSelectionStepContent({ agentState }: { agentState: TrainingAgentState }) {
  const features = agentState.feature_spec?.features || []
  const analysisTrace = agentState.analysis_trace || []
  const featureAnalysis = analysisTrace.find((t) => t.step === "feature_selection_specification")
  const keyStats = (featureAnalysis?.key_stats || {}) as KeyStats

  return (
    <div className="space-y-4">
      {keyStats.summary_text && (
        <div className="bg-muted/30 rounded-lg p-3">
          <div className="text-sm">{keyStats.summary_text}</div>
        </div>
      )}

      {keyStats.dataset_overview?.rows && (
        <div className="grid grid-cols-4 gap-2">
          <InfoBox label="Rows" value={keyStats.dataset_overview.rows.toLocaleString()} small />
          <InfoBox label="Columns" value={String(keyStats.dataset_overview.columns)} small />
          <InfoBox label="Numeric" value={String(keyStats.dataset_overview.numeric_columns)} small />
          <InfoBox
            label="Categorical"
            value={String(keyStats.dataset_overview.categorical_columns)}
            small
          />
        </div>
      )}

      {keyStats.feature_correlations && keyStats.feature_correlations.length > 0 && (
        <div>
          <div className="text-sm text-muted-foreground mb-2">Top Correlations with Target</div>
          <div className="space-y-1">
            {keyStats.feature_correlations.slice(0, 5).map((corr, i) => (
              <div key={i} className="flex justify-between text-sm bg-muted/30 rounded px-3 py-1.5">
                <span className="font-mono text-xs">{corr.feature}</span>
                <span
                  className={`font-medium ${corr.correlation >= 0 ? "text-success" : "text-destructive"}`}
                >
                  {corr.correlation >= 0 ? "+" : ""}
                  {corr.correlation.toFixed(4)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {keyStats.leakage_warnings && keyStats.leakage_warnings.length > 0 && (
        <div className="bg-destructive/10 rounded-lg p-3">
          <div className="text-sm font-medium text-destructive mb-1">Leakage Warnings</div>
          <ul className="text-sm text-destructive/80 space-y-0.5">
            {keyStats.leakage_warnings.map((w, i) => (
              <li key={i}>• {w}</li>
            ))}
          </ul>
        </div>
      )}

      <div>
        <div className="text-sm text-muted-foreground mb-2">
          Features Specified ({features.length})
        </div>
        <div className="space-y-1 max-h-[200px] overflow-y-auto">
          {features.map((f, i) => (
            <div key={i} className="flex items-center gap-2 text-sm bg-muted/30 rounded px-3 py-2">
              <span className="font-medium">{renderValue(f.name)}</span>
              <Badge variant="outline" className="text-xs font-normal">
                {renderValue(f.encoding)}
              </Badge>
              {f.formula && (
                <span className="text-muted-foreground text-xs font-mono truncate">
                  {renderValue(f.formula)}
                </span>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

/**
 * Content for the feature engineering executor step
 */
function FeatureEngineeringStepContent({
  agentState,
  audit,
}: {
  agentState: TrainingAgentState
  audit: Record<string, unknown> | undefined
}) {
  return (
    <div className="space-y-3">
      <InfoBox
        label="Validation"
        value={agentState.feature_validation_passed ? "Passed" : "Issues Found"}
        highlight={agentState.feature_validation_passed}
      />
      {audit?.features_created != null && Array.isArray(audit.features_created) ? (
        <div>
          <div className="text-sm text-muted-foreground mb-1">
            Features Created ({(audit.features_created as string[]).length})
          </div>
          <p className="text-sm">{(audit.features_created as string[]).join(", ")}</p>
        </div>
      ) : null}
      {audit?.shapes != null && typeof audit.shapes === "object" ? (
        <div>
          <div className="text-sm text-muted-foreground mb-2">Dataset Shapes</div>
          <div className="grid grid-cols-3 gap-2">
            {Object.entries(audit.shapes as Record<string, number[]>).map(([key, shape]) => (
              <InfoBox
                key={key}
                label={key}
                value={Array.isArray(shape) ? `${shape[0]} × ${shape[1]}` : String(shape)}
                small
              />
            ))}
          </div>
        </div>
      ) : null}
      {audit?.errors != null &&
      Array.isArray(audit.errors) &&
      (audit.errors as string[]).length > 0 ? (
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

/**
 * Content for the training step
 */
function TrainingStepContent({ agentState }: { agentState: TrainingAgentState }) {
  const metrics = agentState.training_metrics
  const iterations = metrics?.iterations || []
  const hasClassification = metrics?.test_accuracy != null || metrics?.test_roc_auc != null
  const hasRegression =
    metrics?.test_r2 != null || metrics?.val_r2 != null || iterations[0]?.test_r2 != null

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3">
        <InfoBox
          label="Status"
          value={metrics?.success ? "Success" : "Failed"}
          highlight={metrics?.success}
        />
        <InfoBox
          label="Total Iterations"
          value={String(metrics?.num_iterations || iterations.length || 1)}
        />
      </div>
      <InfoBox label="Best Model" value={metrics?.model_name || agentState.model_weights_path} mono />

      {hasClassification && (
        <div>
          <div className="text-sm text-muted-foreground mb-2">Final Classification Metrics</div>
          <div className="grid grid-cols-2 gap-2 text-sm">
            {metrics?.val_accuracy != null && (
              <span>
                Val Accuracy: <strong>{(metrics.val_accuracy * 100).toFixed(2)}%</strong>
              </span>
            )}
            {metrics?.test_accuracy != null && (
              <span>
                Test Accuracy: <strong>{(metrics.test_accuracy * 100).toFixed(2)}%</strong>
              </span>
            )}
            {metrics?.val_roc_auc != null && (
              <span>
                Val ROC-AUC: <strong>{metrics.val_roc_auc.toFixed(4)}</strong>
              </span>
            )}
            {metrics?.test_roc_auc != null && (
              <span>
                Test ROC-AUC: <strong>{metrics.test_roc_auc.toFixed(4)}</strong>
              </span>
            )}
          </div>
        </div>
      )}

      {hasRegression && (
        <div>
          <div className="text-sm text-muted-foreground mb-2">Final Regression Metrics</div>
          <div className="grid grid-cols-2 gap-2 text-sm">
            {(metrics?.val_r2 ?? iterations[0]?.val_r2) != null && (
              <span>
                Val R²: <strong>{(metrics?.val_r2 ?? iterations[0]?.val_r2)?.toFixed(4)}</strong>
              </span>
            )}
            {(metrics?.test_r2 ?? iterations[0]?.test_r2) != null && (
              <span>
                Test R²: <strong>{(metrics?.test_r2 ?? iterations[0]?.test_r2)?.toFixed(4)}</strong>
              </span>
            )}
            {(metrics?.val_rmse ?? iterations[0]?.val_rmse) != null && (
              <span>
                Val RMSE: <strong>{(metrics?.val_rmse ?? iterations[0]?.val_rmse)?.toFixed(2)}</strong>
              </span>
            )}
            {(metrics?.test_rmse ?? iterations[0]?.test_rmse) != null && (
              <span>
                Test RMSE:{" "}
                <strong>{(metrics?.test_rmse ?? iterations[0]?.test_rmse)?.toFixed(2)}</strong>
              </span>
            )}
            {(metrics?.test_mae ?? iterations[0]?.test_mae) != null && (
              <span>
                Test MAE: <strong>{(metrics?.test_mae ?? iterations[0]?.test_mae)?.toFixed(2)}</strong>
              </span>
            )}
          </div>
        </div>
      )}

      {/* All Iterations Log */}
      {iterations.length > 0 && (
        <div>
          <div className="text-sm font-medium mb-2">
            All Training Iterations ({iterations.length})
          </div>
          <div className="space-y-2 max-h-[400px] overflow-y-auto">
            {iterations.map((iter, i) => (
              <TrainingIterationRow
                key={i}
                iteration={iter}
                index={i}
                isBest={
                  metrics?.best_iteration && typeof metrics.best_iteration === "object"
                    ? (metrics.best_iteration as Record<string, unknown>).model_name ===
                      getIterationMetrics(iter).model_name
                    : i === iterations.length - 1
                }
              />
            ))}
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

/**
 * Renders a single training iteration row
 */
function TrainingIterationRow({
  iteration,
  index,
  isBest,
}: {
  iteration: Parameters<typeof getIterationMetrics>[0]
  index: number
  isBest: boolean
}) {
  const iterMetrics = getIterationMetrics(iteration)
  const hasIterClassification = iterMetrics.val_accuracy != null || iterMetrics.val_roc_auc != null
  const hasIterRegression = iterMetrics.val_r2 != null || iterMetrics.test_r2 != null

  return (
    <div
      className={`p-3 rounded-lg ${
        isBest
          ? "bg-success/10 ring-1 ring-success/20"
          : iterMetrics.success === false
            ? "bg-destructive/10"
            : "bg-muted/30"
      }`}
    >
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium">Iteration {iteration.iteration ?? index + 1}</span>
          {isBest && (
            <Badge
              variant="secondary"
              className="text-xs bg-success/15 text-success dark:bg-success/20"
            >
              Best
            </Badge>
          )}
          {iterMetrics.success === false && (
            <Badge variant="destructive" className="text-xs">
              Failed
            </Badge>
          )}
          {iterMetrics.success === true && !isBest && (
            <Badge variant="outline" className="text-xs">
              OK
            </Badge>
          )}
        </div>
      </div>

      {/* Model and tool info */}
      <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground mb-2">
        {iterMetrics.model_name && (
          <span>
            Model:{" "}
            <code className="text-foreground bg-muted/50 px-1 rounded">{iterMetrics.model_name}</code>
          </span>
        )}
        {iterMetrics.tool && (
          <span>
            Tool:{" "}
            <code className="text-foreground bg-muted/50 px-1 rounded">{iterMetrics.tool}</code>
          </span>
        )}
      </div>

      {/* Metrics */}
      {hasIterClassification && (
        <div className="grid grid-cols-2 gap-2 text-xs">
          {iterMetrics.val_accuracy != null && (
            <span>
              Val Accuracy: <strong>{(iterMetrics.val_accuracy * 100).toFixed(2)}%</strong>
            </span>
          )}
          {iterMetrics.val_roc_auc != null && (
            <span>
              Val ROC-AUC: <strong>{iterMetrics.val_roc_auc.toFixed(4)}</strong>
            </span>
          )}
        </div>
      )}

      {hasIterRegression && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs">
          {iterMetrics.val_r2 != null && (
            <span>
              Val R²: <strong>{iterMetrics.val_r2.toFixed(4)}</strong>
            </span>
          )}
          {iterMetrics.test_r2 != null && (
            <span>
              Test R²: <strong>{iterMetrics.test_r2.toFixed(4)}</strong>
            </span>
          )}
          {iterMetrics.val_rmse != null && (
            <span>
              Val RMSE: <strong>{iterMetrics.val_rmse.toFixed(2)}</strong>
            </span>
          )}
          {iterMetrics.test_rmse != null && (
            <span>
              Test RMSE: <strong>{iterMetrics.test_rmse.toFixed(2)}</strong>
            </span>
          )}
        </div>
      )}

      {/* Error */}
      {iterMetrics.error && <div className="mt-2 text-xs text-destructive">Error: {iterMetrics.error}</div>}
    </div>
  )
}
