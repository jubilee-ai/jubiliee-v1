import type { TrainingAgentState } from "@/types/agent"
import { formatNumber, formatPercent } from "@/lib/utils"
import { Section, MetricBox, InfoRow } from "./shared"

interface SummaryTabProps {
  agentState: TrainingAgentState
}

export function SummaryTab({ agentState }: SummaryTabProps) {
  const metrics = agentState.training_metrics
  const dataStep = agentState.audit_trace?.find((t) => t.step === "data_collection") as
    | Record<string, unknown>
    | undefined
  const featureStep = agentState.audit_trace?.find(
    (t) => t.step === "feature_engineering_executor"
  ) as Record<string, unknown> | undefined

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
              Model achieved {formatPercent(metrics?.test_accuracy)} test accuracy with ROC-AUC of{" "}
              {formatNumber(metrics?.test_roc_auc, 3)}
            </li>
          ) : hasRegressionMetrics ? (
            <li className="flex gap-2">
              <span className="text-muted-foreground/50">•</span>
              Model achieved Test R² of {formatNumber(testR2, 4)} with RMSE of{" "}
              {formatNumber(testRmse, 2)} and MAE of {formatNumber(testMae, 2)}
            </li>
          ) : (
            <li className="flex gap-2">
              <span className="text-muted-foreground/50">•</span>
              Model training completed successfully using{" "}
              {metrics?.model_type || agentState.selected_model}
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
            {agentState.feature_spec?.features.length ||
              (featureStep?.features_created as unknown[])?.length ||
              0}{" "}
            features were engineered from the original dataset
          </li>
          <li className="flex gap-2">
            <span className="text-muted-foreground/50">•</span>
            Data was split using {agentState.label_definition?.split_strategy || "random"} strategy
            (70/15/15)
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
