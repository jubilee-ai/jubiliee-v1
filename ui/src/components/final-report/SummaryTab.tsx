import { useMemo } from "react"
import type { TrainingAgentState, TaskPlanSummary } from "@/types/agent"
import type { Dataset as ApiDataset } from "@/lib/api"
import { resolveDatasetDisplayNames } from "@/lib/datasetDisplay"
import { formatDisplayDateTime, formatNumber, formatPercent } from "@/lib/utils"
import { Section, MetricBox, InfoRow } from "./shared"

interface SummaryTabProps {
  agentState: TrainingAgentState
  datasets?: ApiDataset[]
}

export function SummaryTab({ agentState, datasets }: SummaryTabProps) {
  const metrics = agentState.training_metrics
  const dataStep = agentState.audit_trace?.find((t) => t.step === "data_collection") as
    | Record<string, unknown>
    | undefined
  const featureStep = agentState.audit_trace?.find(
    (t) => t.step === "feature_engineering_executor"
  ) as Record<string, unknown> | undefined

  const bestIter = metrics?.best_iteration as Record<string, unknown> | undefined
  const testR2 = metrics?.test_r2 ?? (bestIter?.test_r2 as number | undefined)
  const testRmse = metrics?.test_rmse ?? (bestIter?.test_rmse as number | undefined)
  const testMae = metrics?.test_mae ?? (bestIter?.test_mae as number | undefined)
  const valR2 = metrics?.val_r2 ?? (bestIter?.val_r2 as number | undefined)

  const hasClassificationMetrics = metrics?.test_accuracy != null || metrics?.test_roc_auc != null
  const hasRegressionMetrics = testR2 != null || testRmse != null || testMae != null
  const modelTypeLabel = metrics?.model_type || agentState.selected_model || "N/A"
  const featureCount =
    agentState.feature_spec?.features.length ||
    (featureStep?.features_created as unknown[])?.length ||
    0
  const iterLabel = String(metrics?.num_iterations || 1)

  const taskDatasetRecap = useMemo(() => {
    const tp = agentState.task_plan as TaskPlanSummary | null | undefined
    if (!tp?.datasetLabels?.length) return ""
    const refs = tp.datasetRefs?.length ? tp.datasetRefs : (agentState.linked_datasets ?? [])
    return resolveDatasetDisplayNames(tp.datasetLabels, refs, datasets ?? []).join(", ")
  }, [agentState.task_plan, agentState.linked_datasets, datasets])

  return (
    <div className="space-y-8">
      {/* Hero metrics — test headline scores + run shape; model type once (or feature count when model is already in slot 2) */}
      <div className="flex flex-wrap gap-4 [&>div]:flex-[1_1_11rem] [&>div]:min-w-0 [&>div]:max-w-full [&>div>div:first-child]:text-[11px] [&>div>div:last-child]:text-md">
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
            <MetricBox label="Model" value={modelTypeLabel} />
          </>
        )}
        <MetricBox label="Iterations" value={iterLabel} />
        {hasClassificationMetrics || hasRegressionMetrics ? (
          <MetricBox label="Model" value={modelTypeLabel} />
        ) : (
          <MetricBox label="Features" value={featureCount > 0 ? String(featureCount) : "—"} />
        )}
      </div>
      {(hasClassificationMetrics || hasRegressionMetrics) && metrics?.model_name ? (
        <p className="text-xs text-muted-foreground -mt-4">
          Top row shows hold-out test metrics for the saved artifact{" "}
          <span className="text-foreground font-medium">{metrics.model_name}</span>
          {metrics.summary ? (
            <>
              . The training narrative below may include experiments that were not selected as that
              artifact.
            </>
          ) : (
            "."
          )}
        </p>
      ) : null}

      {/* Background task recap (assign-task flow) */}
      {agentState.task_plan && typeof agentState.task_plan === "object" && (
        <Section title="Assigned task recap">
          <div className="space-y-3 text-sm">
            <InfoRow label="Goal" value={agentState.task_plan.goal} />
            {taskDatasetRecap ? (
              <InfoRow label="Datasets" value={taskDatasetRecap} />
            ) : null}
            {agentState.task_completed_at ? (
              <InfoRow label="Finished at" value={formatDisplayDateTime(agentState.task_completed_at)} />
            ) : null}
            {agentState.task_plan.steps?.length ? (
              <div>
                <p className="text-muted-foreground text-xs font-medium uppercase tracking-wide mb-2">Plan</p>
                <ol className="list-decimal list-inside space-y-1 text-muted-foreground">
                  {agentState.task_plan.steps.map((step) => (
                    <li key={step}>{step}</li>
                  ))}
                </ol>
              </div>
            ) : null}
          </div>
        </Section>
      )}

      {/* Overview */}
      <Section title="Overview">
        <div className="space-y-3">
          <InfoRow label="Goal" value={agentState.goal} />
          <InfoRow label="Dataset" value={agentState.collected_dataset_ref} />
          <InfoRow label="Target Column" value={agentState.label_definition?.target_column} />
          <InfoRow label="Split Strategy" value={agentState.label_definition?.split_strategy} />
        </div>
      </Section>

      {/* Pointers to other tabs — avoids repeating hero numbers and Overview rows */}
      <Section title="Where to look next">
        <ul className="space-y-2.5 text-sm text-muted-foreground leading-relaxed">
          {hasClassificationMetrics || hasRegressionMetrics ? (
            <li className="flex gap-2">
              <span className="text-muted-foreground/50 shrink-0">→</span>
              <span>
                <span className="text-foreground font-medium">Metrics</span> has validation vs. test
                scores and each training iteration.
                {hasRegressionMetrics && valR2 != null && (
                  <> Validation R² there: {formatNumber(valR2, 4)}.</>
                )}
                {hasClassificationMetrics && metrics?.val_roc_auc != null && (
                  <> Validation ROC-AUC there: {formatNumber(metrics.val_roc_auc, 3)}.</>
                )}
              </span>
            </li>
          ) : (
            <li className="flex gap-2">
              <span className="text-muted-foreground/50 shrink-0">→</span>
              <span>
                <span className="text-foreground font-medium">Metrics</span> lists any logged scores
                and iteration history.
              </span>
            </li>
          )}
          {featureCount > 0 && (
            <li className="flex gap-2">
              <span className="text-muted-foreground/50 shrink-0">→</span>
              <span>
                <span className="text-foreground font-medium">Features</span> lists all{" "}
                {featureCount} engineered feature{featureCount !== 1 ? "s" : ""} and the data pipeline.
              </span>
            </li>
          )}
          <li className="flex gap-2">
            <span className="text-muted-foreground/50 shrink-0">→</span>
            <span>
              <span className="text-foreground font-medium">Analysis</span> shows correlations,
              distributions, and data-quality signals from feature selection.
            </span>
          </li>
          {dataStep?.rows != null && (
            <li className="flex gap-2">
              <span className="text-muted-foreground/50 shrink-0">→</span>
              <span>Collected dataset: {String(dataStep.rows)} rows (after data collection).</span>
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

      {/* Artifacts — name and paths only; model type is in the hero row */}
      {/* <Section title="Saved artifacts">
        <div className="space-y-3">
          <div>
            <div className="text-sm text-muted-foreground mb-1">Model artifact</div>
            <code className="text-sm bg-muted/50 px-3 py-2 rounded-lg block break-all">
              {metrics?.model_name || agentState.model_weights_path || "N/A"}
            </code>
          </div>
          {agentState.model_weights_path &&
            agentState.model_weights_path !== metrics?.model_name && (
              <div>
                <div className="text-sm text-muted-foreground mb-1">Weights path</div>
                <code className="text-sm bg-muted/50 px-3 py-2 rounded-lg block break-all">
                  {agentState.model_weights_path}
                </code>
              </div>
            )}
          {agentState.model_explanation && (
            <div>
              <div className="text-sm text-muted-foreground mb-1">Why this model</div>
              <p className="text-sm leading-relaxed bg-muted/30 px-3 py-2 rounded-lg">
                {agentState.model_explanation}
              </p>
            </div>
          )}
        </div>
      </Section> */}
    </div>
  )
}
