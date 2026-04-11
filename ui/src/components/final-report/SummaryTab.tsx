import { useMemo } from "react"
import type { TrainingAgentState, TaskPlanSummary } from "@/types/agent"
import type { Dataset as ApiDataset } from "@/lib/api"
import { resolveDatasetDisplayNames } from "@/lib/datasetDisplay"
import { formatDisplayDateTime, formatNumber } from "@/lib/utils"
import { Section, MetricBox, InfoRow } from "./shared"
import { getHeroMetrics, mergeTrainingMetricSources, pickNumber, resolveModelFamily } from "./metricDefs"

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

  const merged = mergeTrainingMetricSources(metrics)
  const modelFamily = resolveModelFamily(metrics, agentState)
  const hasPrimaryMetrics =
    modelFamily === "classification"
      ? pickNumber(merged, "test_accuracy") != null || pickNumber(merged, "test_roc_auc") != null
      : modelFamily === "regression"
        ? pickNumber(merged, "test_r2") != null ||
          pickNumber(merged, "test_rmse") != null ||
          pickNumber(merged, "test_mae") != null
        : pickNumber(merged, "silhouette_score") != null ||
          pickNumber(merged, "davies_bouldin") != null ||
          pickNumber(merged, "inertia") != null ||
          pickNumber(merged, "reconstruction_loss") != null
  const heroMetrics = getHeroMetrics(metrics, agentState)
  const valR2 = metrics?.val_r2 ?? pickNumber(merged, "val_r2")
  const modelTypeLabel = metrics?.model_type || agentState.selected_model || "N/A"
  const planDs = agentState.training_plan?.data_summary as { n_features?: number } | undefined
  const nFeaturesFromPlan =
    typeof planDs?.n_features === "number" && Number.isFinite(planDs.n_features) ? planDs.n_features : null
  const featureCount =
    nFeaturesFromPlan ??
    (agentState.feature_spec?.features.length ||
      (featureStep?.features_created as unknown[])?.length ||
      0)
  const iterLabel = String(metrics?.num_iterations || 1)
  const isUnsupervisedFlow =
    agentState.selected_model === "unsupervised" ||
    agentState.label_definition?.split_strategy === "none"

  const taskDatasetRecap = useMemo(() => {
    const tp = agentState.task_plan as TaskPlanSummary | null | undefined
    if (!tp?.datasetLabels?.length) return ""
    const refs = tp.datasetRefs?.length ? tp.datasetRefs : (agentState.linked_datasets ?? [])
    return resolveDatasetDisplayNames(tp.datasetLabels, refs, datasets ?? []).join(", ")
  }, [agentState.task_plan, agentState.linked_datasets, datasets])

  const goalText = agentState.goal?.trim()

  return (
    <div className="space-y-6">
      {goalText ? (
        <div className="rounded-xl border border-border/50 bg-muted/15 px-4 py-3 sm:px-5">
          <p className="text-caption font-semibold uppercase tracking-wide text-muted-foreground mb-1.5">Goal</p>
          <p className="text-sm text-foreground leading-relaxed [overflow-wrap:anywhere]">{goalText}</p>
        </div>
      ) : null}

      {/* Hero metrics — test headline scores + run shape; model type once (or feature count when model is already in slot 2) */}
      <div className="flex flex-wrap gap-3 sm:gap-4 [&>div]:flex-[1_1_10rem] [&>div]:min-w-0 [&>div]:max-w-full [&>div>div:first-child]:text-caption [&>div>div:last-child]:text-sm">
        {heroMetrics.map((h) => (
          <MetricBox key={h.label} label={h.label} value={h.value} highlight={h.highlight} />
        ))}
        <MetricBox label="Iterations" value={iterLabel} />
        {hasPrimaryMetrics ? (
          <MetricBox label="Model" value={modelTypeLabel} />
        ) : (
          <MetricBox label="Features" value={featureCount > 0 ? String(featureCount) : "—"} />
        )}
      </div>
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
          <InfoRow label="Dataset" value={agentState.collected_dataset_ref} />
          {!isUnsupervisedFlow ? (
            <InfoRow label="Target Column" value={agentState.label_definition?.target_column || "—"} />
          ) : (
            <InfoRow label="Target Column" value="None (unsupervised)" />
          )}
          <InfoRow
            label="Split Strategy"
            value={
              isUnsupervisedFlow
                ? "None — full dataset for training"
                : agentState.label_definition?.split_strategy
            }
          />
        </div>
      </Section>

      {/* Pointers to other tabs — avoids repeating hero numbers and Overview rows */}
      <Section title="Where to look next">
        <ul className="space-y-2.5 text-sm text-muted-foreground leading-relaxed">
          {modelFamily === "classification" || modelFamily === "regression" ? (
            <li className="flex gap-2">
              <span className="text-muted-foreground/50 shrink-0">→</span>
              <span>
                <span className="text-foreground font-medium">Metrics</span> has validation vs. test
                scores and each training iteration.
                {modelFamily === "regression" && valR2 != null && (
                  <> Validation R² there: {formatNumber(valR2, 4)}.</>
                )}
                {modelFamily === "classification" && metrics?.val_roc_auc != null && (
                  <> Validation ROC-AUC there: {formatNumber(metrics.val_roc_auc, 3)}.</>
                )}
              </span>
            </li>
          ) : modelFamily === "unsupervised" ? (
            <li className="flex gap-2">
              <span className="text-muted-foreground/50 shrink-0">→</span>
              <span>
                <span className="text-foreground font-medium">Metrics</span> lists clustering /
                unsupervised scores (silhouette, Davies-Bouldin, inertia, etc.) and each iteration.
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
                <span className="text-foreground font-medium">Features</span> in step details lists all{" "}
                {featureCount} engineered feature{featureCount !== 1 ? "s" : ""}, engineering details, and any feature
                experiments.
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
