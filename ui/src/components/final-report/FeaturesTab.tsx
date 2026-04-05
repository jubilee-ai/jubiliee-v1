import { Badge } from "@/components/ui/badge"
import type { TrainingAgentState } from "@/types/agent"
import { Section, InfoBox, PipelineRow } from "./shared"
import { renderValue } from "./utils"

interface FeaturesTabProps {
  agentState: TrainingAgentState
}

export function FeaturesTab({ agentState }: FeaturesTabProps) {
  const features = agentState.feature_spec?.features || []
  const dataSummary = agentState.training_plan?.data_summary as
    | { n_features?: number; train_rows?: number }
    | undefined
  const nModelInputs =
    typeof dataSummary?.n_features === "number" && Number.isFinite(dataSummary.n_features)
      ? dataSummary.n_features
      : null

  return (
    <div className="space-y-8">
      {/* Features List — logical definitions; encoding expands to more model columns */}
      <Section
        title={`Feature definitions (${features.length})`}
        description={
          nModelInputs != null ? (
            <>
              Logical features used for the final spec. After one-hot and similar transforms, the model typically sees{" "}
              <span className="text-foreground font-medium tabular-nums">{nModelInputs}</span> input columns
              {dataSummary?.train_rows != null ? (
                <>
                  {" "}
                  (training rows: {Number(dataSummary.train_rows).toLocaleString()})
                </>
              ) : null}
              .
            </>
          ) : (
            <>
              Logical feature definitions. One-hot and similar encodings expand to more columns than this list — see the
              feature engineering and training steps for matrix shapes.
            </>
          )
        }
      >
        <div className="space-y-2">
          {features.length > 0 ? (
            features.map((feature, i) => (
              <div key={i} className="p-4 rounded-xl bg-muted/30">
                <div className="flex items-center gap-2 flex-wrap mb-2">
                  <span className="font-medium">{renderValue(feature.name)}</span>
                  <Badge variant="outline" className="text-xs font-normal">
                    {renderValue(feature.encoding)}
                  </Badge>
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
            {agentState.label_definition.forbidden_columns &&
              agentState.label_definition.forbidden_columns.length > 0 && (
                <div className="col-span-2">
                  <div className="text-sm text-muted-foreground mb-1">Forbidden Columns</div>
                  <div className="flex flex-wrap gap-2">
                    {agentState.label_definition.forbidden_columns.map((col, i) => (
                      <Badge key={i} variant="secondary" className="text-xs font-normal">
                        {col}
                      </Badge>
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
            {agentState.cleaning_transformations.map((t, i) => (
              <TransformationRow key={i} transformation={t} />
            ))}
          </div>
        </Section>
      )}
    </div>
  )
}

/**
 * Renders a single cleaning transformation row
 */
function TransformationRow({ transformation }: { transformation: unknown }) {
  if (typeof transformation === "object" && transformation !== null) {
    const transform = transformation as Record<string, unknown>
    const toolName = transform.tool || transform.op || transform.operation || "transform"
    const args = transform.args as Record<string, unknown> | undefined
    const result = transform.result as string | undefined

    // Extract columns from args
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

    // Extract result info
    let resultInfo = ""
    if (result) {
      const match = String(result).match(/→\s*`([^`]+)`\s*(.*)/)
      if (match) {
        resultInfo = match[2] || ""
      }
    }

    return (
      <div className="flex items-start gap-3 py-2 px-3 bg-muted/30 rounded-lg">
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
    <code className="text-xs bg-muted/50 rounded-lg px-3 py-2 block font-mono">
      {String(transformation)}
    </code>
  )
}
