import type { TrainingAgentState } from "@/types/agent"
import { Section } from "./shared"
import {
  FeatureAnalysisPanels,
  getFeatureSelectionKeyStats,
  hasFeatureAnalysisContent,
} from "./FeatureAnalysisPanels"

interface AnalysisTabProps {
  agentState: TrainingAgentState
}

export function AnalysisTab({ agentState }: AnalysisTabProps) {
  const keyStats = getFeatureSelectionKeyStats(agentState)

  if (!hasFeatureAnalysisContent(keyStats)) {
    return (
      <div className="space-y-6">
        <Section title="Feature Analysis">
          <p className="text-xs text-muted-foreground leading-relaxed">
            No feature analysis data available. This data is generated during the feature selection step.
          </p>
        </Section>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <FeatureAnalysisPanels keyStats={keyStats} density="full" />
    </div>
  )
}
