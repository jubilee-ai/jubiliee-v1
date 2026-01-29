// Main component export
export { FinalReport } from "./FinalReport"

// Tab components (for potential standalone use or testing)
export { SummaryTab } from "./SummaryTab"
export { MetricsTab } from "./MetricsTab"
export { AnalysisTab } from "./AnalysisTab"
export { FeaturesTab } from "./FeaturesTab"
export { TraceTab } from "./TraceTab"

// Chart components
export { CorrelationBar } from "./CorrelationBar"

// Shared UI components
export {
  Section,
  MetricBox,
  MetricRow,
  InfoRow,
  InfoBox,
  PipelineRow,
} from "./shared"

// Utility functions
export {
  getIterationMetrics,
  renderValue,
  generateTextReport,
  generateJsonReport,
} from "./utils"
