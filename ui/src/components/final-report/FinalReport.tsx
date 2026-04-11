import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ScrollArea } from "@/components/ui/scroll-area"
import { X, Download, Copy, Check } from "lucide-react"
import type { Dataset as ApiDataset } from "@/lib/api"
import type { TrainingAgentState, StepInfo } from "@/types/agent"
import { SummaryTab } from "./SummaryTab"
import { MetricsTab } from "./MetricsTab"
import { AnalysisTab } from "./AnalysisTab"
import { generateTextReport, generateJsonReport } from "./utils"

interface FinalReportProps {
  agentState: TrainingAgentState
  steps: StepInfo[]
  onClose: () => void
  datasets?: ApiDataset[]
}

export function FinalReport({ agentState, steps, onClose, datasets }: FinalReportProps) {
  const [activeTab, setActiveTab] = useState("summary")
  const [copied, setCopied] = useState(false)
  const metrics = agentState.training_metrics

  const hasResults =
    metrics?.success ||
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
    return <NoResultsModal agentState={agentState} onClose={onClose} />
  }

  return (
    <div className="fixed inset-0 z-50 bg-background/80 backdrop-blur-sm flex items-center justify-center p-4 sm:p-6">
      <div className="w-full max-w-6xl h-[min(90vh,960px)] bg-background rounded-2xl border shadow-2xl flex flex-col overflow-hidden">
        {/* Header */}
        <ReportHeader
          modelName={metrics?.model_name || agentState.model_weights_path || "Model training complete"}
          copied={copied}
          onCopy={copyReport}
          onDownload={downloadReport}
          onClose={onClose}
        />

        {/* Tabs */}
        <Tabs value={activeTab} onValueChange={setActiveTab} className="flex-1 flex flex-col min-h-0">
          <div className="px-5 sm:px-8 pt-2 border-b">
            <TabsList className="bg-transparent p-0 h-auto gap-5 sm:gap-8">
              <ReportTabTrigger value="summary">Summary</ReportTabTrigger>
              <ReportTabTrigger value="analysis">Analysis</ReportTabTrigger>
              <ReportTabTrigger value="metrics">Metrics</ReportTabTrigger>
            </TabsList>
          </div>

          <ScrollArea className="flex-1">
            <div className="px-5 py-5 sm:px-10 sm:py-6">
              <TabsContent value="summary" className="mt-0">
                <SummaryTab agentState={agentState} datasets={datasets} />
              </TabsContent>
              <TabsContent value="analysis" className="mt-0">
                <div className="text-xs leading-relaxed">
                  <AnalysisTab agentState={agentState} />
                </div>
              </TabsContent>
              <TabsContent value="metrics" className="mt-0">
                <MetricsTab agentState={agentState} />
              </TabsContent>
            </div>
          </ScrollArea>
        </Tabs>
      </div>
    </div>
  )
}

/**
 * Report header with title and action buttons
 */
function ReportHeader({
  modelName,
  copied,
  onCopy,
  onDownload,
  onClose,
}: {
  modelName: string
  copied: boolean
  onCopy: () => void
  onDownload: () => void
  onClose: () => void
}) {
  return (
    <div className="flex items-center justify-between px-5 sm:px-8 py-3.5 border-b gap-4">
      <div className="min-w-0">
        <h2 className="text-xl font-semibold tracking-tight font-headline">Training Report</h2>
        <p className="text-caption sm:text-sm text-muted-foreground mt-0.5 break-words [overflow-wrap:anywhere] max-w-full">
          {modelName}
        </p>
      </div>
      <div className="flex items-center gap-2">
        <Button variant="ghost" size="sm" onClick={onCopy} className="h-8 gap-2">
          {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
          <span className="hidden sm:inline">{copied ? "Copied" : "Copy"}</span>
        </Button>
        <Button variant="ghost" size="sm" onClick={onDownload} className="h-8 gap-2">
          <Download className="h-4 w-4" />
          <span className="hidden sm:inline">Download</span>
        </Button>
        <Button variant="ghost" size="sm" onClick={onClose} className="h-8 w-8 p-0">
          <X className="h-4 w-4" />
        </Button>
      </div>
    </div>
  )
}

/**
 * Styled tab trigger for the report tabs
 */
function ReportTabTrigger({ value, children }: { value: string; children: React.ReactNode }) {
  return (
    <TabsTrigger
      value={value}
      className="bg-transparent px-0 pb-3 pt-0 rounded-none border-b-2 border-transparent data-[state=active]:border-foreground data-[state=active]:bg-transparent data-[state=active]:shadow-none"
    >
      {children}
    </TabsTrigger>
  )
}

/**
 * Modal shown when there are no results yet
 */
function NoResultsModal({
  agentState,
  onClose,
}: {
  agentState: TrainingAgentState
  onClose: () => void
}) {
  return (
    <div className="fixed inset-0 z-50 bg-background/80 backdrop-blur-sm flex items-center justify-center p-6">
      <div className="bg-background rounded-2xl border shadow-2xl p-6 max-w-md">
        <h2 className="text-xl font-semibold mb-2">Training Report</h2>
        <p className="text-muted-foreground mb-4">
          Training has not completed yet or no results available.
        </p>
        <p className="text-sm text-muted-foreground mb-2">
          Current step: {agentState.current_step || "unknown"}
        </p>
        {agentState.error && (
          <p className="text-sm text-destructive mb-4">Error: {agentState.error}</p>
        )}
        <Button onClick={onClose}>Close</Button>
      </div>
    </div>
  )
}
