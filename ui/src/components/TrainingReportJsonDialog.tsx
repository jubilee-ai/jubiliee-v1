import { useEffect, useState } from "react"
import { Loader2 } from "lucide-react"
import { toast } from "sonner"
import type { TrainingAgentState } from "@/types/agent"
import { getTrainedModelReport } from "@/lib/api"
import { reportJsonToAgentState } from "@/lib/trainingReport"
import { FinalReport } from "@/components/final-report"

interface RegistryFinalReportProps {
  modelName: string
  onClose: () => void
}

/**
 * Fetches a stored report JSON for a registry model, converts it to
 * `TrainingAgentState`, and renders the same `FinalReport` used in the chat.
 */
export function RegistryFinalReport({ modelName, onClose }: RegistryFinalReportProps) {
  const [agentState, setAgentState] = useState<TrainingAgentState | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(false)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setError(false)
    setAgentState(null)
    ;(async () => {
      try {
        const data = await getTrainedModelReport(modelName)
        if (!cancelled) setAgentState(reportJsonToAgentState(data))
      } catch {
        if (!cancelled) {
          setError(true)
          toast.error("Report not available")
        }
      } finally {
        if (!cancelled) setLoading(false)
      }
    })()
    return () => { cancelled = true }
  }, [modelName])

  if (loading) {
    return (
      <div className="fixed inset-0 z-50 bg-background/80 backdrop-blur-sm flex items-center justify-center">
        <div className="flex items-center gap-3 text-muted-foreground">
          <Loader2 className="h-5 w-5 animate-spin" />
          Loading report…
        </div>
      </div>
    )
  }

  if (error || !agentState) {
    return (
      <div className="fixed inset-0 z-50 bg-background/80 backdrop-blur-sm flex items-center justify-center p-6">
        <div className="bg-background rounded-2xl border shadow-2xl p-6 max-w-md">
          <h2 className="text-xl font-semibold mb-2">Training Report</h2>
          <p className="text-muted-foreground mb-4">Report not available for this model.</p>
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-lg bg-foreground text-background text-sm font-medium"
          >
            Close
          </button>
        </div>
      </div>
    )
  }

  return <FinalReport agentState={agentState} steps={[]} onClose={onClose} />
}
