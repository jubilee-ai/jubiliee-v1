import { useCallback, useEffect, useState } from "react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Badge } from "@/components/ui/badge"
import { Activity, GitBranch, BarChart3, ArrowLeft, Cpu, Coins } from "lucide-react"
import { Button } from "@/components/ui/button"
import {
  getExperimentTimeline,
  getObservabilitySummary,
  getLlmTelemetry,
  type ExperimentTimeline,
  type ObservabilitySummary,
  type LlmTelemetry,
} from "@/lib/api"
import { ExperimentTable } from "@/components/observability/ExperimentTable"
import { RunTimeline } from "@/components/observability/RunTimeline"
import { AgentGraph } from "@/components/observability/AgentGraph"

export function ObservabilityPage() {
  const [summary, setSummary] = useState<ObservabilitySummary | null>(null)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [timeline, setTimeline] = useState<ExperimentTimeline | null>(null)
  const [telemetry, setTelemetry] = useState<LlmTelemetry | null>(null)
  const [loading, setLoading] = useState(true)
  const [detailLoading, setDetailLoading] = useState(false)

  useEffect(() => {
    setLoading(true)
    getObservabilitySummary()
      .then(setSummary)
      .catch(() => setSummary(null))
      .finally(() => setLoading(false))
  }, [])

  const handleSelectExperiment = useCallback(async (id: string) => {
    setSelectedId(id)
    setDetailLoading(true)
    try {
      const [tl, lm] = await Promise.all([
        getExperimentTimeline(id),
        getLlmTelemetry(id),
      ])
      setTimeline(tl)
      setTelemetry(lm)
    } catch {
      setTimeline(null)
      setTelemetry(null)
    } finally {
      setDetailLoading(false)
    }
  }, [])

  const handleBack = useCallback(() => {
    setSelectedId(null)
    setTimeline(null)
    setTelemetry(null)
  }, [])

  const selectedName = summary?.experiments.find((e) => e.experiment_id === selectedId)?.name

  return (
    <div className="flex-1 overflow-auto">
      <div className="max-w-6xl mx-auto px-8 py-10">
        <span className="text-[10px] font-bold text-muted-foreground tracking-widest uppercase">
          Agent Intelligence
        </span>
        <div className="flex items-end justify-between mt-1">
          <div>
            <h1 className="font-headline text-3xl font-semibold text-foreground tracking-tight">
              Observability
            </h1>
            <p className="mt-2 text-muted-foreground text-sm leading-relaxed max-w-lg">
              End-to-end visibility into agent runs, step performance, and LLM cost tracking.
            </p>
          </div>
          {summary && summary.experiments.length > 0 && (
            <Badge variant="secondary" className="mb-1 tabular-nums">
              {summary.experiments.length} experiment{summary.experiments.length !== 1 ? "s" : ""}
            </Badge>
          )}
        </div>

        {/* Detail view for a selected experiment */}
        {selectedId ? (
          <div className="mt-6 space-y-6">
            <div className="flex items-center gap-3">
              <Button variant="ghost" size="sm" onClick={handleBack} className="gap-1.5 h-8">
                <ArrowLeft className="h-4 w-4" />
                All Experiments
              </Button>
              <div className="h-5 w-px bg-border" />
              <span className="text-sm font-medium text-foreground truncate">
                {selectedName || selectedId}
              </span>
            </div>

            {/* LLM Telemetry Summary Cards */}
            {telemetry?.configured && (
              <div className="grid grid-cols-3 gap-3">
                <div className="rounded-xl border border-border bg-card p-4">
                  <div className="flex items-center gap-2 text-xs text-muted-foreground">
                    <Coins className="h-3.5 w-3.5" />
                    Total LLM Cost
                  </div>
                  <div className="mt-1 text-xl font-semibold text-foreground tabular-nums">
                    ${telemetry.total_cost?.toFixed(4) ?? "--"}
                  </div>
                </div>
                <div className="rounded-xl border border-border bg-card p-4">
                  <div className="flex items-center gap-2 text-xs text-muted-foreground">
                    <Cpu className="h-3.5 w-3.5" />
                    Input Tokens
                  </div>
                  <div className="mt-1 text-xl font-semibold text-foreground tabular-nums">
                    {telemetry.total_input_tokens?.toLocaleString() ?? "--"}
                  </div>
                </div>
                <div className="rounded-xl border border-border bg-card p-4">
                  <div className="flex items-center gap-2 text-xs text-muted-foreground">
                    <Cpu className="h-3.5 w-3.5" />
                    Output Tokens
                  </div>
                  <div className="mt-1 text-xl font-semibold text-foreground tabular-nums">
                    {telemetry.total_output_tokens?.toLocaleString() ?? "--"}
                  </div>
                </div>
              </div>
            )}

            {detailLoading ? (
              <div className="rounded-xl bg-card p-8 text-center text-muted-foreground text-sm animate-pulse">
                Loading experiment timeline...
              </div>
            ) : timeline ? (
              <Tabs defaultValue="timeline" className="w-full">
                <TabsList className="bg-muted/50">
                  <TabsTrigger value="timeline" className="gap-1.5 text-xs">
                    <BarChart3 className="h-3.5 w-3.5" />
                    Timeline
                  </TabsTrigger>
                  <TabsTrigger value="graph" className="gap-1.5 text-xs">
                    <GitBranch className="h-3.5 w-3.5" />
                    Agent Graph
                  </TabsTrigger>
                  <TabsTrigger value="events" className="gap-1.5 text-xs">
                    <Activity className="h-3.5 w-3.5" />
                    Raw Events
                  </TabsTrigger>
                </TabsList>

                <TabsContent value="timeline" className="mt-4">
                  <RunTimeline events={timeline.events} />
                </TabsContent>

                <TabsContent value="graph" className="mt-4">
                  <AgentGraph events={timeline.events} />
                </TabsContent>

                <TabsContent value="events" className="mt-4">
                  <div className="rounded-xl border border-border bg-card overflow-hidden">
                    <table className="w-full text-xs">
                      <thead>
                        <tr className="border-b border-border/50 bg-muted/30">
                          <th className="text-left px-3 py-2 font-semibold text-muted-foreground uppercase tracking-wider">Type</th>
                          <th className="text-left px-3 py-2 font-semibold text-muted-foreground uppercase tracking-wider">Node</th>
                          <th className="text-right px-3 py-2 font-semibold text-muted-foreground uppercase tracking-wider">Duration</th>
                          <th className="text-right px-3 py-2 font-semibold text-muted-foreground uppercase tracking-wider">Time</th>
                        </tr>
                      </thead>
                      <tbody>
                        {timeline.events.map((ev) => (
                          <tr key={ev.id} className="border-b border-border/20 hover:bg-muted/20">
                            <td className="px-3 py-2">
                              <Badge variant="secondary" className="text-[10px] px-1.5 py-0 font-mono">
                                {ev.event_type}
                              </Badge>
                            </td>
                            <td className="px-3 py-2 text-foreground">{ev.node || "--"}</td>
                            <td className="px-3 py-2 text-right tabular-nums text-muted-foreground">
                              {ev.duration_ms != null ? `${ev.duration_ms}ms` : "--"}
                            </td>
                            <td className="px-3 py-2 text-right tabular-nums text-muted-foreground/60">
                              {ev.created_at ? new Date(ev.created_at).toLocaleTimeString() : "--"}
                            </td>
                          </tr>
                        ))}
                        {timeline.events.length === 0 && (
                          <tr>
                            <td colSpan={4} className="px-3 py-8 text-center text-muted-foreground">
                              No events recorded for this experiment.
                            </td>
                          </tr>
                        )}
                      </tbody>
                    </table>
                  </div>
                </TabsContent>
              </Tabs>
            ) : (
              <div className="rounded-xl bg-card p-8 text-center text-muted-foreground text-sm">
                No timeline data available for this experiment.
              </div>
            )}
          </div>
        ) : (
          /* Overview: experiment comparison table */
          <div className="mt-8">
            {loading ? (
              <div className="rounded-xl bg-card p-8 text-center text-muted-foreground text-sm animate-pulse">
                Loading experiments...
              </div>
            ) : (
              <ExperimentTable
                experiments={summary?.experiments ?? []}
                onSelectExperiment={handleSelectExperiment}
                selectedExperimentId={selectedId}
              />
            )}
          </div>
        )}
      </div>
    </div>
  )
}
