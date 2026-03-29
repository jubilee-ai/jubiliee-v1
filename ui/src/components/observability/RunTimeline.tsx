import { useMemo } from "react"
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
} from "recharts"
import type { TimelineEvent } from "@/lib/api"

interface RunTimelineProps {
  events: TimelineEvent[]
  onNodeClick?: (node: string) => void
}

const NODE_COLORS: Record<string, string> = {
  planner: "hsl(221, 83%, 53%)",
  dispatcher: "hsl(213, 27%, 50%)",
  data_collection: "hsl(142, 71%, 45%)",
  select_model: "hsl(262, 83%, 58%)",
  cleaning: "hsl(38, 92%, 50%)",
  label_split_definition: "hsl(330, 81%, 60%)",
  feature_selection_specification: "hsl(173, 58%, 39%)",
  feature_engineering_executor: "hsl(199, 89%, 48%)",
  training_approval: "hsl(25, 95%, 53%)",
  training: "hsl(346, 77%, 50%)",
  generate_report: "hsl(160, 60%, 45%)",
  evaluator: "hsl(280, 47%, 55%)",
}

function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`
  return `${(ms / 60_000).toFixed(1)}m`
}

function formatNodeName(name: string): string {
  return name.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())
}

function CustomTooltip({ active, payload }: { active?: boolean; payload?: Array<{ payload: { node: string; duration_ms: number; event_count: number } }> }) {
  if (!active || !payload?.[0]) return null
  const data = payload[0].payload
  return (
    <div className="rounded-lg border border-border bg-card px-3 py-2 shadow-lg">
      <p className="text-sm font-medium text-foreground">{formatNodeName(data.node)}</p>
      <p className="text-xs text-muted-foreground">
        Duration: {formatDuration(data.duration_ms)}
      </p>
      <p className="text-xs text-muted-foreground">Events: {data.event_count}</p>
    </div>
  )
}

export function RunTimeline({ events, onNodeClick }: RunTimelineProps) {
  const chartData = useMemo(() => {
    const stepCompleteEvents = events.filter(
      (e) => e.event_type === "step.complete" && e.node
    )

    const byNode = new Map<string, { total_ms: number; count: number }>()
    for (const ev of stepCompleteEvents) {
      const node = ev.node!
      const existing = byNode.get(node) || { total_ms: 0, count: 0 }
      existing.total_ms += ev.duration_ms || 0
      existing.count += 1
      byNode.set(node, existing)
    }

    if (byNode.size === 0) {
      const allNodes = new Map<string, number>()
      for (const ev of events) {
        if (ev.node) allNodes.set(ev.node, (allNodes.get(ev.node) || 0) + 1)
      }
      return Array.from(allNodes.entries()).map(([node, count]) => ({
        node,
        duration_ms: 0,
        event_count: count,
      }))
    }

    return Array.from(byNode.entries()).map(([node, data]) => ({
      node,
      duration_ms: data.total_ms,
      event_count: data.count,
    }))
  }, [events])

  if (chartData.length === 0) {
    return (
      <div className="flex items-center justify-center rounded-xl border border-border bg-card p-12 text-sm text-muted-foreground">
        No step events recorded yet. Run an experiment to see the timeline.
      </div>
    )
  }

  return (
    <div className="rounded-xl border border-border bg-card p-4">
      <h3 className="text-xs font-bold text-muted-foreground tracking-widest uppercase mb-4">
        Step Duration Waterfall
      </h3>
      <ResponsiveContainer width="100%" height={Math.max(200, chartData.length * 44)}>
        <BarChart data={chartData} layout="vertical" margin={{ left: 20, right: 20, top: 5, bottom: 5 }}>
          <XAxis
            type="number"
            tickFormatter={(v: number) => formatDuration(v)}
            tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))" }}
            axisLine={false}
            tickLine={false}
          />
          <YAxis
            type="category"
            dataKey="node"
            width={160}
            tickFormatter={formatNodeName}
            tick={{ fontSize: 12, fill: "hsl(var(--foreground))" }}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip content={<CustomTooltip />} cursor={{ fill: "hsl(var(--muted)/0.3)" }} />
          <Bar
            dataKey="duration_ms"
            radius={[0, 6, 6, 0]}
            cursor="pointer"
            onClick={(data) => onNodeClick?.((data as unknown as { node: string }).node)}
          >
            {chartData.map((entry) => (
              <Cell
                key={entry.node}
                fill={NODE_COLORS[entry.node] || "hsl(var(--primary))"}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
