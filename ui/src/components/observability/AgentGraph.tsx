import { useCallback, useMemo } from "react"
import {
  ReactFlow,
  Background,
  Controls,
  type Node,
  type Edge,
  type NodeTypes,
  Handle,
  Position,
} from "@xyflow/react"
import "@xyflow/react/dist/style.css"
import type { TimelineEvent } from "@/lib/api"
import { cn } from "@/lib/utils"

interface AgentGraphProps {
  events: TimelineEvent[]
  onNodeClick?: (node: string) => void
}

const STATUS_COLORS: Record<string, string> = {
  completed: "border-emerald-500 bg-emerald-500/10",
  running: "border-blue-500 bg-blue-500/10 animate-pulse",
  error: "border-red-500 bg-red-500/10",
  skipped: "border-muted-foreground/30 bg-muted/30",
  pending: "border-border bg-card",
  review: "border-amber-500 bg-amber-500/10",
}

function formatNodeName(name: string): string {
  const names: Record<string, string> = {
    planner: "Planner",
    dispatcher: "Dispatcher",
    data_collection: "Data Collection",
    select_model: "Model Selection",
    cleaning: "Cleaning",
    label_split_definition: "Label & Split",
    feature_selection_specification: "Feature Spec",
    feature_engineering_executor: "Feature Eng.",
    training_approval: "Training Config",
    training: "Training",
    generate_report: "Report",
    evaluator: "Evaluator",
  }
  return names[name] || name.replace(/_/g, " ")
}

function PipelineNode({ data }: { data: { label: string; nodeId: string; status: string; duration_ms: number | null; event_count: number } }) {
  const colorClass = STATUS_COLORS[data.status] || STATUS_COLORS.pending
  return (
    <>
      <Handle type="target" position={Position.Top} className="!bg-border !w-2 !h-2" />
      <div className={cn("rounded-lg border-2 px-3 py-2 min-w-[120px] text-center transition-colors", colorClass)}>
        <div className="text-xs font-semibold text-foreground">{data.label}</div>
        {data.duration_ms != null && data.duration_ms > 0 && (
          <div className="text-[10px] text-muted-foreground mt-0.5 tabular-nums">
            {data.duration_ms < 1000 ? `${data.duration_ms}ms` : `${(data.duration_ms / 1000).toFixed(1)}s`}
          </div>
        )}
        {data.event_count > 0 && (
          <div className="text-[10px] text-muted-foreground/60 mt-0.5">
            {data.event_count} event{data.event_count !== 1 ? "s" : ""}
          </div>
        )}
      </div>
      <Handle type="source" position={Position.Bottom} className="!bg-border !w-2 !h-2" />
    </>
  )
}

const nodeTypes: NodeTypes = {
  pipeline: PipelineNode,
}

const PIPELINE_STEPS = [
  "data_collection",
  "select_model",
  "cleaning",
  "label_split_definition",
  "feature_selection_specification",
  "feature_engineering_executor",
  "training_approval",
  "training",
  "generate_report",
]

export function AgentGraph({ events, onNodeClick }: AgentGraphProps) {
  const { nodeStatuses, nodeDurations, nodeCounts } = useMemo(() => {
    const statuses: Record<string, string> = {}
    const durations: Record<string, number> = {}
    const counts: Record<string, number> = {}

    for (const ev of events) {
      if (!ev.node) continue
      counts[ev.node] = (counts[ev.node] || 0) + 1

      if (ev.event_type === "step.complete") {
        statuses[ev.node] = "completed"
        if (ev.duration_ms) {
          durations[ev.node] = (durations[ev.node] || 0) + ev.duration_ms
        }
      } else if (ev.event_type === "step.skipped") {
        statuses[ev.node] = "skipped"
      } else if (ev.event_type === "review.required") {
        statuses[ev.node] = "review"
      } else if (ev.event_type === "error") {
        statuses[ev.node] = "error"
      }
    }
    return { nodeStatuses: statuses, nodeDurations: durations, nodeCounts: counts }
  }, [events])

  const nodes: Node[] = useMemo(() => {
    const result: Node[] = []
    const y_start = 0
    const x_center = 300

    result.push({
      id: "planner",
      type: "pipeline",
      position: { x: x_center - 60, y: y_start },
      data: { label: "Planner", nodeId: "planner", status: nodeStatuses.planner || "pending", duration_ms: nodeDurations.planner, event_count: nodeCounts.planner || 0 },
    })

    result.push({
      id: "dispatcher",
      type: "pipeline",
      position: { x: x_center - 60, y: 80 },
      data: { label: "Dispatcher", nodeId: "dispatcher", status: nodeStatuses.dispatcher || "pending", duration_ms: nodeDurations.dispatcher, event_count: nodeCounts.dispatcher || 0 },
    })

    const stepsPerRow = 3
    const xSpacing = 170
    const ySpacing = 80
    const startX = x_center - ((stepsPerRow - 1) * xSpacing) / 2 - 60

    PIPELINE_STEPS.forEach((step, i) => {
      const row = Math.floor(i / stepsPerRow)
      const col = i % stepsPerRow
      result.push({
        id: step,
        type: "pipeline",
        position: { x: startX + col * xSpacing, y: 180 + row * ySpacing },
        data: {
          label: formatNodeName(step),
          nodeId: step,
          status: nodeStatuses[step] || "pending",
          duration_ms: nodeDurations[step],
          event_count: nodeCounts[step] || 0,
        },
      })
    })

    const evalY = 180 + Math.ceil(PIPELINE_STEPS.length / stepsPerRow) * ySpacing + 20
    result.push({
      id: "evaluator",
      type: "pipeline",
      position: { x: x_center - 60, y: evalY },
      data: { label: "Evaluator", nodeId: "evaluator", status: nodeStatuses.evaluator || "pending", duration_ms: nodeDurations.evaluator, event_count: nodeCounts.evaluator || 0 },
    })

    return result
  }, [nodeStatuses, nodeDurations, nodeCounts])

  const edges: Edge[] = useMemo(() => {
    const result: Edge[] = [
      { id: "planner-dispatcher", source: "planner", target: "dispatcher", animated: true, style: { stroke: "hsl(var(--border))" } },
    ]

    for (const step of PIPELINE_STEPS) {
      result.push({
        id: `dispatcher-${step}`,
        source: "dispatcher",
        target: step,
        style: { stroke: "hsl(var(--border))", strokeDasharray: "4 2" },
      })
      result.push({
        id: `${step}-evaluator`,
        source: step,
        target: "evaluator",
        style: { stroke: "hsl(var(--border))", strokeDasharray: "4 2" },
      })
    }

    result.push({
      id: "evaluator-dispatcher",
      source: "evaluator",
      target: "dispatcher",
      label: "continue",
      animated: true,
      style: { stroke: "hsl(142, 71%, 45%)" },
    })
    result.push({
      id: "evaluator-planner",
      source: "evaluator",
      target: "planner",
      label: "replan",
      style: { stroke: "hsl(38, 92%, 50%)" },
    })

    return result
  }, [])

  const handleNodeClick = useCallback(
    (_: unknown, node: Node) => {
      onNodeClick?.(node.id)
    },
    [onNodeClick]
  )

  return (
    <div className="rounded-xl border border-border bg-card" style={{ height: 600 }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        onNodeClick={handleNodeClick}
        fitView
        fitViewOptions={{ padding: 0.2 }}
        proOptions={{ hideAttribution: true }}
        minZoom={0.3}
        maxZoom={1.5}
      >
        <Background gap={16} size={1} />
        <Controls showInteractive={false} />
      </ReactFlow>
    </div>
  )
}
