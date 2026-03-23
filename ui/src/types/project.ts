import type { ChatMessage, StepInfo, TrainingAgentState } from "@/types/agent"

export interface ProjectSessionSnapshot {
  agentState: TrainingAgentState
  steps: StepInfo[]
  messages: ChatMessage[]
  progress: number
  threadId: string | null
  chatThreadId: string | null
}

export interface Project {
  id: string
  goal: string
  status: "idle" | "running" | "completed" | "error"
  createdAt: number
  updatedAt: number
  datasets: string[]
  modelType: string | null
  modelName: string | null
  metrics: ProjectMetrics | null
  session: ProjectSessionSnapshot | null
}

export interface ProjectMetrics {
  testAccuracy?: number
  testRocAuc?: number
  testR2?: number
  testRmse?: number
  testMae?: number
  modelName?: string
  modelType?: string
  numIterations?: number
}
