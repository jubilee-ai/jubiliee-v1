import { useState, useCallback, useEffect, useRef } from "react"
import type { Project, ProjectMetrics, ProjectSessionSnapshot } from "@/types/project"
import type { TrainingAgentState } from "@/types/agent"
import { uid } from "@/lib/utils"

const STORAGE_KEY = "jubilee_projects"

function loadProjects(): Project[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    return raw ? JSON.parse(raw) : []
  } catch {
    return []
  }
}

function saveProjects(projects: Project[]) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(projects))
}

function metricsFromState(state: TrainingAgentState): ProjectMetrics | null {
  const m = state.training_metrics
  if (!m) return null
  return {
    testAccuracy: m.test_accuracy ?? undefined,
    testRocAuc: m.test_roc_auc ?? undefined,
    testR2: m.test_r2 ?? undefined,
    testRmse: m.test_rmse ?? undefined,
    testMae: m.test_mae ?? undefined,
    modelName: m.model_name ?? undefined,
    modelType: m.model_type ?? undefined,
    numIterations: m.num_iterations ?? undefined,
  }
}

export interface UseProjectsReturn {
  projects: Project[]
  activeProjectId: string | null
  createProject: (goal: string, datasets?: string[]) => string
  createEmptyProject: () => string
  updateProjectFromState: (state: TrainingAgentState, isRunning: boolean) => void
  updateProjectSession: (id: string, session: ProjectSessionSnapshot) => void
  selectProject: (id: string) => void
  deleteProject: (id: string) => void
  activeProject: Project | null
  getProjectById: (id: string) => Project | null
}

export function useProjects(): UseProjectsReturn {
  const [projects, setProjects] = useState<Project[]>(loadProjects)
  const [activeProjectId, setActiveProjectId] = useState<string | null>(null)
  const prevGoalRef = useRef<string>("")

  useEffect(() => {
    saveProjects(projects)
  }, [projects])

  const createProject = useCallback((goal: string, datasets?: string[]): string => {
    const id = uid("proj")
    const project: Project = {
      id,
      goal,
      status: "running",
      createdAt: Date.now(),
      updatedAt: Date.now(),
      datasets: datasets || [],
      modelType: null,
      modelName: null,
      metrics: null,
      session: null,
    }
    setProjects((prev) => [project, ...prev])
    setActiveProjectId(id)
    prevGoalRef.current = goal
    return id
  }, [])

  const createEmptyProject = useCallback((): string => {
    const id = uid("proj")
    const now = Date.now()
    const project: Project = {
      id,
      goal: "New Project",
      status: "idle",
      createdAt: now,
      updatedAt: now,
      datasets: [],
      modelType: null,
      modelName: null,
      metrics: null,
      session: null,
    }
    setProjects((prev) => [project, ...prev])
    setActiveProjectId(id)
    prevGoalRef.current = ""
    return id
  }, [])

  const updateProjectFromState = useCallback((state: TrainingAgentState, isRunning: boolean) => {
    setActiveProjectId((currentActiveId) => {
      if (!currentActiveId) return currentActiveId

      setProjects((prev) =>
        prev.map((p) => {
          if (p.id !== currentActiveId) return p

          let status: Project["status"] = p.status
          if (state.error) {
            status = "error"
          } else if (state.training_metrics?.success || state.report_path) {
            status = "completed"
          } else if (isRunning) {
            status = "running"
          }

          return {
            ...p,
            status,
            updatedAt: Date.now(),
            modelType: state.selected_model || p.modelType,
            modelName: state.training_metrics?.model_name || p.modelName,
            metrics: metricsFromState(state) || p.metrics,
            goal: state.goal || p.goal,
          }
        })
      )

      return currentActiveId
    })
  }, [])

  const selectProject = useCallback((id: string) => {
    setActiveProjectId(id)
  }, [])

  const updateProjectSession = useCallback((id: string, session: ProjectSessionSnapshot) => {
    setProjects((prev) =>
      prev.map((p) => (p.id === id ? { ...p, session } : p))
    )
  }, [])

  const deleteProject = useCallback((id: string) => {
    setProjects((prev) => prev.filter((p) => p.id !== id))
    setActiveProjectId((current) => (current === id ? null : current))
  }, [])

  const activeProject = projects.find((p) => p.id === activeProjectId) || null
  const getProjectById = useCallback((id: string) => projects.find((p) => p.id === id) || null, [projects])

  return {
    projects,
    activeProjectId,
    createProject,
    createEmptyProject,
    updateProjectFromState,
    updateProjectSession,
    selectProject,
    deleteProject,
    activeProject,
    getProjectById,
  }
}
