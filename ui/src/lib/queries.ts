/**
 * TanStack Query hooks for experiment server state.
 */

import {
  useQuery,
  useMutation,
  useQueryClient,
  type UseQueryResult,
} from "@tanstack/react-query"
import {
  listExperiments,
  getExperiment,
  deleteExperiment,
  updateExperiment,
  getDatasets,
  getDatasetPreview,
  uploadDataset,
  getModelRiskInventory,
  getModelRiskDetail,
  getModelRiskDashboard,
  postModelRiskApproval,
  patchModelRiskProfile,
  type ExperimentSummary,
  type ExperimentDetail,
  type Dataset,
  type DatasetPreview,
  type ModelRiskInventoryItem,
  type ModelRiskProfileDetail,
  type ModelRiskDashboard,
  type ModelRiskPortalRole,
  type ModelRiskApprovalRequest,
  type ModelRiskProfilePatch,
} from "@/lib/api"

function queryDebug(event: string, payload?: Record<string, unknown>) {
  const ts = new Date().toISOString()
  if (payload) {
    console.log(`[query:experiments][${ts}] ${event}`, payload)
    return
  }
  console.log(`[query:experiments][${ts}] ${event}`)
}

export const experimentKeys = {
  all: ["experiments"] as const,
  list: () => [...experimentKeys.all, "list"] as const,
  detail: (id: string) => [...experimentKeys.all, "detail", id] as const,
}

export const datasetKeys = {
  all: ["datasets"] as const,
  list: () => [...datasetKeys.all, "list"] as const,
  preview: (name: string, limit: number) =>
    [...datasetKeys.all, "preview", name, limit] as const,
}

export const modelRiskKeys = {
  all: ["model-risk"] as const,
  list: () => [...modelRiskKeys.all, "inventory"] as const,
  detail: (name: string) => [...modelRiskKeys.all, "detail", name] as const,
  dashboard: (portal: ModelRiskPortalRole) =>
    [...modelRiskKeys.all, "dashboard", portal] as const,
}

export function useExperimentsList(options?: { enabled?: boolean }): UseQueryResult<ExperimentSummary[], Error> {
  return useQuery<ExperimentSummary[], Error>({
    queryKey: experimentKeys.list(),
    queryFn: async () => {
      queryDebug("list-query-start")
      try {
        const data = await listExperiments()
        queryDebug("list-query-success", { count: data.length })
        return data
      } catch (err) {
        queryDebug("list-query-error", { error: String(err) })
        throw err
      }
    },
    enabled: options?.enabled !== false,
    staleTime: 30_000,
  })
}

export function useExperimentDetailQuery(
  id: string | null,
  options?: { enabled?: boolean },
): UseQueryResult<ExperimentDetail, Error> {
  const enabled = !!id && (options?.enabled !== false)
  return useQuery<ExperimentDetail, Error>({
    queryKey: experimentKeys.detail(id ?? "__none__"),
    queryFn: async () => {
      queryDebug("detail-query-start", { id })
      try {
        const data = await getExperiment(id!)
        queryDebug("detail-query-success", {
          id: data.id,
          chatHistoryCount: Array.isArray(data.chat_history) ? data.chat_history.length : null,
          hasTrainingState: !!data.training_state,
        })
        return data
      } catch (err) {
        queryDebug("detail-query-error", { id, error: String(err) })
        throw err
      }
    },
    enabled,
    staleTime: 0,
    gcTime: 5 * 60 * 1000,
    /** Avoid overwriting local chat with a stale GET while the user is working in the lab. */
    refetchOnWindowFocus: false,
  })
}

export function useDeleteExperimentMutation() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (id: string) => deleteExperiment(id),
    onSuccess: (_void, id) => {
      queryClient.invalidateQueries({ queryKey: experimentKeys.list() })
      queryClient.removeQueries({ queryKey: experimentKeys.detail(id) })
    },
  })
}

export function useUpdateExperimentMutation() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: ({
      id,
      updates,
    }: {
      id: string
      updates: Parameters<typeof updateExperiment>[1]
    }) => updateExperiment(id, updates),
    onSuccess: (_void, { id }) => {
      queryClient.invalidateQueries({ queryKey: experimentKeys.list() })
      queryClient.invalidateQueries({ queryKey: experimentKeys.detail(id) })
    },
  })
}

export function useDatasetsQuery(options?: { enabled?: boolean }) {
  return useQuery<Dataset[], Error>({
    queryKey: datasetKeys.list(),
    queryFn: () => getDatasets(),
    enabled: options?.enabled !== false,
    staleTime: 30_000,
  })
}

const DEFAULT_PREVIEW_LIMIT = 25

export function useDatasetPreviewQuery(
  name: string | null,
  options?: { enabled?: boolean; limit?: number },
): UseQueryResult<DatasetPreview, Error> {
  const limit = options?.limit ?? DEFAULT_PREVIEW_LIMIT
  const enabled = !!name && (options?.enabled !== false)
  return useQuery<DatasetPreview, Error>({
    queryKey: datasetKeys.preview(name ?? "__none__", limit),
    queryFn: () => getDatasetPreview(name!, { limit }),
    enabled,
    staleTime: 60_000,
  })
}

export function useUploadDatasetMutation() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (vars: { file: File; name?: string; description?: string }) =>
      uploadDataset(vars.file, { name: vars.name, description: vars.description }),
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: datasetKeys.all })
    },
  })
}

export function useModelRiskList(options?: { enabled?: boolean }) {
  return useQuery<ModelRiskInventoryItem[], Error>({
    queryKey: modelRiskKeys.list(),
    queryFn: () => getModelRiskInventory(),
    enabled: options?.enabled !== false,
    staleTime: 20_000,
  })
}

export function useModelRiskDetail(
  modelName: string | null,
  options?: { enabled?: boolean },
) {
  const enabled = !!modelName?.trim() && (options?.enabled !== false)
  return useQuery<ModelRiskProfileDetail, Error>({
    queryKey: modelRiskKeys.detail(modelName ?? "__none__"),
    queryFn: () => getModelRiskDetail(modelName!.trim()),
    enabled,
    staleTime: 10_000,
  })
}

export function useModelRiskDashboard(
  portal: ModelRiskPortalRole,
  options?: { enabled?: boolean },
) {
  return useQuery<ModelRiskDashboard, Error>({
    queryKey: modelRiskKeys.dashboard(portal),
    queryFn: () => getModelRiskDashboard(portal),
    enabled: options?.enabled !== false,
    staleTime: 20_000,
  })
}

export function useModelRiskApprovalMutation() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (vars: { modelName: string; body: ModelRiskApprovalRequest }) =>
      postModelRiskApproval(vars.modelName, vars.body),
    onSuccess: async (_void, vars) => {
      await queryClient.invalidateQueries({ queryKey: modelRiskKeys.list() })
      await queryClient.invalidateQueries({
        queryKey: modelRiskKeys.detail(vars.modelName),
      })
      await queryClient.invalidateQueries({ queryKey: modelRiskKeys.all })
    },
  })
}

export function useModelRiskProfilePatchMutation() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (vars: { modelName: string; body: ModelRiskProfilePatch }) =>
      patchModelRiskProfile(vars.modelName, vars.body),
    onSuccess: async (_void, vars) => {
      await queryClient.invalidateQueries({ queryKey: modelRiskKeys.list() })
      await queryClient.invalidateQueries({
        queryKey: modelRiskKeys.detail(vars.modelName),
      })
      await queryClient.invalidateQueries({ queryKey: modelRiskKeys.all })
    },
  })
}
