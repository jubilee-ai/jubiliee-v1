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
  type ExperimentSummary,
  type ExperimentDetail,
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
