import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { useDatasetPreviewQuery } from "@/lib/queries"

export const DATASET_PREVIEW_ROW_LIMIT = 25

export function formatPreviewCell(value: unknown): string {
  if (value === null || value === undefined) return ""
  if (typeof value === "object") return JSON.stringify(value)
  return String(value)
}

interface DatasetPreviewDialogProps {
  datasetName: string | null
  onOpenChange: (open: boolean) => void
  /** Defaults to {@link DATASET_PREVIEW_ROW_LIMIT}. */
  limit?: number
}

/**
 * Same dataset table preview as Data Assets — uses `/api/datasets/.../preview`.
 */
export function DatasetPreviewDialog({
  datasetName,
  onOpenChange,
  limit = DATASET_PREVIEW_ROW_LIMIT,
}: DatasetPreviewDialogProps) {
  const previewQuery = useDatasetPreviewQuery(datasetName, {
    enabled: datasetName !== null,
    limit,
  })

  return (
    <Dialog open={datasetName !== null} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-6xl w-full max-h-[85vh] flex flex-col gap-0 p-0 overflow-hidden">
        <DialogHeader className="px-6 pt-6 pb-3 shrink-0 border-b border-border">
          <DialogTitle className="font-headline pr-8">
            {datasetName ? (
              <span className="truncate block" title={datasetName}>
                Preview: {datasetName}
              </span>
            ) : (
              "Dataset preview"
            )}
          </DialogTitle>
          <DialogDescription>First {limit} rows</DialogDescription>
        </DialogHeader>
        <div className="min-h-0 flex-1 overflow-auto px-6 pb-6">
          {previewQuery.isPending && (
            <div className="py-12 text-center text-sm text-muted-foreground">Loading preview…</div>
          )}
          {previewQuery.isError && (
            <div className="py-12 text-center text-sm text-destructive">
              {previewQuery.error instanceof Error ? previewQuery.error.message : "Preview failed"}
            </div>
          )}
          {previewQuery.data && !previewQuery.isPending && (
            <div className="rounded-lg border border-border overflow-x-auto">
              <table className="w-full text-xs border-collapse">
                <thead>
                  <tr className="bg-muted/50 border-b border-border">
                    {previewQuery.data.columns.map((col) => (
                      <th
                        key={col}
                        className="text-left font-medium text-muted-foreground px-3 py-2 whitespace-nowrap max-w-[14rem]"
                      >
                        <span className="font-mono">{col}</span>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {previewQuery.data.rows.map((row, ri) => (
                    <tr
                      key={ri}
                      className="border-b border-border/60 last:border-0 hover:bg-muted/20"
                    >
                      {previewQuery.data!.columns.map((col) => {
                        const raw = row[col]
                        const text = formatPreviewCell(raw)
                        return (
                          <td
                            key={col}
                            className="px-3 py-1.5 max-w-[14rem] font-mono text-foreground/90 align-top"
                            title={text.length > 80 ? text : undefined}
                          >
                            <span className="line-clamp-3 break-all">{text || "—"}</span>
                          </td>
                        )
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  )
}
