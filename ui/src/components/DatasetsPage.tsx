import { useState, useRef, type ChangeEvent } from "react"
import {
  Database,
  Download,
  Eye,
  Search,
  ChevronDown,
  ChevronRight,
  Rows3,
  Columns3,
  Upload,
} from "lucide-react"
import { toast } from "sonner"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { useDatasetsQuery, useUploadDatasetMutation } from "@/lib/queries"
import { DatasetPreviewDialog } from "@/components/DatasetPreviewDialog"

interface DatasetsPageProps {
  enabled: boolean
  onDatasetsChanged?: () => void
}

function fileStem(filename: string): string {
  const i = filename.lastIndexOf(".")
  return i > 0 ? filename.slice(0, i) : filename
}

function formatRowCount(n?: number): string {
  if (n == null) return "--"
  return n.toLocaleString()
}

function ColumnList({ columns }: { columns: string[] }) {
  const [open, setOpen] = useState(false)
  if (!columns.length) return <span className="text-muted-foreground">--</span>

  const preview = columns.slice(0, 3)
  const rest = columns.length - 3

  return (
    <div>
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
      >
        {open ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
        <Columns3 className="h-3 w-3" />
        <span className="font-medium">{columns.length}</span>
      </button>
      {!open && (
        <span className="ml-1.5 text-xs text-muted-foreground/70">
          {preview.join(", ")}
          {rest > 0 && ` +${rest}`}
        </span>
      )}
      {open && (
        <div className="mt-1.5 flex flex-wrap gap-1">
          {columns.map((col) => (
            <span
              key={col}
              className="rounded bg-muted px-1.5 py-0.5 text-overline font-mono text-muted-foreground"
            >
              {col}
            </span>
          ))}
        </div>
      )}
    </div>
  )
}

const ACCEPT = ".csv,.parquet"

export function DatasetsPage({ enabled, onDatasetsChanged }: DatasetsPageProps) {
  const [search, setSearch] = useState("")
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [uploadOpen, setUploadOpen] = useState(false)
  const [pendingFile, setPendingFile] = useState<File | null>(null)
  const [uploadName, setUploadName] = useState("")
  const [uploadDescription, setUploadDescription] = useState("")
  const [previewName, setPreviewName] = useState<string | null>(null)

  const datasetsQuery = useDatasetsQuery({ enabled })
  const uploadMutation = useUploadDatasetMutation()

  const datasets = datasetsQuery.data ?? []
  const listLoading = enabled && datasetsQuery.isPending && datasetsQuery.data === undefined

  const filtered = datasets.filter((d) =>
    !search || d.name.toLowerCase().includes(search.toLowerCase()) ||
    d.description?.toLowerCase().includes(search.toLowerCase())
  )

  const openFilePicker = () => fileInputRef.current?.click()

  const onFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    e.target.value = ""
    if (!file) return
    const lower = file.name.toLowerCase()
    if (!lower.endsWith(".csv") && !lower.endsWith(".parquet")) {
      toast.error("Choose a .csv or .parquet file.")
      return
    }
    setPendingFile(file)
    setUploadName(fileStem(file.name))
    setUploadDescription("")
    setUploadOpen(true)
  }

  const handleUpload = () => {
    if (!pendingFile) return
    uploadMutation.mutate(
      {
        file: pendingFile,
        name: uploadName.trim() || undefined,
        description: uploadDescription.trim() || undefined,
      },
      {
        onSuccess: () => {
          toast.success("Dataset uploaded")
          setUploadOpen(false)
          setPendingFile(null)
          onDatasetsChanged?.()
        },
        onError: (err) => {
          toast.error(err instanceof Error ? err.message : "Upload failed")
        },
      },
    )
  }

  const uploadBusy = uploadMutation.isPending

  return (
    <div className="flex-1 overflow-auto">
      <input
        ref={fileInputRef}
        type="file"
        accept={ACCEPT}
        className="hidden"
        onChange={onFileChange}
      />

      <Dialog
        open={uploadOpen}
        onOpenChange={(open) => {
          if (!open && !uploadBusy) {
            setUploadOpen(false)
            setPendingFile(null)
          }
        }}
      >
        <DialogContent
          className="sm:max-w-md"
          onPointerDownOutside={(e) => {
            if (uploadBusy) e.preventDefault()
          }}
          onEscapeKeyDown={(e) => {
            if (uploadBusy) e.preventDefault()
          }}
        >
          <DialogHeader>
            <DialogTitle className="font-headline">Upload dataset</DialogTitle>
            <DialogDescription>
              {pendingFile && (
                <span className="font-mono text-foreground">{pendingFile.name}</span>
              )}
            </DialogDescription>
          </DialogHeader>
          <div className="grid gap-3">
            <div>
              <label className="text-xs font-medium text-muted-foreground">Name</label>
              <input
                type="text"
                value={uploadName}
                onChange={(e) => setUploadName(e.target.value)}
                disabled={uploadBusy}
                className="mt-1 w-full rounded-lg border border-border bg-card px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-ring"
              />
            </div>
            <div>
              <label className="text-xs font-medium text-muted-foreground">Description (optional)</label>
              <textarea
                value={uploadDescription}
                onChange={(e) => setUploadDescription(e.target.value)}
                disabled={uploadBusy}
                rows={3}
                className="mt-1 w-full resize-none rounded-lg border border-border bg-card px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-ring"
              />
            </div>
          </div>
          <DialogFooter className="gap-2 sm:gap-0">
            <Button
              type="button"
              variant="outline"
              onClick={() => {
                if (uploadBusy) return
                setUploadOpen(false)
                setPendingFile(null)
              }}
              disabled={uploadBusy}
            >
              Cancel
            </Button>
            <Button type="button" onClick={handleUpload} disabled={uploadBusy || !pendingFile}>
              {uploadBusy ? "Uploading…" : "Upload"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <DatasetPreviewDialog
        datasetName={previewName}
        onOpenChange={(open) => {
          if (!open) setPreviewName(null)
        }}
      />

      <div className="max-w-5xl mx-auto px-8 py-10">
        <span className="text-overline font-bold text-muted-foreground tracking-widest uppercase">
          Data Management
        </span>
        <div className="flex items-end justify-between mt-1 gap-4">
          <div>
            <h1 className="font-headline text-2xl sm:text-3xl font-semibold text-foreground tracking-tight">
              Data Assets
            </h1>
            <p className="mt-2 text-muted-foreground text-base leading-relaxed max-w-lg">
              Manage and explore your datasets for training and evaluation.
            </p>
          </div>
          <div className="flex items-center gap-2 shrink-0 mb-1">
            {datasets.length > 0 && (
              <Badge variant="secondary" className="tabular-nums">
                {datasets.length} dataset{datasets.length !== 1 ? "s" : ""}
              </Badge>
            )}
            <Button
              type="button"
              size="sm"
              variant="secondary"
              className="gap-1.5"
              onClick={openFilePicker}
              disabled={!enabled || listLoading}
            >
              <Upload className="h-3.5 w-3.5" />
              Upload
            </Button>
          </div>
        </div>

        {listLoading ? (
          <div className="mt-10 rounded-xl border border-border bg-card p-10 text-center text-sm text-muted-foreground">
            Loading datasets…
          </div>
        ) : (
          <>
            {datasets.length > 0 && (
              <div className="relative mt-6">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
                <input
                  type="text"
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  placeholder="Filter datasets..."
                  className="w-full max-w-sm rounded-lg border border-border bg-card pl-9 pr-3 py-2 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring"
                />
              </div>
            )}

            {filtered.length > 0 ? (
              <div className="mt-6 space-y-2">
                {filtered.map((ds) => {
                  return (
                    <div
                      key={ds.id ?? ds.name}
                      className="rounded-xl border border-border bg-card p-4 transition-colors hover:bg-accent/30"
                    >
                      <div className="flex items-start justify-between gap-4">
                        <div className="min-w-0 flex-1">
                          <div className="flex items-center gap-2">
                            <Database className="h-4 w-4 shrink-0 text-primary" />
                            <span className="font-medium text-sm text-foreground truncate">
                              {ds.name}
                            </span>
                          </div>
                          {ds.description && (
                            <p className="mt-1 text-xs text-muted-foreground line-clamp-2 pl-6">
                              {ds.description}
                            </p>
                          )}
                        </div>

                        {ds.trainable && ds.name && (
                          <div className="shrink-0 flex items-center gap-2">
                            <Button
                              type="button"
                              variant="outline"
                              size="sm"
                              className="gap-1.5 h-8 text-xs font-medium"
                              onClick={() => setPreviewName(ds.name)}
                            >
                              <Eye className="h-3 w-3" />
                              Preview
                            </Button>
                            <a
                              href={`/api/datasets/${encodeURIComponent(ds.name)}/download`}
                              className="inline-flex items-center gap-1 rounded-md border border-border bg-background px-2.5 py-1.5 text-xs font-medium text-muted-foreground hover:text-foreground hover:bg-accent transition-colors"
                            >
                              <Download className="h-3 w-3" />
                              Download
                            </a>
                          </div>
                        )}
                      </div>

                      <div className="mt-3 pl-6 flex items-center gap-6 text-xs text-muted-foreground">
                        {ds.rows != null && (
                          <span className="flex items-center gap-1">
                            <Rows3 className="h-3 w-3" />
                            <span className="font-medium tabular-nums">{formatRowCount(ds.rows)}</span> rows
                          </span>
                        )}
                        <ColumnList columns={ds.columns ?? []} />
                      </div>
                    </div>
                  )
                })}
              </div>
            ) : datasets.length > 0 ? (
              <div className="mt-10 rounded-xl bg-card p-8 text-center text-muted-foreground text-sm">
                No datasets match "{search}".
              </div>
            ) : (
              <div className="mt-10 rounded-xl bg-card border border-border p-10 text-center">
                <Database className="mx-auto h-8 w-8 text-muted-foreground/50" />
                <p className="mt-3 text-sm text-muted-foreground">
                  No datasets yet. Upload a CSV or Parquet file, or train a model in the Experiment Lab.
                </p>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}
