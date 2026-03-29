import { useState } from "react"
import { Database, Download, Search, ChevronDown, ChevronRight, Rows3, Columns3, FileText, Share2, Users } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { shareDataset, unshareDataset, type Dataset } from "@/lib/api"

interface DatasetsPageProps {
  datasets: Dataset[]
}

const sourceTypeBadge: Record<string, { label: string; className: string }> = {
  catalog: { label: "Catalog", className: "bg-blue-500/15 text-blue-400 border-transparent" },
  derived: { label: "Derived", className: "bg-purple-500/15 text-purple-400 border-transparent" },
  uploaded: { label: "Uploaded", className: "bg-emerald-500/15 text-emerald-400 border-transparent" },
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
              className="rounded bg-muted px-1.5 py-0.5 text-[10px] font-mono text-muted-foreground"
            >
              {col}
            </span>
          ))}
        </div>
      )}
    </div>
  )
}

export function DatasetsPage({ datasets }: DatasetsPageProps) {
  const [search, setSearch] = useState("")
  const [shareState, setShareState] = useState<Record<string, boolean>>({})

  const filtered = datasets.filter((d) =>
    !search || d.name.toLowerCase().includes(search.toLowerCase()) ||
    d.description?.toLowerCase().includes(search.toLowerCase())
  )

  const handleShare = async (ds: Dataset) => {
    const id = ds.id
    if (!id) return
    const currentlyShared = shareState[id] ?? ds.shared_with_org ?? false
    const next = !currentlyShared
    setShareState((prev) => ({ ...prev, [id]: next }))
    try {
      if (next) await shareDataset(id)
      else await unshareDataset(id)
    } catch {
      setShareState((prev) => ({ ...prev, [id]: !next }))
    }
  }

  return (
    <div className="flex-1 overflow-auto">
      <div className="max-w-5xl mx-auto px-8 py-10">
        <span className="text-[10px] font-bold text-muted-foreground tracking-widest uppercase">
          Data Management
        </span>
        <div className="flex items-end justify-between mt-1">
          <div>
            <h1 className="font-headline text-3xl font-semibold text-foreground tracking-tight">
              Data Assets
            </h1>
            <p className="mt-2 text-muted-foreground text-sm leading-relaxed max-w-lg">
              Manage and explore your datasets for training and evaluation.
            </p>
          </div>
          {datasets.length > 0 && (
            <Badge variant="secondary" className="mb-1 tabular-nums">
              {datasets.length} dataset{datasets.length !== 1 ? "s" : ""}
            </Badge>
          )}
        </div>

        {/* Search */}
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

        {/* Table */}
        {filtered.length > 0 ? (
          <div className="mt-6 space-y-2">
            {filtered.map((ds) => {
              const badge = sourceTypeBadge[ds.source_type ?? ""] ?? {
                label: ds.source_type ?? "unknown",
                className: "bg-muted text-muted-foreground border-transparent",
              }
              const isOwner = ds.is_owner !== false
              const isShared = shareState[ds.id ?? ""] ?? ds.shared_with_org ?? false
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
                        <Badge className={`text-[10px] px-1.5 py-0 ${badge.className}`}>
                          {badge.label}
                        </Badge>
                        {ds.format && (
                          <span className="flex items-center gap-0.5 text-[10px] text-muted-foreground font-mono uppercase">
                            <FileText className="h-3 w-3" />
                            {ds.format}
                          </span>
                        )}
                        {!isOwner && (
                          <span className="inline-flex items-center gap-0.5 rounded-full bg-blue-500/10 px-1.5 py-0.5 text-[10px] font-medium text-blue-400">
                            <Users className="h-3 w-3" />
                            Shared
                          </span>
                        )}
                      </div>
                      {ds.description && (
                        <p className="mt-1 text-xs text-muted-foreground line-clamp-2 pl-6">
                          {ds.description}
                        </p>
                      )}
                    </div>

                    <div className="flex items-center gap-1.5 shrink-0">
                      {isOwner && ds.id && (
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => handleShare(ds)}
                          className={
                            isShared
                              ? "h-8 px-2 text-emerald-400 hover:text-emerald-500 hover:bg-emerald-500/10"
                              : "h-8 px-2 text-muted-foreground hover:text-foreground"
                          }
                          title={isShared ? "Shared — click to unshare" : "Share with organization"}
                        >
                          <Share2 className="h-3.5 w-3.5 mr-1" />
                          <span className="text-xs">{isShared ? "Shared" : "Share"}</span>
                        </Button>
                      )}
                      {ds.trainable && ds.name && (
                        <a
                          href={`/api/datasets/${encodeURIComponent(ds.name)}/download`}
                          className="inline-flex items-center gap-1 rounded-md border border-border px-2.5 py-1.5 text-xs font-medium text-muted-foreground hover:text-foreground hover:bg-accent transition-colors"
                        >
                          <Download className="h-3 w-3" />
                          Download
                        </a>
                      )}
                    </div>
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
              No datasets yet. Train a model in the Experiment Lab to generate datasets.
            </p>
          </div>
        )}
      </div>
    </div>
  )
}
