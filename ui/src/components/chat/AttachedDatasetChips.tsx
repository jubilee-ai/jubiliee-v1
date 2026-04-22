import { useState } from "react"
import { Database, ChevronDown, ChevronUp } from "lucide-react"
import { cn } from "@/lib/utils"
import type { AttachedDatasetSnapshot } from "@/types/agent"

export function AttachedDatasetChips({ datasets }: { datasets: AttachedDatasetSnapshot[] }) {
  const [expanded, setExpanded] = useState<string | null>(null)
  if (!datasets.length) return null

  return (
    <div className="flex flex-col gap-2 px-1 pb-3">
      <div className="flex flex-wrap gap-2">
        {datasets.map((ds) => (
          <button
            key={ds.ref}
            type="button"
            onClick={() => setExpanded(expanded === ds.ref ? null : ds.ref)}
            className={cn(
              "inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs font-medium transition-colors",
              "border-primary/40 bg-primary/10 text-primary hover:bg-primary/15",
            )}
          >
            <Database className="h-3.5 w-3.5 shrink-0" />
            <span className="truncate max-w-[220px]">{ds.ref}</span>
            {expanded === ds.ref ? (
              <ChevronUp className="h-3.5 w-3.5 shrink-0 opacity-70" />
            ) : (
              <ChevronDown className="h-3.5 w-3.5 shrink-0 opacity-70" />
            )}
          </button>
        ))}
      </div>
      {expanded &&
        datasets.map((ds) =>
          ds.ref === expanded ? (
            <div
              key={`detail-${ds.ref}`}
              className="rounded-lg border border-border/60 bg-muted/30 p-3 text-xs space-y-2 animate-in fade-in duration-150"
            >
              <div className="flex flex-wrap gap-3 text-muted-foreground">
                {ds.rows != null && <span>{ds.rows.toLocaleString()} rows</span>}
                {ds.n_columns != null && <span>{ds.n_columns} columns</span>}
              </div>
              {ds.columns && ds.columns.length > 0 && (
                <div className="overflow-x-auto">
                  <table className="w-full text-[11px] border-collapse">
                    <thead>
                      <tr className="border-b border-border/60">
                        <th className="text-left py-1 pr-2 font-semibold">Column</th>
                        <th className="text-left py-1 font-semibold">Type</th>
                      </tr>
                    </thead>
                    <tbody>
                      {ds.columns.slice(0, 40).map((c) => (
                        <tr key={c.name} className="border-b border-border/30">
                          <td className="py-0.5 pr-2 font-mono">{c.name}</td>
                          <td className="py-0.5 text-muted-foreground">{c.dtype}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  {ds.columns.length > 40 && (
                    <p className="text-[10px] text-muted-foreground mt-1">Showing 40 of {ds.columns.length} columns</p>
                  )}
                </div>
              )}
              {ds.sample && ds.sample.length > 0 && (
                <pre className="text-[10px] leading-snug overflow-x-auto max-h-40 p-2 rounded-md bg-background/80 border border-border/40">
                  {JSON.stringify(ds.sample.slice(0, 5), null, 2)}
                </pre>
              )}
            </div>
          ) : null,
        )}
    </div>
  )
}
