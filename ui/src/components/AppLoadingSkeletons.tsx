import { Skeleton } from "@/components/ui/skeleton"

const NAV_CLASS =
  "fixed top-0 w-full z-50 border-b border-border/60 bg-background/90 backdrop-blur-xl shadow-[0_1px_0_hsl(var(--border)/0.35),0_12px_40px_-12px_rgba(18,86,210,0.07)] flex items-center justify-between px-6 h-14"

/** Full shell while Clerk auth is resolving — matches post-auth layout. */
export function AuthLoadingShell() {
  return (
    <div className="h-screen bg-background flex flex-col overflow-hidden">
      <nav className={NAV_CLASS}>
        <Skeleton className="h-7 w-28 rounded-md" />
        <div className="flex items-center gap-1.5">
          <Skeleton className="h-8 w-8 rounded-md" />
          <Skeleton className="h-8 w-8 rounded-md" />
          <Skeleton className="h-8 w-8 rounded-md" />
          <Skeleton className="h-8 w-8 rounded-full ml-1" />
        </div>
      </nav>
      <div className="flex flex-1 min-h-0 overflow-hidden pt-14">
        <aside className="hidden md:flex w-[240px] shrink-0 fixed left-0 top-14 bottom-0 flex-col border-r border-border/50 bg-background px-3 pt-3">
          {Array.from({ length: 5 }).map((_, i) => (
            <Skeleton key={i} className="h-9 w-full rounded-lg mb-0.5" />
          ))}
          <div className="mt-5 px-2">
            <Skeleton className="h-3 w-14 mb-3" />
            <SidebarExperimentsSkeleton />
          </div>
        </aside>
        <main className="flex-1 min-h-0 flex flex-col overflow-hidden md:ml-[240px] relative">
          <div className="jubilee-dot-grid absolute inset-0 opacity-[0.72]" />
          <div className="relative flex flex-1 min-h-0 flex-col p-6 pt-10">
            <Skeleton className="h-4 w-40 mx-auto mb-8 rounded-full" />
            <div className="max-w-xl mx-auto w-full space-y-4 flex-1">
              <Skeleton className="h-24 w-full rounded-2xl" />
              <Skeleton className="h-16 w-3/4 rounded-2xl" />
            </div>
            <Skeleton className="h-14 w-full max-w-3xl mx-auto rounded-xl mt-auto" />
          </div>
        </main>
      </div>
    </div>
  )
}

export function SidebarExperimentsSkeleton() {
  return (
    <div className="space-y-2 pr-1">
      {Array.from({ length: 5 }).map((_, i) => (
        <div key={i} className="rounded-lg border border-transparent p-2.5 flex gap-2">
          <Skeleton className="h-3.5 w-3.5 shrink-0 rounded-full mt-0.5" />
          <div className="flex-1 min-w-0 space-y-2">
            <div className="flex justify-between gap-2">
              <Skeleton className="h-4 flex-1 rounded-md" />
              <Skeleton className="h-3 w-10 shrink-0 rounded-md" />
            </div>
            <Skeleton className="h-3 w-full rounded-md" />
          </div>
        </div>
      ))}
    </div>
  )
}

/** Lab main area while the first health check has not finished. */
export function LabConnectionSkeleton() {
  return (
    <div className="relative flex flex-1 min-h-0 flex-col overflow-hidden">
      <div className="jubilee-dot-grid absolute inset-0 opacity-[0.72]" />
      <div className="relative flex flex-1 min-h-0 flex-col p-6 pt-12">
        <Skeleton className="h-3 w-36 mx-auto mb-10 rounded-full opacity-60" />
        <div className="max-w-xl mx-auto w-full space-y-4 flex-1">
          <Skeleton className="h-28 w-full rounded-2xl" />
          <Skeleton className="h-20 w-[88%] rounded-2xl" />
        </div>
        <Skeleton className="h-[52px] w-full max-w-3xl mx-auto rounded-xl mt-auto opacity-90" />
      </div>
    </div>
  )
}

/** Datasets / Models list placeholder while API reachability or catalog is loading. */
export function CatalogPageSkeleton({ titleWidth = "w-48" }: { titleWidth?: string }) {
  return (
    <div className="flex-1 overflow-auto relative">
      <div className="max-w-5xl mx-auto px-8 py-10 space-y-6">
        <Skeleton className="h-3 w-28 rounded-full" />
        <Skeleton className={`h-9 ${titleWidth} rounded-lg`} />
        <Skeleton className="h-4 w-full max-w-lg rounded-md" />
        <Skeleton className="h-10 w-full max-w-md rounded-lg mt-4" />
        <div className="mt-8 space-y-3">
          {Array.from({ length: 6 }).map((_, i) => (
            <Skeleton key={i} className="h-[72px] w-full rounded-xl" />
          ))}
        </div>
      </div>
    </div>
  )
}
