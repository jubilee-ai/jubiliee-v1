import { OrganizationProfile, UserProfile, useAuth } from "@clerk/react"
import { cn } from "@/lib/utils"

const clerkPanelAppearance = {
  elements: {
    rootBox: "w-full",
    card: "shadow-none border border-border rounded-xl bg-card",
    navbar: "border-b border-border",
    navbarButton: "text-foreground",
  },
}

export function SettingsPage() {
  const { isLoaded, isSignedIn, has } = useAuth()
  const canManageOrg = isLoaded && isSignedIn && has({ role: "org:admin" })

  return (
    <div className="max-w-6xl mx-auto px-6 py-10 pb-16">
      <span className="text-overline font-bold text-muted-foreground tracking-widest uppercase">
        Configuration
      </span>
      <h1 className="font-headline text-3xl font-semibold text-foreground tracking-tight mt-1">
        Settings
      </h1>
      <p className="mt-3 text-muted-foreground text-sm leading-relaxed max-w-lg">
        {canManageOrg
          ? "Manage your account and organization: members, roles, and invitations."
          : "Manage your account, sign-in methods, and profile."}
      </p>
      {isLoaded && isSignedIn && !canManageOrg ? (
        <p className="mt-2 text-muted-foreground text-sm leading-relaxed max-w-lg">
          Organization administration is limited to organization admins (Clerk role{" "}
          <span className="font-mono text-xs">org:admin</span>).
        </p>
      ) : null}

      <div
        className={cn(
          "mt-10 grid gap-12",
          canManageOrg ? "lg:grid-cols-2" : "max-w-xl",
        )}
      >
        {canManageOrg ? (
          <section className="min-w-0">
            <h2 className="text-sm font-semibold text-foreground mb-4">Organization</h2>
            <div className="rounded-xl border border-border bg-card/50 overflow-hidden min-h-[480px]">
              <OrganizationProfile routing="hash" appearance={clerkPanelAppearance} />
            </div>
          </section>
        ) : null}
        <section className={cn("min-w-0", !canManageOrg && "max-w-xl")}>
          <h2 className="text-sm font-semibold text-foreground mb-4">Account</h2>
          <div className="rounded-xl border border-border bg-card/50 overflow-hidden min-h-[480px]">
            <UserProfile routing="hash" appearance={clerkPanelAppearance} />
          </div>
        </section>
      </div>
    </div>
  )
}
