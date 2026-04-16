import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import './index.css'
import App from './App.tsx'
import { isPublishableKey } from '@clerk/shared/keys'
import { ClerkProvider } from '@clerk/react'
import { Toaster } from 'sonner'

const publishableKey = import.meta.env.VITE_CLERK_PUBLISHABLE_KEY
if (!publishableKey) {
  throw new Error('Missing VITE_CLERK_PUBLISHABLE_KEY')
}
if (!isPublishableKey(publishableKey)) {
  throw new Error(
    'Invalid VITE_CLERK_PUBLISHABLE_KEY. Paste the full publishable key from Clerk Dashboard → API Keys. ' +
      'A truncated or edited value cannot be decoded, so Clerk builds script URLs like https://npm/... and the browser fails DNS.',
  )
}

const browserRedirectUrl =
  typeof window === 'undefined' ? '/' : `${window.location.origin}${window.location.pathname}`

const signInUrl = import.meta.env.VITE_CLERK_SIGN_IN_URL ?? browserRedirectUrl
const signUpUrl = import.meta.env.VITE_CLERK_SIGN_UP_URL ?? browserRedirectUrl

const signInForceRedirectUrl =
  import.meta.env.VITE_CLERK_SIGN_IN_FORCE_REDIRECT_URL ?? browserRedirectUrl
const signInFallbackRedirectUrl =
  import.meta.env.VITE_CLERK_SIGN_IN_FALLBACK_REDIRECT_URL ?? browserRedirectUrl

const signUpForceRedirectUrl =
  import.meta.env.VITE_CLERK_SIGN_UP_FORCE_REDIRECT_URL ?? browserRedirectUrl
const signUpFallbackRedirectUrl =
  import.meta.env.VITE_CLERK_SIGN_UP_FALLBACK_REDIRECT_URL ?? browserRedirectUrl

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
    },
  },
})

/** Matches Jubilee product palette (light shell + clinical blues). */
const clerkAppearance = {
  theme: 'simple' as const,
  variables: {
    colorPrimary: 'hsl(222 36% 81%)',
    colorPrimaryForeground: 'hsl(222 58% 24%)',
    colorForeground: 'hsl(222 69% 13%)',
    colorMutedForeground: 'hsl(222 32% 47%)',
    colorBackground: 'hsl(0 0% 100%)',
    colorInput: 'hsl(223 100% 96%)',
    colorInputForeground: 'hsl(222 69% 13%)',
    colorNeutral: 'hsl(221 100% 92%)',
    colorBorder: 'hsl(221 100% 92%)',
    colorRing: 'hsl(218 62% 53%)',
    colorDanger: 'hsl(349 52% 44%)',
    colorSuccess: 'hsl(217 88% 57%)',
    colorWarning: 'hsl(42 96% 58%)',
    colorShadow: 'rgba(18, 86, 210, 0.12)',
    colorModalBackdrop: 'rgba(10, 26, 54, 0.35)',
    fontFamily: 'Inter, system-ui, -apple-system, sans-serif',
    fontFamilyButtons: 'Inter, system-ui, -apple-system, sans-serif',
    borderRadius: '0.75rem',
  },
  elements: {
    userButtonPopoverCard:
      'rounded-xl border border-border bg-card text-card-foreground shadow-2xl',
    userButtonPopoverMain: 'bg-transparent',
    userButtonPopoverActionButton:
      'rounded-lg text-foreground hover:bg-accent hover:text-accent-foreground',
    userButtonPopoverActionButtonText: 'text-foreground',
    userButtonPopoverFooter:
      'border-t border-border bg-muted/50 text-muted-foreground',
  },
}

const clerkLocalization = {
  socialButtonsBlockButton: 'Sign in with {{provider|titleize}}',
} as const

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <ClerkProvider
        publishableKey={publishableKey}
        appearance={clerkAppearance}
        localization={clerkLocalization}
        signInUrl={signInUrl}
        signUpUrl={signUpUrl}
        signInForceRedirectUrl={signInForceRedirectUrl}
        signInFallbackRedirectUrl={signInFallbackRedirectUrl}
        signUpForceRedirectUrl={signUpForceRedirectUrl}
        signUpFallbackRedirectUrl={signUpFallbackRedirectUrl}
      >
        <App />
        <Toaster position="bottom-right" closeButton richColors offset={20} />
      </ClerkProvider>
    </QueryClientProvider>
  </StrictMode>,
)
