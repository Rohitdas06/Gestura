/**
 * src/app/index.tsx — Home tab route.
 *
 * Re-exports LiveTranslationScreen as the default component for the
 * Expo Router "/" route. Keeping the route file thin follows the AGENTS.md
 * guideline: "Keep non-route code (components, hooks, utils) outside src/app/."
 */
export { default } from '@/screens/LiveTranslationScreen';