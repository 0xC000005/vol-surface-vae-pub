import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'path'

// https://vite.dev/config/
// Dev: proxy the two Gradio backends under distinct prefixes so the browser stays
// same-origin (avoids CORS; the narrative→scenario backend is frozen, can't add CORS).
// /g1 -> :7860 (Narrative→Scenario), /g2 -> :7861 (Scenario→Narrative). Path is rewritten
// to strip the prefix so each Gradio app sees its own root. In prod, co-serve same-origin.
// No dev proxy needed: the Gradio backends send CORS headers (reflecting the origin),
// so the browser connects to them directly (VITE_D1_URL / VITE_D2_URL in src/lib/backend.ts).
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: { '@': path.resolve(__dirname, './src') },
  },
})
