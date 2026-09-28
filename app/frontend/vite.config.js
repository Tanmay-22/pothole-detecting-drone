import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// Dev: `npm run dev` on :5173 forwards API and sample requests to the FastAPI server on :8000.
// Build: `npm run build` → dist/, which FastAPI serves at / (one command to run the demo).
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': 'http://127.0.0.1:8000',
      '/samples': 'http://127.0.0.1:8000',
    },
  },
})
