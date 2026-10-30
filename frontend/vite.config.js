import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

const BACKEND_PORT = process.env.VITE_BACKEND_PORT || 8000
const BACKEND_HTTP = `http://127.0.0.1:${BACKEND_PORT}`
const BACKEND_WS = `ws://127.0.0.1:${BACKEND_PORT}`

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    host: true,
    port: 5173,
    proxy: {
      '/api': {
        target: BACKEND_HTTP,
        changeOrigin: true,
      },
      '/tts': {
        target: BACKEND_HTTP,
        changeOrigin: true,
      },
      '/ws': {
        target: BACKEND_WS,
        ws: true,
        changeOrigin: true,
      },
    },
  },
})
