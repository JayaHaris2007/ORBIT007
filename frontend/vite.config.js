import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { VitePWA } from 'vite-plugin-pwa'
import { spawnSync } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const repoRoot = path.resolve(__dirname, '..')
const bridgeScript = path.join(repoRoot, 'spidey_api_bridge.py')
const isCloudContainer = fs.existsSync('/app/control-plane-api')

function spideyBackendBridge() {
  return {
    name: 'spidey-backend-bridge',
    configureServer(server) {
      server.middlewares.use(async (req, res, next) => {
        const url = req.url || ''
        if (
          !url.startsWith('/api/') &&
          url !== '/api' &&
          !url.startsWith('/notifications/') &&
          url !== '/chat'
        ) {
          return next()
        }

        const chunks = []
        for await (const chunk of req) {
          chunks.push(chunk)
        }
        const rawBody = Buffer.concat(chunks).toString('utf-8')
        let parsedBody = null
        if (rawBody) {
          try {
            parsedBody = JSON.parse(rawBody)
          } catch {
            parsedBody = null
          }
        }

        // On local desktop (non-cloud container), try forwarding to local FastAPI on 127.0.0.1:8000 first
        if (!isCloudContainer) {
          try {
            const target = (process.env.VITE_SPIDEY_BACKEND_URL || 'http://127.0.0.1:8000') + url
            const upstream = await fetch(target, {
              method: req.method,
              headers: { 'Content-Type': 'application/json' },
              body: ['GET', 'HEAD'].includes(req.method) ? undefined : rawBody,
            })
            const text = await upstream.text()
            res.statusCode = upstream.status
            res.setHeader('Content-Type', upstream.headers.get('content-type') || 'application/json')
            res.end(text)
            return
          } catch {
            // Fall through to local Python bridge if FastAPI isn't running yet
          }
        }

        const proc = spawnSync(
          'python3',
          [bridgeScript],
          {
            cwd: repoRoot,
            input: JSON.stringify({
              method: req.method || 'GET',
              url,
              body: parsedBody,
            }),
            encoding: 'utf-8',
            timeout: 15000,
          }
        )

        res.setHeader('Content-Type', 'application/json')
        if (proc.status === 0 && proc.stdout) {
          try {
            // Find last JSON line in stdout in case of startup print statements
            const lines = proc.stdout.trim().split('\n')
            const result = JSON.parse(lines[lines.length - 1])
            res.statusCode = result.status || 200
            res.end(JSON.stringify(result.body))
            return
          } catch (err) {
            res.statusCode = 500
            res.end(JSON.stringify({ error: 'Bridge JSON parse error', details: String(err) }))
            return
          }
        }

        res.statusCode = 500
        res.end(
          JSON.stringify({
            error: (proc.stderr || 'Python bridge execution failed').trim(),
          })
        )
      })
    },
  }
}

// https://vite.dev/config/
export default defineConfig({
  server: {
    port: 3000,
    host: '0.0.0.0',
    allowedHosts: true,
    hmr: process.env.DISABLE_HMR !== 'true',
    watch: process.env.DISABLE_HMR === 'true' ? null : {},
  },
  preview: {
    allowedHosts: true,
  },
  plugins: [
    spideyBackendBridge(),
    react(),
    tailwindcss(),
    VitePWA({
      registerType: 'autoUpdate',
      includeAssets: ['logo.png', 'vite.svg'],
      manifest: {
        name: 'S.P.I.D.E.Y - Personal Assistant',
        short_name: 'S.P.I.D.E.Y',
        description: 'Local personal assistant with habits, tasks, reminders, and memory.',
        theme_color: '#0a0a0a',
        background_color: '#0a0a0a',
        display: 'standalone',
        scope: '/',
        start_url: '/',
        orientation: 'portrait',
        icons: [
          {
            src: 'logo.png',
            sizes: '192x192',
            type: 'image/png',
          },
          {
            src: 'logo.png',
            sizes: '512x512',
            type: 'image/png',
          },
        ],
      },
    }),
  ],
})
