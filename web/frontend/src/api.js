// Thin wrappers over the FastAPI backend (web/backend/main.py).
async function get(path) {
  const r = await fetch(path)
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || `${r.status} ${r.statusText}`)
  return r.json()
}

export const api = {
  config: () => get('/api/config'),
  flights: () => get('/api/flights'),
  flight: (id) => get(`/api/flights/${id}`),
  job: (id) => get(`/api/jobs/${id}`),
  cropUrl: (id, crop) => `/api/flights/${id}/${crop}`,
  imageUrl: (id, name) => `/api/flights/${id}/images/${name}`,
  downloadUrl: (id, kind) => `/api/flights/${id}/download/${kind}`,
  // XMLHttpRequest instead of fetch: it reports upload progress
  upload(file, name, onProgress) {
    return new Promise((resolve, reject) => {
      const x = new XMLHttpRequest()
      const form = new FormData()
      form.append('file', file)
      form.append('name', name)
      x.open('POST', '/api/flights')
      x.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total)
      x.onload = () => {
        let body = {}
        try { body = JSON.parse(x.responseText) } catch { /* not JSON */ }
        if (x.status >= 200 && x.status < 300) resolve(body)
        else reject(new Error(body.detail || `upload failed (${x.status})`))
      }
      x.onerror = () => reject(new Error('network error during upload'))
      x.send(form)
    })
  },
}
