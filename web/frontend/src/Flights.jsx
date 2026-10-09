import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from './api.js'
import { go } from './App.jsx'
import { fmtDate, fmtDuration } from './format.js'

const STATUS = {
  processed: ['ok', 'processed'], processing: ['warn', 'processing'], queued: ['warn', 'queued'],
  failed: ['bad', 'failed'],
}

export default function Flights() {
  const [flights, setFlights] = useState(null)
  const [error, setError] = useState('')
  const refresh = useCallback(() => api.flights().then(setFlights).catch((e) => setError(e.message)), [])
  useEffect(() => { refresh() }, [refresh])

  // While any flight is being processed, refresh the list every few seconds.
  const busy = (flights || []).some((f) => f.status === 'queued' || f.status === 'processing')
  useEffect(() => {
    if (!busy) return
    const t = setInterval(refresh, 4000)
    return () => clearInterval(t)
  }, [busy, refresh])

  return (
    <div className="flights-page">
      <Upload onChange={refresh} />
      <div className="card">
        <div className="row spread" style={{ marginBottom: 8 }}>
          <h2>Flights</h2>
          <button className="btn small" onClick={refresh}>Refresh</button>
        </div>
        {error && <div className="error">{error}</div>}
        {!flights && !error && <p className="muted">Loading…</p>}
        {flights && !flights.length && <p className="muted">No flights yet — upload one above.</p>}
        {flights && flights.length > 0 && (
          <div className="table-wrap">
            <table>
              <thead>
                <tr><th>Flight</th><th>Date</th><th>Duration</th><th>Distance</th><th>Photos</th><th>Potholes</th><th>Status</th><th /></tr>
              </thead>
              <tbody>
                {flights.map((f) => {
                  const s = f.summary
                  const [cls, label] = STATUS[f.status] || ['', f.status]
                  return (
                    <tr key={f.id}>
                      <td><b>{f.name}</b><div className="muted small">{f.id}</div></td>
                      <td>{fmtDate(s?.start || f.uploaded_at)}</td>
                      <td className="num">{s ? fmtDuration(s.duration_s) : '–'}</td>
                      <td className="num">{s ? `${Math.round(s.distance_m)} m` : '–'}</td>
                      <td className="num">{s ? s.frames : '–'}</td>
                      <td className="num">
                        {s ? (
                          <span className="row" style={{ gap: 6 }}>
                            <b>{s.potholes}</b>
                            <span className="muted small">
                              <i className="dot high" /> {s.by_severity.high} <i className="dot medium" /> {s.by_severity.medium} <i className="dot low" /> {s.by_severity.low}
                            </span>
                          </span>
                        ) : '–'}
                      </td>
                      <td>
                        <span className={`badge ${cls}`} title={f.error || ''}>{label}</span>
                        {f.status === 'failed' && <div className="small" style={{ color: '#8a1c12' }}>{f.error}</div>}
                        {(f.status === 'processing' || f.status === 'queued') && f.job_id && <JobBar jobId={f.job_id} />}
                      </td>
                      <td>
                        <button className="btn small primary" disabled={f.status !== 'processed'}
                          onClick={() => go('map', { flight: f.id })}>Open map</button>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}

function JobBar({ jobId }) {
  const [job, setJob] = useState(null)
  useEffect(() => {
    let alive = true
    const poll = () => api.job(jobId).then((j) => alive && setJob(j)).catch(() => {})
    poll()
    const t = setInterval(poll, 1500)
    return () => { alive = false; clearInterval(t) }
  }, [jobId])
  if (!job) return null
  return (
    <div style={{ marginTop: 4, minWidth: 140 }}>
      <div className="bar"><div style={{ width: `${Math.round(job.progress * 100)}%` }} /></div>
      <div className="muted small">{Math.round(job.progress * 100)} % · {job.message}</div>
    </div>
  )
}

function Upload({ onChange }) {
  const [file, setFile] = useState(null)
  const [name, setName] = useState('')
  const [over, setOver] = useState(false)
  const [sent, setSent] = useState(null)       // upload progress 0..1 while sending
  const [error, setError] = useState('')
  const [done, setDone] = useState('')
  const input = useRef()

  const pick = (f) => {
    setError(''); setDone('')
    if (f && !f.name.toLowerCase().endsWith('.zip')) { setError('Please choose a .zip file.'); return }
    setFile(f || null)
  }

  async function submit() {
    setError(''); setDone(''); setSent(0)
    try {
      const r = await api.upload(file, name, setSent)
      setDone(`Uploaded ${r.frames} photos as ${r.flight_id} — processing in the background.`)
      setFile(null); setName('')
      if (input.current) input.current.value = ''
      onChange()
    } catch (e) {
      setError(e.message)
    } finally {
      setSent(null)
    }
  }

  return (
    <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
      <h2>Upload a flight</h2>
      <p className="muted">
        After landing, copy the flight folder from the Pi and zip it. The zip needs <code>images/</code>,{' '}
        <code>frames.csv</code> (frame, image, timestamp) and <code>telemetry.csv</code> (timestamp, lat, lon,
        alt_m, heading_deg); <code>flight.json</code> is optional. The server runs the detector on every photo,
        places each pothole on the map and merges repeats (about 0.5 s per 640 px photo, 20 s per 4K photo on this PC).
      </p>
      <div className={over ? 'drop over' : 'drop'}
        onDragOver={(e) => { e.preventDefault(); setOver(true) }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => { e.preventDefault(); setOver(false); pick(e.dataTransfer.files[0]) }}>
        <div className="row">
          <input ref={input} type="file" accept=".zip" onChange={(e) => pick(e.target.files[0])} aria-label="Flight zip" />
        </div>
        <div className="muted small">…or drop the zip here. Demo file: <code>dist/synthetic_flight_001.zip</code></div>
        <div className="row">
          <input type="text" placeholder="Flight name (optional)" value={name} onChange={(e) => setName(e.target.value)}
            style={{ flex: 1, minWidth: 180 }} aria-label="Flight name" />
          <button className="btn primary" disabled={!file || sent !== null} onClick={submit}>
            {sent !== null ? 'Uploading…' : 'Upload & process'}
          </button>
        </div>
        {sent !== null && (
          <div>
            <div className="bar"><div style={{ width: `${Math.round(sent * 100)}%` }} /></div>
            <div className="muted small">Uploading {file?.name}: {Math.round(sent * 100)} %</div>
          </div>
        )}
      </div>
      {error && <div className="error">{error}</div>}
      {done && <div className="note">{done}</div>}
    </div>
  )
}
