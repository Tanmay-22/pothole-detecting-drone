import L from 'leaflet'
import { useEffect, useMemo, useState } from 'react'
import { Marker, Polyline, Tooltip, useMap } from 'react-leaflet'
import { api } from './api.js'
import { go } from './App.jsx'
import BaseMap from './BaseMap.jsx'
import { fmtDate, fmtDuration, fmtTime } from './format.js'

const SEVERITIES = ['high', 'medium', 'low']
const SEV_RANK = { high: 0, medium: 1, low: 2 }

function pinIcon(p, selected) {
  const cls = ['pin', p.severity, p.unverified ? 'unverified' : '', selected ? 'sel' : ''].join(' ')
  return L.divIcon({ className: '', html: `<div class="${cls}"></div>`, iconSize: [18, 18], iconAnchor: [9, 9] })
}

// Fit the map to the flight once per flight, and pan to the selected pothole.
function MapFocus({ flight, selected }) {
  const map = useMap()
  useEffect(() => {
    const pts = [...flight.track, ...flight.potholes.map((p) => [p.lat, p.lon])]
    if (pts.length) map.fitBounds(pts, { padding: [30, 30], maxZoom: 19 })
  }, [flight.id]) // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (selected) map.setView([selected.lat, selected.lon], Math.max(map.getZoom(), 19), { animate: true })
  }, [selected?.id]) // eslint-disable-line react-hooks/exhaustive-deps
  return null
}

export default function ResultsMap({ flightId, potholeId }) {
  const [flights, setFlights] = useState(null)
  const [flight, setFlight] = useState(null)
  const [error, setError] = useState('')
  const [show, setShow] = useState({ high: true, medium: true, low: true })
  const [hideUnverified, setHideUnverified] = useState(false)

  useEffect(() => {
    api.flights().then(setFlights).catch((e) => setError(e.message))
  }, [])

  const processed = (flights || []).filter((f) => f.status === 'processed')
  const activeId = flightId || processed[0]?.id

  useEffect(() => {
    if (!activeId) return
    setFlight(null)
    api.flight(activeId).then(setFlight).catch((e) => setError(e.message))
  }, [activeId])

  const visible = useMemo(
    () => (flight?.potholes || [])
      .filter((p) => show[p.severity] && !(hideUnverified && p.unverified))
      .sort((a, b) => SEV_RANK[a.severity] - SEV_RANK[b.severity] || b.diameter_m - a.diameter_m),
    [flight, show, hideUnverified],
  )
  const selected = flight?.potholes.find((p) => p.id === potholeId) || null
  const select = (p) => go('map', p ? { flight: activeId, pothole: p.id } : { flight: activeId })

  if (error) return <div className="flights-page"><div className="error">{error}</div></div>
  if (flights && !processed.length) {
    return (
      <div className="flights-page">
        <div className="note">No processed flights yet. Upload one on the <a href="#/flights">Flights</a> page.</div>
      </div>
    )
  }

  const s = flight?.summary
  return (
    <div className="page">
      <div className="maparea">
        <BaseMap>
          {flight && <MapFocus flight={flight} selected={selected} />}
          {flight && <Polyline positions={flight.track} pathOptions={{ color: '#2458d6', weight: 3, opacity: 0.7 }} />}
          {visible.map((p) => (
            <Marker key={p.id} position={[p.lat, p.lon]} icon={pinIcon(p, p.id === selected?.id)}
              zIndexOffset={p.id === selected?.id ? 1000 : 0} eventHandlers={{ click: () => select(p) }}>
              <Tooltip direction="top" offset={[0, -8]}>{p.id} · {p.severity} · {p.diameter_m.toFixed(2)} m</Tooltip>
            </Marker>
          ))}
        </BaseMap>
      </div>

      <aside className="panel">
        <div className="row spread">
          <select value={activeId || ''} onChange={(e) => go('map', { flight: e.target.value })} aria-label="Flight"
            style={{ flex: 1, minWidth: 0 }}>
            {processed.map((f) => <option key={f.id} value={f.id}>{f.name}</option>)}
          </select>
        </div>

        {!flight && <p className="muted">Loading flight…</p>}
        {flight && selected && <Details flight={flight} p={selected} list={visible} onSelect={select} />}
        {flight && !selected && (
          <>
            <div className="muted small">
              {fmtDate(s.start)} · {fmtDuration(s.duration_s)} · {(s.distance_m / 1000).toFixed(2)} km · {s.frames} photos · {s.mean_alt_m} m altitude
            </div>
            {flight.source === 'synthetic' && (
              <div className="note">Synthetic test flight: the road and potholes are simulated, so the track does
                not line up with a real road on the basemap. Pins are the real detector's output.</div>
            )}
            <div className="stats">
              <div className="stat"><b>{s.potholes}</b><span>potholes</span></div>
              {SEVERITIES.map((k) => (
                <div className="stat" key={k}><b>{s.by_severity[k]}</b><span className="sev"><i className={`dot ${k}`} />{k}</span></div>
              ))}
            </div>

            <div className="row">
              {SEVERITIES.map((k) => (
                <button key={k} className={show[k] ? 'chip' : 'chip off'} onClick={() => setShow({ ...show, [k]: !show[k] })}
                  aria-pressed={show[k]}>
                  <i className={`dot ${k}`} />{k}
                </button>
              ))}
              <label className="row small muted" style={{ gap: 4 }}>
                <input type="checkbox" checked={hideUnverified} onChange={(e) => setHideUnverified(e.target.checked)} />
                hide unverified ({s.unverified})
              </label>
            </div>

            <div className="row spread">
              <h3>{visible.length} shown</h3>
              <div className="row">
                <a className="btn small" href={api.downloadUrl(flight.id, 'geojson')}>GeoJSON</a>
                <a className="btn small" href={api.downloadUrl(flight.id, 'csv')}>CSV</a>
              </div>
            </div>
            <div className="plist">
              {visible.map((p) => (
                <button key={p.id} onClick={() => select(p)}>
                  <b>{p.id}</b>
                  <span className="sev"><i className={`dot ${p.severity}`} />{p.severity}
                    {p.unverified && <span className="badge warn">unverified</span>}</span>
                  <span className="muted">{p.diameter_m.toFixed(2)} m</span>
                </button>
              ))}
              {!visible.length && <p className="muted" style={{ padding: 10 }}>No potholes match the filters.</p>}
            </div>
            <p className="muted small">
              Click a pin or a row for details. Detector threshold {s.detector_conf}; a pothole must be seen in
              ≥ {s.min_frames} photos; detections within {s.merge_radius_m} m are merged.
            </p>
          </>
        )}
      </aside>
    </div>
  )
}

function Details({ flight, p, list, onSelect }) {
  const i = list.findIndex((q) => q.id === p.id)
  return (
    <div className="details" style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      <div className="row spread">
        <button className="btn small" onClick={() => onSelect(null)}>← All potholes</button>
        <div className="row">
          <button className="btn small" disabled={i <= 0} onClick={() => onSelect(list[i - 1])} aria-label="Previous">‹</button>
          <button className="btn small" disabled={i < 0 || i >= list.length - 1} onClick={() => onSelect(list[i + 1])} aria-label="Next">›</button>
        </div>
      </div>
      <div className="row spread">
        <h2>Pothole {p.id}</h2>
        <span className="row">
          <span className="sev"><i className={`dot ${p.severity}`} />{p.severity}</span>
          {p.unverified && <span className="badge warn">unverified</span>}
        </span>
      </div>
      <img className="crop" src={api.cropUrl(flight.id, p.crop)} alt={`Photo crop of pothole ${p.id}`} />
      {p.unverified && (
        <div className="note">Low detector confidence ({Math.round(p.confidence * 100)} %): check this one on site.</div>
      )}
      <dl className="kv">
        <dt>Size (diameter)</dt><dd>{p.diameter_m.toFixed(2)} m</dd>
        <dt>Width × length</dt><dd>{p.width_m.toFixed(2)} × {p.length_m.toFixed(2)} m</dd>
        <dt>Area (approx.)</dt><dd>{p.area_m2.toFixed(2)} m²</dd>
        <dt>Confidence</dt><dd>{Math.round(p.confidence * 100)} % best · {Math.round(p.mean_confidence * 100)} % mean</dd>
        <dt>Seen in</dt><dd>{p.frames_seen} photos</dd>
        <dt>Time</dt><dd>{fmtDate(p.first_seen)} {fmtTime(p.first_seen)}</dd>
        <dt>Flight</dt><dd>{flight.name}</dd>
        <dt>Location</dt><dd>{p.lat.toFixed(6)}, {p.lon.toFixed(6)}</dd>
        <dt>Position spread</dt><dd>± {p.position_spread_m.toFixed(2)} m across photos</dd>
      </dl>
      <div className="row">
        <a className="btn small" href={api.imageUrl(flight.id, p.best_image)} target="_blank" rel="noreferrer">Full photo</a>
        <a className="btn small" href={`https://www.google.com/maps?q=${p.lat},${p.lon}`} target="_blank" rel="noreferrer">Open in Google Maps</a>
      </div>
    </div>
  )
}
