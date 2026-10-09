import L from 'leaflet'
import { useEffect, useMemo, useState } from 'react'
import { Marker, Polyline, Tooltip, useMapEvents } from 'react-leaflet'
import { api } from './api.js'
import BaseMap from './BaseMap.jsx'
import { fmtDuration } from './format.js'

const STORE = 'planner.v1'
// Camera / flight defaults = the synthetic test camera (camera model not decided yet).
const DEFAULTS = { alt: 9, speed: 5, hfov: 82, imgW: 640, aspect: 1, overlap: 70 }
// The synthetic demo flight: 300 m north from its start point.
const DEMO_ROUTE = [[12.904826, 80.227419], [12.904826 + 300 / 111320, 80.227419]]

function load() {
  try { return JSON.parse(localStorage.getItem(STORE)) || {} } catch { return {} }
}

export function distM([la1, lo1], [la2, lo2]) {
  const r = Math.PI / 180
  const a = Math.sin(((la2 - la1) * r) / 2) ** 2 + Math.cos(la1 * r) * Math.cos(la2 * r) * Math.sin(((lo2 - lo1) * r) / 2) ** 2
  return 2 * 6371000 * Math.asin(Math.sqrt(a))
}

// Survey maths for a downward camera: strip width, photo spacing, counts.
export function planStats(points, s) {
  const legs = points.slice(1).map((p, i) => distM(points[i], p))
  const length = legs.reduce((a, b) => a + b, 0)
  const width = 2 * s.alt * Math.tan((s.hfov * Math.PI) / 360)       // across the road
  const along = width * s.aspect                                       // image height on the ground
  const spacing = along * (1 - s.overlap / 100)
  return {
    legs, length, width, along, spacing,
    time: s.speed > 0 ? length / s.speed : 0,
    photos: length > 0 && spacing > 0 ? Math.ceil(length / spacing) + 1 : 0,
    interval: s.speed > 0 ? spacing / s.speed : 0,
    gsdCm: (width / s.imgW) * 100,
    views: s.overlap < 100 ? 1 / (1 - s.overlap / 100) : Infinity,     // photos each spot appears in
  }
}

const wpIcon = (i) => L.divIcon({
  className: '', html: `<div class="wp ${i === 0 ? 'home' : ''}">${i === 0 ? 'H' : i}</div>`,
  iconSize: [22, 22], iconAnchor: [11, 11],
})

function ClickToAdd({ onAdd }) {
  useMapEvents({ click: (e) => onAdd([e.latlng.lat, e.latlng.lng]) })
  return null
}

export default function Planner() {
  const saved = load()
  const [points, setPoints] = useState(saved.points || [])
  const [s, setS] = useState({ ...DEFAULTS, ...saved.settings })
  const [cfgHfov, setCfgHfov] = useState(null)

  useEffect(() => {
    api.config().then((c) => setCfgHfov(c.camera.hfov_deg)).catch(() => {})
  }, [])
  useEffect(() => {
    try { localStorage.setItem(STORE, JSON.stringify({ points, settings: s })) } catch { /* storage unavailable */ }
  }, [points, s])

  const st = useMemo(() => planStats(points, s), [points, s])
  const set = (k) => (e) => setS({ ...s, [k]: e.target.value === '' ? '' : Number(e.target.value) })
  const move = (i, ll) => setPoints(points.map((p, j) => (j === i ? [ll.lat, ll.lng] : p)))
  const remove = (i) => setPoints(points.filter((_, j) => j !== i))
  const valid = [s.alt, s.speed, s.hfov, s.imgW].every((v) => v > 0) && s.overlap >= 0 && s.overlap < 100

  return (
    <div className="page">
      <div className="maparea">
        <div className="map-hint">Click the map to add waypoints · drag to move · click a waypoint to delete</div>
        <BaseMap zoom={17}>
          <ClickToAdd onAdd={(p) => setPoints([...points, p])} />
          {points.length > 1 && (
            <Polyline positions={points} pathOptions={{ color: '#2458d6', weight: Math.max(3, Math.min(40, st.width)), opacity: 0.18 }} />
          )}
          {points.length > 1 && <Polyline positions={points} pathOptions={{ color: '#2458d6', weight: 3 }} />}
          {points.map((p, i) => (
            <Marker key={`${i}-${points.length}`} position={p} icon={wpIcon(i)} draggable
              eventHandlers={{ dragend: (e) => move(i, e.target.getLatLng()), click: () => remove(i) }}>
              <Tooltip direction="top" offset={[0, -10]}>{i === 0 ? 'Take-off / home' : `Waypoint ${i}`} · click to delete</Tooltip>
            </Marker>
          ))}
        </BaseMap>
      </div>

      <aside className="panel">
        <div>
          <h2>Plan a survey route</h2>
          <p className="muted small">Draw the road to fly. The drone follows the line with the camera pointing down.</p>
        </div>

        <div className="stats" style={{ gridTemplateColumns: 'repeat(2, minmax(0, 1fr))' }}>
          <div className="stat"><b>{st.length >= 1000 ? `${(st.length / 1000).toFixed(2)} km` : `${Math.round(st.length)} m`}</b><span>route length</span></div>
          <div className="stat"><b>{valid ? fmtDuration(st.time) : '–'}</b><span>flight time (survey)</span></div>
          <div className="stat"><b>{valid ? st.photos : '–'}</b><span>photos</span></div>
          <div className="stat"><b>{valid ? `${st.width.toFixed(1)} m` : '–'}</b><span>strip width on the ground</span></div>
        </div>
        <dl className="kv">
          <dt>Photo every</dt><dd>{valid ? `${st.spacing.toFixed(1)} m · ${st.interval.toFixed(2)} s` : '–'}</dd>
          <dt>Ground detail</dt><dd>{valid ? `${st.gsdCm.toFixed(2)} cm per pixel` : '–'}</dd>
          <dt>Each spot seen in</dt><dd>{valid ? `≈ ${st.views.toFixed(1)} photos` : '–'}</dd>
          <dt>Waypoints</dt><dd>{points.length}</dd>
        </dl>
        {valid && st.width < 7 && <div className="note">The strip is narrower than a two-lane road (~7 m): fly higher or use a wider lens.</div>}
        {valid && st.views < 2 && <div className="note">Less than 2 photos per spot: potholes need ≥ 2 photos to be confirmed. Increase the overlap.</div>}

        <div className="row">
          <button className="btn small" disabled={!points.length} onClick={() => setPoints(points.slice(0, -1))}>Undo last</button>
          <button className="btn small" disabled={!points.length} onClick={() => setPoints([])}>Clear</button>
          <button className="btn small" onClick={() => setPoints(DEMO_ROUTE)}>Load demo route</button>
        </div>

        {points.length > 0 && (
          <div className="wplist">
            {points.map((p, i) => (
              <div key={i}>
                <span><b>{i === 0 ? 'H' : i}</b> {p[0].toFixed(6)}, {p[1].toFixed(6)}</span>
                <span className="muted">{i > 0 ? `${Math.round(st.legs[i - 1])} m` : 'take-off'}</span>
                <button className="btn small" onClick={() => remove(i)} aria-label={`Delete waypoint ${i}`}>✕</button>
              </div>
            ))}
          </div>
        )}

        <h3>Flight &amp; camera</h3>
        <label className="field">Altitude above road (m)<input type="number" min="1" step="0.5" value={s.alt} onChange={set('alt')} /></label>
        <label className="field">Speed (m/s)<input type="number" min="0.5" step="0.5" value={s.speed} onChange={set('speed')} /></label>
        <label className="field">Photo overlap (%)<input type="number" min="0" max="95" step="5" value={s.overlap} onChange={set('overlap')} /></label>
        <label className="field">Camera HFOV (°)<input type="number" min="10" max="170" value={s.hfov} onChange={set('hfov')} /></label>
        <label className="field">Image width (px)<input type="number" min="100" step="1" value={s.imgW} onChange={set('imgW')} /></label>
        <label className="field">Image shape
          <select value={s.aspect} onChange={set('aspect')}>
            <option value={1}>1 : 1</option>
            <option value={0.75}>4 : 3</option>
            <option value={0.5625}>16 : 9</option>
          </select>
        </label>
        <div className="row">
          <button className="btn small" onClick={() => setS(DEFAULTS)}>Reset to defaults</button>
        </div>
        <p className="muted small">
          Camera and flight controller are not chosen yet: defaults match the synthetic test camera
          ({cfgHfov ?? DEFAULTS.hfov}° HFOV, 640 px, 8-10 m altitude). Mission export (GeoJSON / QGroundControl
          .plan) will be added once the flight controller is decided. The route is kept in this browser only.
        </p>
      </aside>
    </div>
  )
}
