import { useEffect, useMemo, useRef, useState } from 'react'
import { applyThreshold, matchToLabels } from './detections.js'

const pct = (v) => `${Math.round(v * 100)}%`

export default function App() {
  const [model, setModel] = useState(null)
  const [samples, setSamples] = useState([])
  const [image, setImage] = useState(null) // {id?, title, source, note, url, labels|null}
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [elapsed, setElapsed] = useState(0)
  const [error, setError] = useState('')
  const [threshold, setThreshold] = useState(0.3)
  const [showLabels, setShowLabels] = useState(true)
  const [selected, setSelected] = useState(-1)
  const request = useRef(0)

  useEffect(() => {
    fetch('/api/model').then((r) => r.json()).then((m) => { setModel(m); setThreshold(m.threshold) })
      .catch(() => setError('Cannot reach the backend. Start it with: .venv\\Scripts\\python.exe app/backend/main.py'))
    fetch('/api/samples').then((r) => r.json()).then(setSamples).catch(() => {})
  }, [])

  useEffect(() => {
    if (!loading) return
    const start = Date.now()
    const t = setInterval(() => setElapsed((Date.now() - start) / 1000), 200)
    return () => clearInterval(t)
  }, [loading])

  async function runDetection(img, body) {
    const id = ++request.current
    setImage(img); setResult(null); setError(''); setSelected(-1); setLoading(true); setElapsed(0)
    try {
      const r = await fetch('/api/detect', { method: 'POST', body })
      const data = await r.json()
      if (id !== request.current) return // a newer image was picked meanwhile
      if (!r.ok) throw new Error(data.detail || `Request failed (${r.status})`)
      setResult(data)
    } catch (e) {
      if (id === request.current) setError(e.message)
    } finally {
      if (id === request.current) setLoading(false)
    }
  }

  function pickSample(s) {
    const body = new FormData(); body.append('sample_id', s.id)
    runDetection({ ...s, labels: s.ground_truth }, body)
  }

  function upload(file) {
    if (!file) return
    const body = new FormData(); body.append('file', file)
    runDetection({ title: file.name, source: 'Your upload', note: `${(file.size / 1e6).toFixed(1)} MB`,
      url: URL.createObjectURL(file), labels: null }, body)
  }

  const dets = useMemo(() => (result ? applyThreshold(result.detections, threshold) : []), [result, threshold])
  const labels = image?.labels ?? null
  const match = useMemo(() => (labels ? matchToLabels(dets, labels) : null), [dets, labels])

  return (
    <div className="app">
      <header>
        <div>
          <h1>Pothole Detector <span className="tag">Sprint 1 demo</span></h1>
          <p>Top-down road images → pothole boxes. YOLO11s running on this PC's CPU.</p>
        </div>
        {model && <div className="hdr-meta">model <b>{model.name}</b> · trained at {model.imgsz} px · default threshold <b>{model.threshold.toFixed(2)}</b></div>}
      </header>

      <div className="layout">
        <aside>
          <section className="card">
            <h2>Try an image</h2>
            <label className="drop" onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => { e.preventDefault(); upload(e.dataTransfer.files[0]) }}>
              <input type="file" accept="image/png,image/jpeg" onChange={(e) => upload(e.target.files[0])} />
              <b>Upload a road photo</b>
              <span>JPG or PNG, top-down view works best. Click or drop here.</span>
            </label>
            <h3>Test samples <small>(never seen in training)</small></h3>
            <div className="samples">
              {samples.map((s) => (
                <button key={s.id} className={`sample ${image?.id === s.id ? 'active' : ''}`} onClick={() => pickSample(s)}>
                  <img src={s.url} alt="" loading="lazy" />
                  <span className="s-title">{s.title}</span>
                  <span className="s-sub">{s.source} · {s.ground_truth.length} labelled</span>
                </button>
              ))}
              {!samples.length && <p className="muted">No samples. Build them with: python app/make_samples.py</p>}
            </div>
          </section>
          {model && <ModelPanel model={model} />}
        </aside>

        <main>
          <section className="card controls">
            <div className="slider">
              <label htmlFor="thr">Confidence threshold <b>{threshold.toFixed(2)}</b></label>
              <input id="thr" type="range" min={model?.raw_conf ?? 0.05} max="0.95" step="0.01" value={threshold}
                onChange={(e) => setThreshold(Number(e.target.value))} />
              <div className="slider-hints"><span>more potholes found, more false pins</span><span>fewer false pins, more missed</span></div>
            </div>
            <div className="toggles">
              {model && threshold !== model.threshold && <button className="link" onClick={() => setThreshold(model.threshold)}>Reset to {model.threshold.toFixed(2)}</button>}
              <label className={labels ? '' : 'disabled'}>
                <input type="checkbox" checked={showLabels} disabled={!labels} onChange={(e) => setShowLabels(e.target.checked)} />
                Show labelled potholes
              </label>
            </div>
          </section>

          {error && <div className="card error">{error}</div>}
          {!image && !error && <div className="card empty">Pick a test sample or upload an image to run the detector.</div>}

          {image && (
            <section className="card viewer-card">
              <div className="verdict-row">
                <Verdict loading={loading} elapsed={elapsed} result={result} dets={dets} labels={labels} match={match} />
                <div className="img-meta">
                  <b>{image.title}</b><span>{image.source}{image.note ? ` · ${image.note}` : ''}</span>
                </div>
              </div>
              <div className="viewer">
                <img src={image.url} alt={image.title} />
                {result && (
                  <div className="overlay">
                    {showLabels && labels?.map((g, j) => (
                      <div key={`g${j}`} className={`box gt ${match?.labelsFound.has(j) ? 'gt-hit' : ''}`} style={boxStyle(g, result)} />
                    ))}
                    {dets.map((d, i) => (
                      <div key={i} className={`box det ${match && !match.hit[i] ? 'det-miss' : ''} ${selected === i ? 'sel' : ''}`}
                        style={boxStyle(d.box, result)} onMouseEnter={() => setSelected(i)} onMouseLeave={() => setSelected(-1)}>
                        <span>{d.confidence.toFixed(2)}</span>
                      </div>
                    ))}
                  </div>
                )}
                {loading && <div className="busy"><div className="spinner" />Running the model… {elapsed.toFixed(0)} s</div>}
              </div>
              <Legend labels={labels && showLabels} />
              {result && <DetectionTable dets={dets} match={match} selected={selected} setSelected={setSelected} />}
            </section>
          )}
        </main>
      </div>
    </div>
  )
}

function boxStyle([x0, y0, x1, y1], { width, height }) {
  return { left: pct(x0 / width), top: pct(y0 / height), width: pct((x1 - x0) / width), height: pct((y1 - y0) / height) }
}

function Verdict({ loading, elapsed, result, dets, labels, match }) {
  if (loading) return <div className="verdict wait">Detecting… {elapsed.toFixed(0)} s <small>(4K frames take ~20 s on CPU)</small></div>
  if (!result) return <div className="verdict wait">—</div>
  const time = result.cached ? 'cached result' : `${result.seconds} s on CPU`
  return (
    <div className={`verdict ${dets.length ? 'yes' : 'no'}`}>
      <div className="big">{dets.length ? `${dets.length} pothole${dets.length > 1 ? 's' : ''} detected` : 'No potholes above threshold'}</div>
      <small>
        {labels && `found ${match.found} of ${labels.length} labelled · ${dets.length - match.found} other detection${dets.length - match.found === 1 ? '' : 's'} · `}
        {result.width}×{result.height} px · {result.tiles} tile{result.tiles > 1 ? 's' : ''} · {time}
      </small>
    </div>
  )
}

function Legend({ labels }) {
  return (
    <div className="legend">
      <span><i className="sw det" /> model detection (confidence)</span>
      {labels && <><span><i className="sw det-miss" /> detection not matching a label</span>
        <span><i className="sw gt" /> labelled pothole</span></>}
    </div>
  )
}

function DetectionTable({ dets, match, selected, setSelected }) {
  if (!dets.length) return null
  return (
    <table className="dets">
      <thead><tr><th>#</th><th>Confidence</th><th>Size (px)</th><th>Centre (px)</th>{match && <th>Matches a label?</th>}</tr></thead>
      <tbody>
        {dets.map((d, i) => {
          const [x0, y0, x1, y1] = d.box
          return (
            <tr key={i} className={selected === i ? 'sel' : ''} onMouseEnter={() => setSelected(i)} onMouseLeave={() => setSelected(-1)}>
              <td>{i + 1}</td>
              <td><span className="bar" style={{ width: pct(d.confidence) }} />{d.confidence.toFixed(2)}</td>
              <td>{Math.round(x1 - x0)} × {Math.round(y1 - y0)}</td>
              <td>{Math.round((x0 + x1) / 2)}, {Math.round((y0 + y1) / 2)}</td>
              {match && <td>{match.hit[i] ? 'yes' : 'no'}</td>}
            </tr>
          )
        })}
      </tbody>
    </table>
  )
}

function ModelPanel({ model }) {
  const t = model.test
  return (
    <section className="card model">
      <h2>About the model</h2>
      <p className="muted">{model.architecture}. Trained on {model.trained_on}.</p>
      <h3>Test mAP50 <small>({t.images} images, half pothole-free)</small></h3>
      <ul className="bars">
        <li><span>All test images</span><i style={{ width: pct(t.mAP50) }} /><b>{t.mAP50.toFixed(2)}</b></li>
        {Object.entries(t.per_source_mAP50).map(([k, v]) => (
          <li key={k}><span>{k}</span><i style={{ width: pct(v) }} /><b>{v.toFixed(2)}</b></li>
        ))}
      </ul>
      <h3>Full 4K frames <small>({model.full_frame.note})</small></h3>
      <table className="mini">
        <thead><tr><th>Threshold</th><th>Found</th><th>False pins / frame</th></tr></thead>
        <tbody>{model.full_frame.rows.map((r) => (
          <tr key={r.threshold} className={r.threshold === model.threshold ? 'hl' : ''}>
            <td>{r.threshold.toFixed(2)}{r.threshold === model.threshold ? ' (default)' : ''}</td><td>{pct(r.found)}</td><td>{r.false_pins_per_frame}</td>
          </tr>))}
        </tbody>
      </table>
      <h3>Known weaknesses</h3>
      <ul className="weak">{model.weaknesses.map((w) => <li key={w}>{w}</li>)}</ul>
    </section>
  )
}
