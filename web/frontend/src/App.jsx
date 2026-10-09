import { useEffect, useState } from 'react'
import Flights from './Flights.jsx'
import Planner from './Planner.jsx'
import ResultsMap from './ResultsMap.jsx'

const TABS = [
  { id: 'plan', label: 'Plan route' },
  { id: 'map', label: 'Results map' },
  { id: 'flights', label: 'Flights' },
]

// Route = location hash: #/map?flight=<id>, #/plan, #/flights
function parseHash() {
  const [path, query = ''] = window.location.hash.replace(/^#\/?/, '').split('?')
  return { tab: TABS.some((t) => t.id === path) ? path : 'map', params: new URLSearchParams(query) }
}

export function go(tab, params = {}) {
  const q = new URLSearchParams(params).toString()
  window.location.hash = `/${tab}${q ? `?${q}` : ''}`
}

export default function App() {
  const [route, setRoute] = useState(parseHash)
  useEffect(() => {
    const on = () => setRoute(parseHash())
    window.addEventListener('hashchange', on)
    return () => window.removeEventListener('hashchange', on)
  }, [])

  return (
    <div className="shell">
      <header>
        <div className="brand">
          <span className="logo" aria-hidden>◉</span>
          <div>
            <h1>Pothole Drone</h1>
            <p>Road survey dashboard · demo</p>
          </div>
        </div>
        <nav>
          {TABS.map((t) => (
            <button key={t.id} className={route.tab === t.id ? 'tab active' : 'tab'} onClick={() => go(t.id)}>
              {t.label}
            </button>
          ))}
        </nav>
      </header>
      <main>
        {route.tab === 'plan' && <Planner />}
        {route.tab === 'map' && <ResultsMap flightId={route.params.get('flight')} potholeId={route.params.get('pothole')} />}
        {route.tab === 'flights' && <Flights />}
      </main>
    </div>
  )
}
