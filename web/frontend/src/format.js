// Display helpers (timestamps are unix seconds).
export const fmtDate = (ts) => (ts ? new Date(ts * 1000).toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric' }) : '–')
export const fmtTime = (ts) => (ts ? new Date(ts * 1000).toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit', second: '2-digit' }) : '')

export function fmtDuration(s) {
  if (s == null) return '–'
  const m = Math.floor(s / 60)
  const r = Math.round(s % 60)
  return m ? `${m} min ${r} s` : `${r} s`
}
