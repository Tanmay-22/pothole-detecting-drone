// Same post-processing as ml/predict.py, so the threshold slider shows exactly what
// `predict.py --conf <threshold>` would return for this image.

// Port of merge_fragments(): merge boxes where most of the smaller box lies inside the other
// (one pothole split across overlapping tiles). Keeps the union box and the highest confidence.
export function mergeFragments(dets, minOverlap = 0.5) {
  let out = [...dets].sort((a, b) => b.confidence - a.confidence).map((d) => ({ ...d, box: [...d.box] }))
  let merged = true
  while (merged) {
    merged = false
    outer: for (let i = 0; i < out.length; i++) {
      for (let j = i + 1; j < out.length; j++) {
        const [a, o] = [out[i].box, out[j].box]
        const iw = Math.min(a[2], o[2]) - Math.max(a[0], o[0])
        const ih = Math.min(a[3], o[3]) - Math.max(a[1], o[1])
        if (iw <= 0 || ih <= 0) continue
        const smaller = Math.min((a[2] - a[0]) * (a[3] - a[1]), (o[2] - o[0]) * (o[3] - o[1]))
        if (iw * ih >= minOverlap * smaller) {
          out[i] = {
            box: [Math.min(a[0], o[0]), Math.min(a[1], o[1]), Math.max(a[2], o[2]), Math.max(a[3], o[3])],
            confidence: Math.max(out[i].confidence, out[j].confidence),
          }
          out.splice(j, 1)
          merged = true
          break outer
        }
      }
    }
  }
  return out
}

export function applyThreshold(rawDetections, threshold) {
  return mergeFragments(rawDetections.filter((d) => d.confidence >= threshold))
}

function iou(a, b) {
  const iw = Math.min(a[2], b[2]) - Math.max(a[0], b[0])
  const ih = Math.min(a[3], b[3]) - Math.max(a[1], b[1])
  if (iw <= 0 || ih <= 0) return 0
  const inter = iw * ih
  return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
}

// Greedy matching like the full-frame evaluation (IoU >= 0.3): which detections hit a labelled pothole.
export function matchToLabels(dets, labels, minIou = 0.3) {
  const used = new Set()
  const hit = dets.map(() => false)
  dets.forEach((d, i) => {
    let best = -1
    let bestIou = minIou
    labels.forEach((g, j) => {
      const v = iou(d.box, g)
      if (!used.has(j) && v >= bestIou) { best = j; bestIou = v }
    })
    if (best >= 0) { used.add(best); hit[i] = true }
  })
  return { hit, found: used.size, labelsFound: used }
}
