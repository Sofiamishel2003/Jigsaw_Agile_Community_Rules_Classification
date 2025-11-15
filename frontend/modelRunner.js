function tokenize(text) {
  return text
    .toLowerCase()
    .replace(/[\W_]+/g, ' ')
    .trim()
    .split(/\s+/)
    .filter(Boolean);
}

function computeTf(tokens, vocabulary) {
  const counts = {};
  for (const t of tokens) {
    if (t in vocabulary) {
      const idx = vocabulary[t];
      counts[idx] = (counts[idx] || 0) + 1;
    }
  }
  return counts; // sparse map index->tf
}

function buildVector(counts, idf, size, norm) {
  const vec = new Array(size).fill(0.0);
  for (const [idxStr, tf] of Object.entries(counts)) {
    const idx = parseInt(idxStr, 10);
    const val = idf ? tf * idf[idx] : tf;
    vec[idx] = val;
  }
  if (norm === 'l2') {
    let sum = 0.0;
    for (let i = 0; i < vec.length; i++) sum += vec[i] * vec[i];
    const denom = Math.sqrt(sum) || 1.0;
    for (let i = 0; i < vec.length; i++) vec[i] = vec[i] / denom;
  }
  return vec;
}

function dot(coefRow, vec) {
  let s = 0.0;
  const n = Math.min(coefRow.length, vec.length);
  for (let i = 0; i < n; i++) s += coefRow[i] * (vec[i] || 0);
  return s;
}

function softmax(arr) {
  const max = Math.max(...arr);
  const exps = arr.map(x => Math.exp(x - max));
  const sum = exps.reduce((a,b) => a+b, 0);
  return exps.map(e => e / sum);
}

export async function predictFromBundle(bundleJson, text) {
  // bundleJson: parsed JSON object
  const vocab = bundleJson.vocabulary;
  const idf = bundleJson.idf;
  const norm = bundleJson.norm;
  const coef = bundleJson.coef; // array of arrays
  const intercept = bundleJson.intercept;
  const classes = bundleJson.classes;

  const tokens = tokenize(text);
  const counts = computeTf(tokens, vocab);
  const size = bundleJson.inv_vocabulary ? bundleJson.inv_vocabulary.length : Object.keys(vocab).length;
  const vec = buildVector(counts, idf, size, norm);

  // compute scores
  const scores = [];
  for (let i = 0; i < coef.length; i++) {
    const s = dot(coef[i], vec) + (intercept[i] || 0);
    scores.push(s);
  }

  let probs;
  if (scores.length === 1) {
    // binary stored as single row; apply sigmoid
    const s = scores[0];
    const p = 1 / (1 + Math.exp(-s));
    probs = [1-p, p];
    // classes may be 2 entries; map accordingly
  } else {
    probs = softmax(scores);
  }

  // pick top
  let bestIdx = 0;
  for (let i = 1; i < probs.length; i++) if (probs[i] > probs[bestIdx]) bestIdx = i;

  const label = classes[bestIdx] !== undefined ? classes[bestIdx] : String(bestIdx);
  const confidence = probs[bestIdx];

  return { label, confidence, scores, probs };
}

export default { predictFromBundle };
