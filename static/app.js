const modelSelect = document.getElementById('modelSelect');
const content = document.getElementById('content');

// dynamic data containers filled from /models
let availableModels = [];
let modelsMetadata = {};
// modelsData maps registryKey -> { displayName, f1, accuracy, precision, recall, auc, color, description, classes, confusion_matrix }
let modelsData = {};

function el(tag, props = {}, children = []){
  const e = document.createElement(tag);
  Object.entries(props).forEach(([k,v])=>{ if(k==='class') e.className = v; else if(k==='html') e.innerHTML = v; else e.setAttribute(k,v)});
  children.flat().forEach(c => { if(typeof c === 'string') e.appendChild(document.createTextNode(c)); else if(c) e.appendChild(c)});
  return e;
}

async function loadModels(){
  try{
    const res = await fetch('/models');
    if(!res.ok) return [];
    const data = await res.json();
    const models = data.models || [];
    modelsMetadata = data.metadata || {};
    // build modelsData from metadata when available; support nested `metrics` key
    models.forEach(name => {
      const md = modelsMetadata[name] || {};
      const met = md.metrics || md;
      // populate if we have any metric info or at least metadata
      if(met && (met.f1 !== undefined || met.accuracy !== undefined || met.precision !== undefined || met.recall !== undefined || met.auc !== undefined || md)){
        modelsData[name] = {
          f1: (met.f1 ?? modelsData[name]?.f1 ?? 0),
          accuracy: (met.accuracy ?? modelsData[name]?.accuracy ?? 0),
          precision: (met.precision ?? modelsData[name]?.precision ?? 0),
          recall: (met.recall ?? modelsData[name]?.recall ?? 0),
          auc: (met.auc ?? modelsData[name]?.auc ?? 0),
          color: (md.color ?? modelsData[name]?.color ?? '#64748b'),
          description: (md.description ?? modelsData[name]?.description ?? ''),
          classes: (met.classes ?? md.classes ?? modelsData[name]?.classes ?? []),
          confusion_matrix: (met.confusion_matrix ?? met.confusionMatrix ?? md.confusion_matrix ?? md.confusionMatrix ?? modelsData[name]?.confusion_matrix ?? null),
          n_val: ((met.n_val ?? met.n) ?? (md.n_val ?? md.n) ?? modelsData[name]?.n_val ?? null)
        };
      }
    });
    availableModels = models;
    // fill select
    modelSelect.innerHTML = '';
    // add All Models option
    modelSelect.appendChild(el('option',{value:'ALL'},['All Models']));
    models.forEach(m => modelSelect.appendChild(el('option',{value:m},[m])));
    if(models.length===0) modelSelect.appendChild(el('option',{value:''},['(no hay modelos)']));
    return models;
  }catch(e){
    console.warn('loadModels error', e);
    modelSelect.innerHTML = '';
    modelSelect.appendChild(el('option',{value:''},['(error)']));
    return [];
  }
}

// ---- render functions for tabs ----
function renderOverview(){
  const header = el('div',{class:'card'},[
    el('div',{class:'grid-cards'},
      Object.entries(modelsData).map(([name,d]) => (
        el('div',{class:'card-metric', style:`border-top:4px solid ${d.color}`},[
          el('div',{class:'metric-title'},[name]),
          el('div',{class:'metric-value'},[`${(d.f1*100).toFixed(1)}%`]),
          el('div',{class:'small'},[d.description])
        ])
      ))
    )
  ]);

  const distribution = el('div',{class:'card'},[
    el('h3',{},['Distribución del Dataset']),
    el('div',{html:'<div style="display:flex;gap:12px;margin-top:12px"><div style="flex:1"><strong>Training</strong><div>1623 muestras</div></div><div style="flex:1"><strong>Validation</strong><div>406 muestras</div></div><div style="flex:1"><strong>Test</strong><div>10 muestras</div></div></div>'})
  ]);

  content.innerHTML = '';
  content.appendChild(header);
  content.appendChild(distribution);
}

function renderComparison(){
  content.innerHTML = '';
  const card = el('div',{class:'card'},[
    el('h3',{},['Comparación de Métricas']),
    el('div',{id:'comparisonCharts'})
  ]);
  content.appendChild(card);

  // build grouped bar chart using Plotly
  const metrics = ['f1','accuracy','precision','recall','auc'];
  const metricNames = {'f1':'F1','accuracy':'Accuracy','precision':'Precision','recall':'Recall','auc':'AUC'};
  const modelNames = Object.keys(modelsData);
  const traces = metrics.map(m => ({
    x: modelNames,
    y: modelNames.map(name => (modelsData[name] && modelsData[name][m]) ? +(modelsData[name][m]*100).toFixed(2) : 0),
    name: metricNames[m],
    type: 'bar'
  }));
  const barDiv = el('div',{id:'barChart', style:'height:360px;'});
  document.getElementById('comparisonCharts').appendChild(barDiv);
  Plotly.newPlot(barDiv, traces, {barmode:'group', margin:{t:30}, legend:{orientation:'h'}});

  // radar plot (precision, recall, f1, accuracy, auc) per model
  const radarDiv = el('div',{id:'radarChart', style:'height:420px;margin-top:18px'});
  document.getElementById('comparisonCharts').appendChild(radarDiv);
  const theta = ['Precision','Recall','F1','Accuracy','AUC'];
  const radarTraces = Object.keys(modelsData).map(name => ({
    r: [modelsData[name].precision, modelsData[name].recall, modelsData[name].f1, modelsData[name].accuracy, modelsData[name].auc].map(v=>v||0),
    theta: theta,
    fill: 'toself',
    name: name
  }));
  Plotly.newPlot(radarDiv, radarTraces, {polar:{radialaxis:{range:[0,1]}}, margin:{t:30}});
}

function generateComparisonHTML(){
  // simple bar-like visual using div widths
  const metrics = ['F1','Accuracy','Precision','Recall','AUC'];
  let html = '<div style="display:flex;flex-direction:column;gap:12px;margin-top:12px">';
  metrics.forEach(m =>{
    html += `<div><strong>${m}</strong><div style="display:flex;gap:8px;margin-top:6px">`;
    Object.entries(modelsData).forEach(([name,d])=>{
      const val = Math.round((d[m.toLowerCase()] || d[m] || 0) * 100);
      html += `<div style="flex:1"><div style="font-size:12px;margin-bottom:6px">${name} — ${val}%</div><div style="height:10px;background:#eef2ff;border-radius:8px"><div style="height:10px;width:${val}%;background:${d.color};border-radius:8px"></div></div></div>`;
    })
    html += '</div></div>';
  })
  html += '</div>';
  return html;
}

function renderPredict(){
  const area = el('div',{class:'card'},[
    el('h3',{},['Predicción en Tiempo Real']),
    el('label',{},['Ingrese texto']),
    el('textarea',{id:'inputText',rows:6,style:'width:100%;padding:12px;border-radius:8px;border:1px solid #eef2ff'},[]),
    el('div',{style:'display:flex;gap:12px;align-items:center'},[
      el('div',{style:'flex:1'},[]),
      el('div',{},[el('button',{id:'analyzeBtn'},['Analizar'])])
    ]),
    el('div',{id:'resultsList',class:'results-list'},[])
  ]);

  content.innerHTML = '';
  content.appendChild(area);

  document.getElementById('analyzeBtn').addEventListener('click', async ()=>{
    const text = document.getElementById('inputText').value.trim();
    const selection = modelSelect.value || 'ALL';
    const resultsList = document.getElementById('resultsList');
    resultsList.innerHTML = '';
    if(!text){ resultsList.appendChild(el('div',{class:'muted'},['Ingrese texto para analizar.'])); return; }
    const btn = document.getElementById('analyzeBtn');
    btn.disabled = true;
    try{
      if(selection === 'ALL'){
        const models = availableModels.length? availableModels : Object.keys(modelsData);
        const promises = models.map(async m => {
          const start = performance.now();
          try{
            const res = await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text, model:m})});
            const ms = Math.max(1, Math.round(performance.now() - start));
            if(!res.ok){ const txt = await res.text(); return {model:m, error: txt, ms}; }
            const data = await res.json();
            const r = data.result || data;
            return {model:m, result:r, ms, meta: modelsMetadata[m] || {}};
          }catch(e){ return {model:m, error:String(e), ms: Math.max(1, Math.round(performance.now()-start))} }
        });
        const all = await Promise.all(promises);
        all.forEach(item => resultsList.appendChild(buildResultCard(item)));
      } else {
        const start = performance.now();
        const res = await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text, model:selection})});
        const ms = Math.max(1, Math.round(performance.now()-start));
        if(!res.ok){ resultsList.appendChild(el('div',{class:'muted'},['Error: '+ await res.text()])); return; }
        const data = await res.json();
        const r = data.result || data;
        resultsList.appendChild(buildResultCard({model:selection,result:r,ms,meta: modelsMetadata[selection]||{}}));
      }
    }catch(e){ resultsList.appendChild(el('div',{class:'muted'},['Error comunicando con /predict: '+String(e)])); }
    finally{ btn.disabled = false }
  });

  function buildResultCard(item){
    if(item.error){
      return el('div',{class:'result-card', style:`border-left:6px solid #ef4444`},[
        el('div',{class:'result-main'},[
          el('div',{class:'model-info'},[
            el('div',{class:'model-name'},[item.model]),
            el('div',{class:'prediction-sub'},[ String(item.error) ])
          ])
        ]),
        el('div',{class:'model-time'},[`${item.ms}ms`])
      ]);
    }
    const r = item.result || {};
    const meta = item.meta || {};
    const color = (modelsData[item.model]?.color) || meta.color || '#10b981';
    const classes = (meta.classes || (meta.metrics && meta.metrics.classes) || modelsData[item.model]?.classes || []);
    let labelText = r.label;
    if(typeof r.label === 'number' && classes.length > r.label) labelText = classes[r.label];
    const confidence = (r.confidence !== undefined) ? r.confidence : (r.probs? Math.max(...r.probs) : 0);
    const percent = Math.round((confidence||0)*100);
    return el('div',{class:'result-card', style:`border-left:6px solid ${color}`},[
      el('div',{class:'result-main'},[
        el('div',{class:'model-info'},[
          el('div',{style:'display:flex;align-items:center;gap:8px'},[
            el('div',{class:'model-name'},[item.model]),
            el('div',{class:'model-time'},[`${item.ms}ms`])
          ]),
          el('div',{class:'prediction-label'},[ String(labelText) ]),
          el('div',{class:'prediction-sub'},[ `Confianza: ${percent}%` ]),
          el('div',{class:'confidence-row'},[
            el('div',{class:'conf-perc'},[ `${percent}%` ]),
            el('div',{class:'conf-bar'},[ el('div',{class:'conf-bar-inner', style:`width:${percent}%;background:${color}`},[]) ])
          ])
        ])
      ])
    ]);
  }
}

function renderPerformance(){
  const card = el('div',{class:'card'},[
    el('h3',{},['Métricas Detalladas']),
    el('div',{html:generatePerformanceHTML()})
  ]);
  content.innerHTML = '';
  content.appendChild(card);
}

function generatePerformanceHTML(){
  let html = '<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">';
  Object.entries(modelsData).forEach(([name,d])=>{
    html += `<div style="border-left:6px solid ${d.color};padding:12px;border-radius:8px;background:#fff"><strong>${name}</strong><div style="margin-top:8px;font-size:13px">Precision: ${(d.precision*100).toFixed(1)}%<br>Recall: ${(d.recall*100).toFixed(1)}%<br>F1: ${(d.f1*100).toFixed(1)}%</div></div>`;
  });
  html += '</div>';
  return html;
}

// tab switching
document.addEventListener('click', (e)=>{
  const t = e.target.closest('.tab');
  if(!t) return;
  document.querySelectorAll('.tab').forEach(b=>b.classList.remove('active'));
  t.classList.add('active');
  const tab = t.dataset.tab;
  if(tab==='overview') renderOverview();
  if(tab==='comparison') renderComparison();
  if(tab==='predict') renderPredict();
  if(tab==='performance') renderPerformance();
});

window.addEventListener('load', async ()=>{
  await loadModels();
  renderOverview();
});
