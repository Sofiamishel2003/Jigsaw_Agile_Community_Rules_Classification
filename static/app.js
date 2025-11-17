const modelSelect = document.getElementById('modelSelect');
const content = document.getElementById('content');

// dynamic data containers filled from /models
let availableModels = [];
let modelsMetadata = {};
// modelsData maps registryKey -> { displayName, f1, accuracy, precision, recall, auc, color, description, classes, confusion_matrix }
let modelsData = {};

// pleasant color palette for models (fallback if metadata has no color)
const PALETTE = ['#4f46e5','#f59e0b','#14b8a6','#8b5cf6','#ef4444','#0ea5e9','#10b981','#f97316'];
// metric theme colors
const METRIC_COLORS = { f1:'#1F6FEB', accuracy:'#00C2C7', precision:'#F39C12', recall:'#6B6FD6', auc:'#C51A64' };
let donutMetric = 'f1';
let RESULT_COUNTER = 0;

// color utilities
function hexToRgb(hex){
  const m = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex||'#000000');
  if(!m) return {r:0,g:0,b:0};
  return {r:parseInt(m[1],16), g:parseInt(m[2],16), b:parseInt(m[3],16)};
}
function toHex(n){ const h = Math.max(0,Math.min(255,Math.round(n))).toString(16).padStart(2,'0'); return h; }
function shadeColor(hex, percent){
  const {r,g,b} = hexToRgb(hex);
  const p = percent/100; // -100..100
  const nr = r + (p>0?(255-r)*p:r*p);
  const ng = g + (p>0?(255-g)*p:g*p);
  const nb = b + (p>0?(255-b)*p:b*p);
  return `#${toHex(nr)}${toHex(ng)}${toHex(nb)}`;
}

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
    models.forEach((name, idx) => {
      const md = modelsMetadata[name] || {};
      const met = md.metrics || md;
      const fallbackColor = PALETTE[idx % PALETTE.length];
      // populate if we have any metric info or at least metadata
      if(met && (met.f1 !== undefined || met.accuracy !== undefined || met.precision !== undefined || met.recall !== undefined || met.auc !== undefined || md)){
        modelsData[name] = {
          f1: (met.f1 ?? modelsData[name]?.f1 ?? 0),
          accuracy: (met.accuracy ?? modelsData[name]?.accuracy ?? 0),
          precision: (met.precision ?? modelsData[name]?.precision ?? 0),
          recall: (met.recall ?? modelsData[name]?.recall ?? 0),
          auc: (met.auc ?? modelsData[name]?.auc ?? 0),
          color: (md.color ?? modelsData[name]?.color ?? fallbackColor),
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
  content.innerHTML = '';

  // Summary strip (models, best f1, avg f1, avg auc, total val samples)
  const entries = Object.entries(modelsData);
  const totalModels = entries.length;
  const best = entries.reduce((acc,[n,d])=> (d.f1>acc.val?{name:n,val:d.f1}:acc), {name:'—',val:-1});
  const avgF1 = totalModels? (entries.reduce((s,[,d])=>s+(d.f1||0),0)/totalModels) : 0;
  const avgAUC = totalModels? (entries.reduce((s,[,d])=>s+(d.auc||0),0)/totalModels) : 0;
  const totalVal = entries.reduce((s,[,d])=> s + (typeof d.n_val==='number'? d.n_val : 0), 0);
  const summary = el('div',{class:'summary'},[
    el('div',{class:'summary-grid'},[
      buildSummary('Modelos', String(totalModels)),
      buildSummary('Mejor F1', best.val>=0? `${(best.val*100).toFixed(1)}% (${best.name})`:'—'),
      buildSummary('F1 Promedio', `${(avgF1*100).toFixed(1)}%`),
      buildSummary('AUC Promedio', `${(avgAUC*100).toFixed(1)}%`),
      buildSummary('Muestras Val', totalVal? String(totalVal):'—')
    ])
  ]);
  content.appendChild(summary);

  // Removed donut toggle controls per user request

  const grid = el('div',{class:'model-grid'});
  Object.entries(modelsData).forEach(([name,d])=>{
    const card = el('div',{class:'model-card'},[
      el('div',{class:'model-card-header'},[
        el('div',{},[
          el('div',{class:'model-title'},[name]),
          el('div',{class:'model-desc'},[ d.description || '—' ])
        ]),
        el('span',{class:'pill', style:`border:1px solid ${d.color};color:${d.color}`},['Modelo'])
      ]),
      el('div',{class:'kpis'},[
        buildKPI('F1', d.f1),
        buildKPI('Accuracy', d.accuracy),
        buildKPI('Precision', d.precision),
        buildKPI('Recall', d.recall)
      ]),
      el('div',{class:'model-card-body'},[
        el('div',{class:'mini-chart', id:`mini-${name}`}),
        el('div',{class:'donut', id:`donut-${name}`})
      ]),
      el('div',{class:'heatmap', id:`cm-${name}`})
    ]);
    grid.appendChild(card);
    // build charts after mount
    setTimeout(()=>{
      buildMiniBar(`mini-${name}`, d, d.color);
      buildDonut(`donut-${name}`, 'F1', d.f1, d.color);
      if(d.confusion_matrix) buildHeatmap(`cm-${name}`, d.confusion_matrix, d.color);
    },0);
  });
  content.appendChild(grid);
}

function buildKPI(label, value){
  const val = (value!==undefined && value!==null) ? `${(value*100).toFixed(1)}%` : '—';
  return el('div',{class:'kpi'},[
    el('div',{class:'label'},[label]),
    el('div',{class:'value'},[val])
  ]);
}

function buildSummary(label, value){
  return el('div',{class:'summary-kpi'},[
    el('div',{class:'label'},[label]),
    el('div',{class:'value'},[String(value)])
  ]);
}

function buildSummaryWithCaption(label, value, caption){
  return el('div',{class:'summary-kpi'},[
    el('div',{class:'label'},[label]),
    el('div',{class:'value'},[String(value)]),
    el('div',{class:'caption', title: String(caption)},[String(caption)])
  ]);
}

function buildMiniBar(containerId, d, color){
  const x = ['F1','Acc','Prec','Rec','AUC'];
  const y = [d.f1,d.accuracy,d.precision,d.recall,d.auc].map(v=>Math.max(0, (v||0)*100));
  const colors = y.map((_,i)=> shadeColor(color, (i-2)*10));
  const data = [{type:'bar',x,y,marker:{color:colors,opacity:0.95},hoverinfo:'x+y'}];
  const layout = {height:160,margin:{l:28,r:10,t:10,b:28},yaxis:{range:[0,100]},xaxis:{tickfont:{size:10}},showlegend:false};
  Plotly.newPlot(containerId, data, layout, {displayModeBar:false,responsive:true});
}

function buildDonut(containerId, label, value, color){
  const pct = Math.round(Math.max(0, Math.min(1,value||0))*100);
  const data = [{values:[pct,100-pct],labels:[`${pct}%`, ''],marker:{colors:[color,'#e2e8f0']},hole:0.7,type:'pie',textinfo:'none',hoverinfo:'none'}];
  const layout = {height:170,margin:{l:10,r:10,t:10,b:10},showlegend:false,annotations:[{text:`${pct}%`,showarrow:false,font:{size:20,color:'#0f172a'}}]};
  Plotly.newPlot(containerId, data, layout, {displayModeBar:false,responsive:true});
}

function buildDonutSmall(containerId, label, value, color){
  const pct = Math.round(Math.max(0, Math.min(1,value||0))*100);
  const el = document.getElementById(containerId);
  const w = el ? el.clientWidth : 220;
  const h = Math.max(160, Math.min(260, Math.round(w*0.85)));
  const data = [{values:[pct,100-pct],labels:['',''],marker:{colors:[color,'#e6edf5']},hole:0.6,type:'pie',textinfo:'none',hoverinfo:'none'}];
  const layout = {
    height:h,
    margin:{l:8,r:8,t:8,b:8},
    paper_bgcolor:'rgba(0,0,0,0)',
    plot_bgcolor:'#fff',
    font:{family:'Inter, Segoe UI, Arial'},
    showlegend:false,
    annotations:[
      {text:`${pct}%`,x:0.5,y:0.55,showarrow:false,align:'center',font:{size:Math.round(h/5.5),color:'#0f172a',family:'Inter, Segoe UI, Arial',weight:800}},
      {text:`${label}`,x:0.5,y:0.32,showarrow:false,align:'center',font:{size:Math.round(h/9.5),color:'#334155',family:'Inter, Segoe UI, Arial'}}
    ]
  };
  Plotly.newPlot(containerId, data, layout, {displayModeBar:false,responsive:true});
}

function addTargetBar(containerId, value, target, color, label){
  const host = document.getElementById(containerId);
  if(!host || !host.parentElement) return;
  const parent = host.parentElement;
  // remove existing bar if re-rendering
  if(host.nextSibling && host.nextSibling.classList && host.nextSibling.classList.contains('target-bar')){
    parent.removeChild(host.nextSibling);
  }
  const bar = document.createElement('div');
  bar.className = 'target-bar';
  const fill = document.createElement('div');
  fill.className = 'target-fill';
  fill.style.width = `${Math.round(Math.max(0,Math.min(1,value))*100)}%`;
  fill.style.background = color;
  bar.appendChild(fill);
  parent.insertBefore(bar, host.nextSibling);
}

function buildHeatmap(containerId, cm, color){
  // cm expected 2x2
  const data = [{
    z: cm,
    x: ['Pred:No','Pred:Sí'],
    y: ['Real:No','Real:Sí'],
    type: 'heatmap',
    colorscale: [[0,'#e2e8f0'],[1, shadeColor(color, 0)]],
    showscale:false
  }];
  const layout = {height:160,margin:{l:60,r:10,t:10,b:30}};
  Plotly.newPlot(containerId, data, layout, {displayModeBar:false,responsive:true});
}

function renderComparison(){
  content.innerHTML = '';

  const entries = Object.entries(modelsData);
  const totalModels = entries.length;
  const best = entries.reduce((acc,[n,d])=> (d.f1>acc.val?{name:n,val:d.f1}:acc), {name:'—',val:-1});
  const avg = (key)=> totalModels? (entries.reduce((s,[,d])=>s+(d[key]||0),0)/totalModels) : 0;

  const grid = el('div',{class:'dash-grid'});

  // Panel: KPI board
  const kpiPanel = el('div',{class:'panel'},[
    el('div',{class:'kpi-board'},[
      buildSummary('Modelos', String(totalModels)),
      best.val>=0 ? buildSummaryWithCaption('Mejor F1', `${(best.val*100).toFixed(1)}%`, best.name) : buildSummary('Mejor F1','—'),
      buildSummary('F1 Promedio', `${(avg('f1')*100).toFixed(1)}%`),
      buildSummary('AUC Promedio', `${(avg('auc')*100).toFixed(1)}%`)
    ])
  ]);
  grid.appendChild(kpiPanel);

  // Panel: Grouped Bar (All metrics by model)
  const barPanel = el('div',{class:'panel'},[
    el('h3',{class:'panel-title'},['Métricas por Modelo']),
    el('div',{id:'barChart', style:'height:360px;'})
  ]);
  grid.appendChild(barPanel);

  // Panel: Donut averages row
  const donutPanel = el('div',{class:'panel'},[
    el('h3',{class:'panel-title'},['Promedios Globales']),
    el('div',{class:'donut-row'},[
      el('div',{class:'donut-mini',id:'avgF1'}),
      el('div',{class:'donut-mini',id:'avgAcc'}),
      el('div',{class:'donut-mini',id:'avgPrec'}),
      el('div',{class:'donut-mini',id:'avgRec'}),
      el('div',{class:'donut-mini',id:'avgAuc'})
    ])
  ]);
  grid.appendChild(donutPanel);

  // Panel: Radar
  const radarPanel = el('div',{class:'panel'},[
    el('h3',{class:'panel-title'},['Radar por Modelo']),
    el('div',{id:'radarChart', style:'height:420px;'})
  ]);
  grid.appendChild(radarPanel);

  // Row: Ranking (left) + Heatmap (right, wider)
  const rankingPanel = el('div',{class:'panel'},[
    el('div',{style:'display:flex;justify-content:space-between;align-items:center;gap:8px;margin-bottom:6px'},[
      el('h3',{class:'panel-title', style:'margin:0'},['Ranking por Modelo']),
      (function(){
        const s = el('select',{id:'rankMetricSelect', style:'padding:6px 8px;border:1px solid #e2e8f0;border-radius:8px'},[
          el('option',{value:'f1'},['F1']),
          el('option',{value:'accuracy'},['Accuracy']),
          el('option',{value:'precision'},['Precision']),
          el('option',{value:'recall'},['Recall']),
          el('option',{value:'auc'},['AUC'])
        ]);
        return s;
      })()
    ]),
    el('div',{id:'rankF1', style:'height:420px;'})
  ]);
  grid.appendChild(rankingPanel);

  const heatPanel = el('div',{class:'panel'},[
    el('h3',{class:'panel-title'},['Mapa de Calor (Modelos × Métricas)']),
    el('div',{id:'mmHeat', style:'height:520px;'})
  ]);
  grid.appendChild(heatPanel);

  content.appendChild(grid);

  // Build charts
  const metrics = ['f1','accuracy','precision','recall','auc'];
  const metricNames = {'f1':'F1','accuracy':'Accuracy','precision':'Precision','recall':'Recall','auc':'AUC'};
  const modelOrder = Object.entries(modelsData)
    .sort((a,b)=> (b[1].f1||0) - (a[1].f1||0))
    .map(([name])=>name);
  const traces = metrics.map(m => ({
    x: modelOrder,
    y: modelOrder.map(name => (modelsData[name] && modelsData[name][m]) ? +(modelsData[name][m]*100).toFixed(2) : 0),
    name: metricNames[m],
    type: 'bar',
    marker:{color: METRIC_COLORS[m]}
  }));
  Plotly.newPlot('barChart', traces, {barmode:'group', margin:{t:30,l:40,r:20,b:40}, paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'#fff', legend:{orientation:'h'}}, {displayModeBar:false,responsive:true});

  const theta = ['Precision','Recall','F1','Accuracy','AUC'];
  const radarTraces = modelOrder.map(name => ({
    r: [modelsData[name].precision, modelsData[name].recall, modelsData[name].f1, modelsData[name].accuracy, modelsData[name].auc].map(v=>v||0),
    theta: theta,
    fill: 'toself',
    name: name
  }));
  Plotly.newPlot('radarChart', radarTraces, {polar:{radialaxis:{range:[0,1]}}, margin:{t:30}}, {displayModeBar:false,responsive:true});

  // Donut averages + target bars with new theme colors
  buildDonutSmall('avgF1','F1', avg('f1'), METRIC_COLORS.f1);
  buildDonutSmall('avgAcc','Accuracy', avg('accuracy'), METRIC_COLORS.accuracy);
  buildDonutSmall('avgPrec','Precision', avg('precision'), METRIC_COLORS.precision);
  buildDonutSmall('avgRec','Recall', avg('recall'), METRIC_COLORS.recall);
  buildDonutSmall('avgAuc','AUC', avg('auc'), METRIC_COLORS.auc);

  // Heatmap models x metrics
  buildModelsMetricsHeatmap('mmHeat', modelOrder, metrics, metricNames, 520);

  // Ranking horizontal por F1
  buildRankingBar('rankF1', modelOrder, 'f1', '#00C2C7');

  // handler to update charts by selected metric and avoid blue for F1
  const sel = document.getElementById('rankMetricSelect');
  sel.addEventListener('change', ()=>{
    updateComparisonCharts(sel.value, metrics, metricNames);
  });
}

function buildModelsMetricsHeatmap(containerId, modelNames, metrics, metricNames, height){
  const z = modelNames.map(name => metrics.map(m => (modelsData[name] && modelsData[name][m]) ? +(modelsData[name][m]) : 0));
  const data = [{
    z: z,
    x: metrics.map(m=>metricNames[m]),
    y: modelNames,
    type: 'heatmap',
    zmin: 0, zmax: 1,
    colorscale: [[0,'#e2e8f0'],[1,'#1F6FEB']],
    showscale:true
  }];
  const layout = {height: height||520, margin:{l:120,r:30,t:10,b:60}, paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'#fff', xaxis:{tickangle:0}, yaxis:{automargin:true}};
  Plotly.newPlot(containerId, data, layout, {displayModeBar:false,responsive:true});
}

function buildRankingBar(containerId, modelOrder, metric, barColor){
  const y = [...modelOrder].reverse();
  const x = y.map(name => (modelsData[name] && modelsData[name][metric]) ? +(modelsData[name][metric]*100).toFixed(2) : 0);
  const colors = barColor ? y.map(()=>barColor) : y.map(name => modelsData[name]?.color || '#1F6FEB');
  const data = [{
    type:'bar',
    x, y,
    orientation:'h',
    marker:{color:colors},
    hovertemplate:'%{y}: %{x:.1f}%<extra></extra>'
  }];
  const layout = {margin:{l:120,r:20,t:20,b:40}, xaxis:{range:[0,100]}, paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'#fff'};
  Plotly.newPlot(containerId, data, layout, {displayModeBar:false,responsive:true});
}

function updateComparisonCharts(metric, metrics, metricNames){
  const modelOrder = Object.entries(modelsData)
    .sort((a,b)=> (b[1][metric]||0) - (a[1][metric]||0))
    .map(([name])=>name);
  // grouped bars
  const traces = metrics.map(m => ({
    x: modelOrder,
    y: modelOrder.map(name => (modelsData[name] && modelsData[name][m]) ? +(modelsData[name][m]*100).toFixed(2) : 0),
    name: metricNames[m],
    type: 'bar',
    marker:{color: METRIC_COLORS[m]}
  }));
  Plotly.react('barChart', traces, {barmode:'group', margin:{t:30,l:40,r:20,b:40}, paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'#fff', legend:{orientation:'h'}});

  // radar
  const theta = ['Precision','Recall','F1','Accuracy','AUC'];
  const radarTraces = modelOrder.map(name => ({
    r: [modelsData[name].precision, modelsData[name].recall, modelsData[name].f1, modelsData[name].accuracy, modelsData[name].auc].map(v=>v||0),
    theta: theta,
    fill: 'toself',
    name: name
  }));
  Plotly.react('radarChart', radarTraces, {polar:{radialaxis:{range:[0,1]}}, margin:{t:30}});

  // heatmap
  const z = modelOrder.map(name => metrics.map(m => (modelsData[name] && modelsData[name][m]) ? +(modelsData[name][m]) : 0));
  const hmData = [{ z, x: metrics.map(m=>metricNames[m]), y: modelOrder, type:'heatmap', zmin:0, zmax:1, colorscale: [[0,'#e2e8f0'],[1,'#1F6FEB']], showscale:true }];
  Plotly.react('mmHeat', hmData, {height:520, margin:{l:120,r:30,t:10,b:60}, paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'#fff', xaxis:{tickangle:0}, yaxis:{automargin:true}});

  // ranking color: avoid blue for F1
  const rankColor = (metric==='f1') ? '#00C2C7' : METRIC_COLORS[metric] || '#00C2C7';
  buildRankingBar('rankF1', modelOrder, metric, rankColor);
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
            el('div',{class:'result-header'},[
              el('div',{class:'model-name'},[item.model]),
              el('div',{class:'model-time'},[`${item.ms}ms`])
            ]),
            el('div',{class:'prediction-sub'},[ String(item.error) ])
          ])
        ]),
        el('div',{class:'result-right'})
      ]);
    }
    const r = item.result || {};
    const meta = item.meta || {};
    const baseColor = (modelsData[item.model]?.color) || meta.color || '#10b981';
    const classes = (meta.classes || (meta.metrics && meta.metrics.classes) || modelsData[item.model]?.classes || []);
    // Custom friendly labels for binary cases when the model returns 0/1
    const CUSTOM_LABELS = { 0: 'No incumple con Community Rules', 1: 'Incumple con Community Rules' };
    let labelText = r.label;
    if(typeof r.label === 'number' && classes.length > r.label){
      labelText = classes[r.label];
    } else if(r.label === 0 || r.label === 1 || r.label === '0' || r.label === '1'){
      labelText = CUSTOM_LABELS[Number(r.label)] || String(r.label);
    }
    const confidence = (r.confidence !== undefined) ? r.confidence : (r.probs? Math.max(...r.probs) : 0);
    const percent = Math.round((confidence||0)*100);
    // choose status color for binary classification (if we can infer it)
    let statusColor = baseColor;
    let statusChipClass = 'chip';
    if((classes && classes.length===2) || (Array.isArray(r.probs) && r.probs.length===2)){
      // assume index 1 = positivo
      const labelIdx = (typeof r.label === 'number') ? r.label : (classes.indexOf(String(r.label)));
      if(labelIdx === 1){ statusColor = '#ef4444'; statusChipClass = 'chip chip-red'; }
      else { statusColor = '#10b981'; statusChipClass = 'chip chip-green'; }
    }
    const barColor = shadeColor(statusColor, Math.min(30, Math.max(-20, (percent-50)/2)));
    const donutId = `pred-donut-${(item.model||'m').toString().replace(/[^a-zA-Z0-9_-]/g,'-')}-${RESULT_COUNTER++}`;
    const cardEl = el('div',{class:'result-card', style:`border-left:6px solid ${statusColor}`},[
      el('div',{class:'result-main'},[
        el('div',{class:'model-info'},[
          el('div',{class:'result-header'},[
            el('div',{class:'model-name'},[item.model]),
            el('div',{class:'model-time'},[`${item.ms}ms`])
          ]),
          el('div',{class:'prediction-label'},[ String(labelText) ]),
          el('div',{class:'confidence-row'},[
            el('div',{class:'conf-perc'},[ `${percent}%` ]),
            el('div',{class:'conf-bar'},[ el('div',{class:'conf-bar-inner', style:`width:${percent}%;background:${barColor}`},[]) ])
          ]),
          (Array.isArray(r.probs) && r.probs.length>0)
            ? el('div',{class:'probs'},[
                ...r.probs
                  .map((p,i)=>({i,p}))
                  .sort((a,b)=>b.p-a.p)
                  .slice(0,3)
                  .map(({i,p})=>{
                    let cname;
                    if(classes && classes.length>i && classes[i]!==undefined && classes[i]!==null && classes[i]!=='' && classes[i]!==0 && classes[i]!=='0' && classes[i]!==1 && classes[i]!=='1'){
                      cname = classes[i];
                    } else {
                      cname = (i===1)? CUSTOM_LABELS[1] : (i===0? CUSTOM_LABELS[0] : `Clase ${i}`);
                    }
                    return el('span',{class: statusChipClass},[ `${cname}: ${(p*100).toFixed(1)}%` ]);
                  })
              ])
            : el('div')
        ])
      ]),
      el('div',{class:'result-right', id:donutId})
    ]);
    // build confidence donut on the right
    setTimeout(()=>{
      const confVal = (r.confidence !== undefined) ? r.confidence : (r.probs? Math.max(...r.probs) : 0);
      buildDonut(donutId, 'Conf', confVal, statusColor);
    },0);
    return cardEl;
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
});

window.addEventListener('load', async ()=>{
  await loadModels();
  renderOverview();
});
