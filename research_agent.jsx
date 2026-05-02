const { useEffect, useMemo, useRef, useState, useCallback } = React;
const API = "/api";

/* ════════════════════════════════════════════════════════
   MODE CONFIG — Same 5 modes as landing page
   ════════════════════════════════════════════════════════ */
const MODES = [
  { id:"local",      name:"Local Brain",  glyph:"L", desc:"Paper-grounded QA with citations" },
  { id:"global",     name:"Global Brain", glyph:"G", desc:"Broad reasoning + paper context" },
  { id:"writer",     name:"Paper Writer", glyph:"W", desc:"Style-aware drafting engine" },
  { id:"reviewer",   name:"Reviewer",     glyph:"R", desc:"Single-paper deep analysis" },
  { id:"comparator", name:"Comparator",   glyph:"C", desc:"Multi-paper comparison engine" },
];
const CHAT_IDS = new Set(["local","global","writer"]);

const COMPARE_PRESETS = [
  { id:"full",      label:"Full Verdict",  text:"Run a full comparator pass with claim matrix, conflict map, benchmark verdict matrix, and decision by use case." },
  { id:"conflict",  label:"Conflict Map",  text:"Focus on agreements, contradictions, and non-overlap across selected papers, then state what evidence resolves each conflict." },
  { id:"synthesis",  label:"Synthesis",     text:"Build a synthesis blueprint that combines the strongest parts of each paper and proposes one merged experiment." },
];
const FILE_ST = { IDLE:"idle", UPLOADING:"uploading", INDEXING:"indexing", INDEXED:"indexed", ERROR:"error" };

/* ═══ Utilities ═══ */
function modeOf(id) { return MODES.find(m=>m.id===id)||MODES[1]; }
function sid() { return crypto?.randomUUID?.() || `s-${Date.now()}`; }
function clip(t,l) { return t&&t.length>l?t.slice(0,l)+"…":(t||""); }
function compTextById(id) { return (COMPARE_PRESETS.find(p=>p.id===id)||COMPARE_PRESETS[0]).text; }
function compLabelById(id) { return (COMPARE_PRESETS.find(p=>p.id===id)||COMPARE_PRESETS[0]).label; }

function normSnip(t) { return (t||"").replace(/\s+/g," ").trim(); }
function normCites(c) { const s=new Set,o=[]; for(const x of c||[]){ const k=[x.paper_id||"",x.chunk_id||"",x.filename||""].join("|"); if(s.has(k))continue; s.add(k); o.push({...x,snippet:normSnip(x.snippet||"")}); } return o; }
function simplifyErr(d) { const t=String(d||"").trim(); if(/rate limit/i.test(t)&&/groq|tokens/i.test(t))return "Groq quota reached."; return t||"Request failed."; }
function hasRevConvo(h) { return (h||[]).some(i=>i.role==="assistant"&&i.mode==="reviewer"); }

/* ════════════════════════════════════════════════════════
   LaTeX + Markdown — compact renderers
   ════════════════════════════════════════════════════════ */
function renderLatex(e,{displayMode=false,key}={}) {
  let l=String(e||"").trim().replace(/\\\\(?=[A-Za-z])/g,"\\"); if(!l)return null;
  const K=window.katex;
  if(K?.renderToString){try{const h=K.renderToString(l,{displayMode,throwOnError:false,strict:"ignore"});return displayMode?<div key={key} className="my-3 overflow-x-auto rounded-lg border border-[var(--border)] bg-[rgba(0,255,159,0.02)] p-3" dangerouslySetInnerHTML={{__html:h}}/>:<span key={key} className="inline-block rounded px-1 py-0.5 border border-[var(--border)] bg-[rgba(0,255,159,0.02)]" dangerouslySetInnerHTML={{__html:h}}/>;}catch(_){}}
  return displayMode?<div key={key} className="my-3 rounded-lg border border-[var(--border)] bg-[rgba(0,255,159,0.02)] p-3"><pre className="text-emerald-100 font-serif whitespace-pre-wrap">{l}</pre></div>:<code key={key} className="font-serif italic text-emerald-100">{l}</code>;
}
function rInline(text,kp="s") {
  return String(text||"").split(/(\*\*[^*]+\*\*|`[^`]+`|\$[^$\n]+\$)/g).filter(Boolean).map((t,i)=>{
    if(/^\*\*[^*]+\*\*$/.test(t))return <strong key={`${kp}-${i}`} className="text-[var(--text)] font-semibold">{t.slice(2,-2)}</strong>;
    if(/^`[^`]+`$/.test(t))return <code key={`${kp}-${i}`} className="px-1.5 py-0.5 rounded-md bg-[rgba(0,255,159,0.06)] border border-[rgba(0,255,159,0.12)] text-[var(--green)] font-mono text-xs">{t.slice(1,-1)}</code>;
    if(/^\$[^$\n]+\$$/.test(t))return renderLatex(t.slice(1,-1),{displayMode:false,key:`${kp}-${i}`});
    return <React.Fragment key={`${kp}-${i}`}>{t}</React.Fragment>;
  });
}
function renderMd(content) {
  const lines=String(content||"").split("\n"),blocks=[];
  let lt=null,li=[],tr=[],inM=false,ml=[],md="$$",k=0;
  const fL=()=>{if(!lt||!li.length)return;const T=lt==="ol"?"ol":"ul";blocks.push(<T key={`l-${k++}`} className={`mb-3 ${lt==="ol"?"list-decimal":"list-disc"} pl-5 space-y-1 text-[var(--text-muted)]`}>{li.map((x,i)=><li key={i} className="leading-relaxed">{rInline(x)}</li>)}</T>);lt=null;li=[];};
  const pTC=r=>String(r||"").trim().replace(/^\|/,"").replace(/\|$/,"").split("|").map(c=>c.trim());
  const isS=c=>c.length>0&&c.every(x=>/^:?-{3,}:?$/.test(x));
  const fT=()=>{if(!tr.length)return;const pr=tr.map(pTC).filter(c=>c.length>0);tr=[];const rows=pr.filter(c=>!isS(c));if(!rows.length)return;const h=rows[0],b=rows.slice(1);blocks.push(<div key={`t-${k++}`} className="my-3 overflow-x-auto rounded-2xl border border-[var(--border)]"><table className="w-full min-w-[500px] border-collapse"><thead><tr>{h.map((c,i)=><th key={i} className="border-b border-[var(--border)] bg-[rgba(0,255,159,0.03)] px-3 py-2 text-left text-xs font-bold text-white uppercase tracking-wider">{rInline(c)}</th>)}</tr></thead>{b.length?<tbody>{b.map((row,ri)=><tr key={ri} className="border-b border-[var(--border)] last:border-0">{h.map((_,ci)=><td key={ci} className="px-3 py-2 text-xs text-[var(--text-muted)]">{rInline(row[ci]||"")}</td>)}</tr>)}</tbody>:null}</table></div>);};
  const fM=()=>{if(!ml.length)return;const l=ml.join("\n").trim();ml=[];if(l)blocks.push(renderLatex(l,{displayMode:true,key:`m-${k++}`}));};
  for(const line of lines){const t=line.trim();if(inM){if((md==="$$"&&t==="$$")||(md==="\\["&&t==="\\]")){fM();inM=false;md="$$";}else ml.push(line);continue;}const sdm=t.match(/^\$\$(.+)\$\$$/),sbm=t.match(/^\\\[(.+)\\\]$/);if(sdm||sbm){fL();fT();const l=(sdm?sdm[1]:sbm[1]).trim();if(l)blocks.push(renderLatex(l,{displayMode:true,key:`ms-${k++}`}));continue;}if(t==="$$"||t==="\\["){fL();fT();inM=true;md=t==="\\["?"\\[":"$$";ml=[];continue;}if(!t){fL();fT();continue;}if(/^\|.*\|$/.test(t)){fL();tr.push(t);continue;}fT();const om=t.match(/^\s*(\d+)\.\s+(.+)/);if(om){if(lt&&lt!=="ol")fL();lt="ol";li.push(om[2]);continue;}const um=t.match(/^\s*[-*+]\s+(.+)/);if(um){if(lt&&lt!=="ul")fL();lt="ul";li.push(um[1]);continue;}fL();if(t.startsWith("### ")){blocks.push(<h3 key={`h3-${k++}`} className="text-base font-bold text-white mb-2 mt-4">{rInline(t.slice(4))}</h3>);continue;}if(t.startsWith("## ")){blocks.push(<h2 key={`h2-${k++}`} className="text-lg font-bold text-white mb-2 mt-4">{rInline(t.slice(3))}</h2>);continue;}if(t.startsWith("# ")){blocks.push(<h1 key={`h1-${k++}`} className="text-xl font-bold text-white mb-3 mt-4">{rInline(t.slice(2))}</h1>);continue;}blocks.push(<p key={`p-${k++}`} className="mb-2 text-[var(--text-muted)] leading-relaxed">{rInline(t)}</p>);}
  fM();fL();fT();
  return <div className="whitespace-normal">{blocks.length?blocks:<p className="text-[var(--text-muted)]">{content}</p>}</div>;
}

/* ═══ Comparator rendering ═══ */
const COMP_T=["Papers Compared","Claim Matrix","Conflict Map","Benchmark Verdict Matrix","Method Trade-offs","Synthesis Blueprint","Decision By Use Case"];
function normCompMd(c){let n=String(c||"").replace(/\r\n/g,"\n");for(const t of COMP_T){const e=t.replace(/[.*+?^${}()|[\]\\]/g,"\\$&");n=n.replace(new RegExp(`^\\s*${e}\\b\\s*:?(.*)$`,"gmi"),(f,tail)=>{if(f.trim().startsWith("#"))return f;const r=(tail||"").trim();return r?`## ${t}\n${r}`:`## ${t}`;});}return n;}
function splitCompSec(c){const lines=String(c||"").split("\n"),secs=[],intro=[];let ct="",cl=[];const flush=()=>{if(!ct)return;secs.push({title:ct,body:cl.join("\n").trim()});ct="";cl=[];};for(const l of lines){const m=l.trim().match(/^##\s+(.+)$/);if(m){flush();ct=m[1].trim();continue;}ct?cl.push(l):intro.push(l);}flush();return{intro:intro.join("\n").trim(),secs};}
const SC_C={"papers-compared":"border-l-emerald-400/60","claim-matrix":"border-l-[var(--green)]/60","conflict-map":"border-l-orange-400/60","benchmark-verdict-matrix":"border-l-green-400/60","method-trade-offs":"border-l-teal-400/60","synthesis-blueprint":"border-l-cyan-400/60","decision-by-use-case":"border-l-[var(--green)]/80"};
function scClass(t){return String(t||"").toLowerCase().replace(/[^a-z0-9]+/g,"-").replace(/^-+|-+$/g,"")||"general";}
function renderCompAns(c){const n=normCompMd(c);const{intro,secs}=splitCompSec(n);if(!secs.length)return renderMd(n);return(<div className="space-y-4">{intro?<div className="rounded-2xl border border-[var(--border-2)] bg-[var(--surface)] p-4">{renderMd(intro)}</div>:null}<div className="flex flex-wrap gap-1.5">{secs.map((s,i)=><span key={i} className="rounded-full bg-[rgba(0,255,159,0.08)] border border-[rgba(0,255,159,0.2)] px-3 py-1 text-[11px] font-bold text-[var(--green)] uppercase tracking-wider">{s.title}</span>)}</div><div className="grid gap-3">{secs.map((s,i)=><section key={i} className={`rounded-2xl border border-[var(--border-2)] border-l-4 ${SC_C[scClass(s.title)]||"border-l-[var(--green)]/50"} bg-[var(--surface)] p-5`}><div className="flex items-center gap-3 mb-3"><span className="flex items-center justify-center w-7 h-6 rounded-md bg-[rgba(0,255,159,0.08)] border border-[rgba(0,255,159,0.15)] text-[11px] font-bold text-[var(--green)]">{String(i+1).padStart(2,"0")}</span><h3 className="text-sm font-bold text-white">{s.title}</h3></div>{renderMd(s.body||"No detail.")}</section>)}</div></div>);}

/* ═══ Reviewer rendering ═══ */
function normRE(d){return(Array.isArray(d?.round_events)?d.round_events:[]).map(e=>({speaker:String(e?.speaker||"").trim().toLowerCase(),turn:Number(e?.turn||0),content:String(e?.content||"").trim()})).filter(e=>["skeptic","advocate","judge","synthesise"].includes(e.speaker)&&e.content);}
const SK_C={skeptic:"border-l-red-400/60",advocate:"border-l-[var(--green)]/60",judge:"border-l-yellow-400/60",synthesise:"border-l-cyan-400/60"};
const SK_L={skeptic:"Skeptic",advocate:"Advocate",judge:"Judge",synthesise:"Compiler"};
function renderDebate(d){const ev=normRE(d);if(!ev.length)return null;return(<div className="rounded-2xl border border-[var(--border-2)] bg-[var(--surface)] p-5"><div className="text-[11px] font-bold uppercase tracking-[.14em] text-[var(--green)] mb-3">Live Debate Round</div><div className="space-y-2">{ev.map((e,i)=><div key={i} className={`rounded-xl border border-[var(--border)] border-l-4 ${SK_C[e.speaker]||"border-l-[var(--green)]/30"} bg-[rgba(255,255,255,0.02)] p-3`}><div className="flex items-center justify-between mb-1"><span className="text-[10px] font-bold uppercase tracking-wider text-[var(--text-dim)]">{SK_L[e.speaker]||"Panel"}</span>{e.turn?<span className="text-[9px] text-[var(--text-dim)]">Turn {e.turn}</span>:null}</div><div className="text-xs text-[var(--text-muted)] leading-relaxed whitespace-pre-wrap">{rInline(e.content)}</div></div>)}</div></div>);}
function renderReport(r){if(!r||typeof r!=="object")return null;const ag=(r.agreements||[]).filter(Boolean),dg=(r.disagreements||[]).filter(Boolean);return(<div className="rounded-2xl border border-[rgba(0,255,159,0.2)] bg-[var(--surface)] p-5" style={{boxShadow:"0 8px 40px rgba(0,255,159,0.06)"}}><div className="flex items-center mb-4"><span className="text-xs font-bold uppercase tracking-[.14em] text-[var(--green)]">Panel Verdict</span></div><p className="text-sm text-[var(--text)] leading-relaxed mb-4">{r.overview||"Summary ready."}</p><div className="grid gap-3 sm:grid-cols-2 mb-3"><div className="rounded-xl border border-[var(--border)] bg-[rgba(255,255,255,0.02)] p-3"><h4 className="text-[10px] font-bold uppercase text-[var(--green)] opacity-60 mb-2">Agreements</h4><ul className="space-y-1 text-xs text-[var(--text-muted)]">{(ag.length?ag:["No agreements captured."]).map((x,i)=><li key={i}>• {x}</li>)}</ul></div><div className="rounded-xl border border-[var(--border)] bg-[rgba(255,255,255,0.02)] p-3"><h4 className="text-[10px] font-bold uppercase text-red-400/70 mb-2">Disagreements</h4><ul className="space-y-1 text-xs text-[var(--text-muted)]">{(dg.length?dg:["No disagreements captured."]).map((x,i)=><li key={i}>• {x}</li>)}</ul></div></div>{r.final_decision&&<div className="rounded-xl border border-[rgba(0,255,159,0.2)] bg-[rgba(0,255,159,0.04)] p-3"><h4 className="text-[10px] font-bold uppercase text-[var(--green)] mb-2">Final Decision</h4><p className="text-sm text-emerald-100 leading-relaxed">{r.final_decision}</p></div>}</div>);}

/* ═══ API ═══ */
async function api(p,o){const r=await fetch(p,o);if(!r.ok){let d=r.statusText;try{d=(await r.json()).detail||d;}catch(_){}throw new Error(simplifyErr(d));}return r.json();}

/* ════════════════════════════════════════════════════════
   ICON SYSTEM
   ════════════════════════════════════════════════════════ */
function Icon({name,className="w-5 h-5"}){
  const p={fill:"none",stroke:"currentColor",strokeWidth:1.6,strokeLinecap:"round",strokeLinejoin:"round",viewBox:"0 0 24 24"};
  const I={
    upload:<><path d="M12 16V4"/><path d="m7 9 5-5 5 5"/><path d="M5 20h14"/></>,
    local:<><path d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8L12 3Z"/></>,
    global:<><circle cx="12" cy="12" r="8.5"/><path d="M3.5 12h17"/><path d="M12 3.5c2.6 2.7 4 5.6 4 8.5s-1.4 5.8-4 8.5c-2.6-2.7-4-5.6-4-8.5s1.4-5.8 4-8.5Z"/></>,
    writer:<><path d="M12 20h9"/><path d="m16.5 3.5 4 4L8 20H4v-4L16.5 3.5Z"/></>,
    reviewer:<><path d="M12 3 5 6v5c0 4.5 2.8 8.7 7 10 4.2-1.3 7-5.5 7-10V6l-7-3Z"/><path d="m9.5 12 1.8 1.8L15 10.2"/></>,
    comparator:<><path d="M7 5h6v14H7z"/><path d="M11 8h6v11h-6z"/><path d="M3 12h4"/><path d="M17 12h4"/></>,
    send:<><path d="M22 2 11 13"/><path d="M22 2 15 22l-4-9-9-4z"/></>,
    attach:<><path d="M8.5 12.5 14.7 6.3a3 3 0 0 1 4.2 4.2l-8.8 8.8a5 5 0 1 1-7.1-7.1l9.2-9.2"/></>,
    check:<><path d="M20 6 9 17l-5-5"/></>,
    x:<><path d="M18 6 6 18"/><path d="m6 6 12 12"/></>,
    copy:<><rect x="8" y="8" width="12" height="12" rx="2"/><path d="M4 16V4a2 2 0 0 1 2-2h12"/></>,
    file:<><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></>,
    panels:<><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M15 3v18"/></>,
    play:<><polygon points="5 3 19 12 5 21 5 3" fill="currentColor" stroke="none"/></>,
    zap:<><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" fill="currentColor" stroke="none"/></>,
  };
  return <svg className={className} {...p}>{I[name]||<circle cx="12" cy="12" r="8"/>}</svg>;
}

/* ════════════════════════════════════════════════════════
   AMBIENT CANVAS — EXACT port of landing.html particle system
   Mouse-interactive, multi-layer glow, connection lines
   ════════════════════════════════════════════════════════ */
function AmbientCanvas() {
  const ref = useRef(null);
  useEffect(() => {
    const canvas = ref.current, ctx = canvas.getContext("2d");
    let animId, particles = [];
    const PARTICLE_COUNT = 70;
    const CONNECTION_DISTANCE = 140;
    let mouse = { x: null, y: null, radius: 150 };

    function resize() { canvas.width = window.innerWidth; canvas.height = window.innerHeight; }

    function onMouseMove(e) { mouse.x = e.clientX; mouse.y = e.clientY; }
    function onMouseOut() { mouse.x = null; mouse.y = null; }

    class Particle {
      constructor() {
        this.x = Math.random() * canvas.width;
        this.y = Math.random() * canvas.height;
        this.vx = (Math.random() - 0.5) * 0.35;
        this.vy = (Math.random() - 0.5) * 0.35;
        this.radius = Math.random() * 2.5 + 1;
        this.density = Math.random() * 30 + 1;
      }
      update() {
        this.x += this.vx; this.y += this.vy;
        if (this.x < 0 || this.x > canvas.width) this.vx *= -1;
        if (this.y < 0 || this.y > canvas.height) this.vy *= -1;
        if (mouse.x != null && mouse.y != null) {
          let dx = mouse.x - this.x, dy = mouse.y - this.y;
          let dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < mouse.radius) {
            let force = (mouse.radius - dist) / mouse.radius;
            this.x -= (dx / dist) * force * this.density * 0.4;
            this.y -= (dy / dist) * force * this.density * 0.4;
          }
        }
      }
      draw() {
        let opacity = 0.5, glow = 3;
        if (mouse.x != null && mouse.y != null) {
          let dx = mouse.x - this.x, dy = mouse.y - this.y, dist = Math.sqrt(dx*dx+dy*dy);
          if (dist < mouse.radius) { let i = (mouse.radius-dist)/mouse.radius; opacity = 0.5 + i*0.4; glow = 3 + i*3.5; }
        }
        ctx.beginPath(); ctx.arc(this.x, this.y, this.radius * glow, 0, Math.PI*2);
        ctx.fillStyle = `rgba(0,255,159,${opacity*0.08})`; ctx.fill();
        ctx.beginPath(); ctx.arc(this.x, this.y, this.radius * 1.4, 0, Math.PI*2);
        ctx.fillStyle = `rgba(0,255,159,${opacity*0.3})`; ctx.fill();
        ctx.beginPath(); ctx.arc(this.x, this.y, this.radius * 0.6, 0, Math.PI*2);
        ctx.fillStyle = `rgba(200,255,230,${opacity*0.7})`; ctx.fill();
      }
    }

    function init() { particles = []; for (let i = 0; i < PARTICLE_COUNT; i++) particles.push(new Particle()); }

    function animate() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      for (let i = 0; i < particles.length; i++) {
        particles[i].update(); particles[i].draw();
        for (let j = i + 1; j < particles.length; j++) {
          const dx = particles[i].x-particles[j].x, dy = particles[i].y-particles[j].y, d = Math.sqrt(dx*dx+dy*dy);
          if (d < CONNECTION_DISTANCE) { ctx.beginPath(); ctx.moveTo(particles[i].x,particles[i].y); ctx.lineTo(particles[j].x,particles[j].y); ctx.strokeStyle=`rgba(0,255,159,${0.1*(1-d/CONNECTION_DISTANCE)})`; ctx.lineWidth=0.5; ctx.stroke(); }
        }
        if (mouse.x != null && mouse.y != null) {
          const mdx = particles[i].x - mouse.x, mdy = particles[i].y - mouse.y, md = Math.sqrt(mdx*mdx+mdy*mdy);
          if (md < mouse.radius) { ctx.beginPath(); ctx.moveTo(particles[i].x,particles[i].y); ctx.lineTo(mouse.x,mouse.y); ctx.strokeStyle=`rgba(0,255,159,${0.25*(1-md/mouse.radius)})`; ctx.lineWidth=0.8; ctx.stroke(); }
        }
      }
      animId = requestAnimationFrame(animate);
    }

    resize(); init(); animate();
    window.addEventListener("resize", () => { resize(); init(); });
    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseout", onMouseOut);
    return () => { cancelAnimationFrame(animId); window.removeEventListener("mousemove", onMouseMove); window.removeEventListener("mouseout", onMouseOut); };
  }, []);
  return <canvas ref={ref} className="fixed inset-0 z-0 pointer-events-none" />;
}

/* ════════════════════════════════════════════════════════
   SIDEBAR — Permanently expanded with icons + labels
   Upload removed — only via drop zone or chat attach icon
   ════════════════════════════════════════════════════════ */
function Sidebar({ activeMode, onModeChange }) {
  const iconMap = { local:"local", global:"global", writer:"writer", reviewer:"reviewer", comparator:"comparator" };
  return (
    <aside className="fixed left-0 top-0 bottom-0 z-50 flex w-[180px] flex-col border-r border-[var(--border)] py-5" style={{background:"rgba(5,7,9,0.85)",backdropFilter:"blur(20px)"}}>
      {/* Logo — exact match: landing .nav-logo-icon */}
      <div className="flex items-center gap-2.5 px-5 mb-5">
        <div className="flex h-9 w-9 items-center justify-center rounded-[10px] text-[12px] font-extrabold text-[#021a0f] select-none shrink-0"
          style={{background:"linear-gradient(135deg, var(--green), #34d399)", boxShadow:"0 0 20px var(--green-dim)"}}>
          RA
        </div>
        <span className="text-sm font-bold text-[var(--text)]" style={{letterSpacing:"-0.02em"}}>Research Agent</span>
      </div>
      <div className="mx-4 h-px bg-[var(--border)] mb-3" />

      {/* Mode buttons — always show icon + label */}
      <div className="flex flex-col gap-0.5 px-3">
        {MODES.map(mode => {
          const active = activeMode === mode.id;
          return (
            <button key={mode.id} type="button" onClick={() => onModeChange(mode.id)}
              className={`group relative flex items-center gap-3 rounded-xl px-3 py-2.5 transition-all duration-200 w-full text-left ${active ? "text-[var(--green)]" : "text-[var(--text-dim)] hover:text-[var(--text-muted)] hover:bg-[rgba(255,255,255,0.03)]"}`}
              style={active ? {background:"rgba(0,255,159,0.08)", boxShadow:"0 0 20px rgba(0,255,159,0.1)", transition:"all 0.25s cubic-bezier(0.16,1,0.3,1)"} : {transition:"all 0.25s cubic-bezier(0.16,1,0.3,1)"}}>
              <Icon name={iconMap[mode.id]} className="w-[18px] h-[18px] shrink-0" />
              <span className={`text-[13px] font-medium truncate ${active ? "text-[var(--green)]" : ""}`}>{mode.name}</span>
              {active && <span className="absolute right-0 top-1/2 -translate-y-1/2 h-5 w-[3px] rounded-full bg-[var(--green)]" style={{boxShadow:"0 0 8px var(--green)"}} />}
            </button>
          );
        })}
      </div>
      <div className="mt-auto mx-4 h-px bg-[var(--border)] mb-3" />
      <div className="px-5 text-[9px] text-[var(--text-dim)] font-bold tracking-widest opacity-40">LANGRAPH · RAG</div>
    </aside>
  );
}

/* ════════════════════════════════════════════════════════
   HEADER — Minimal, matches landing .nav style
   ════════════════════════════════════════════════════════ */
function Header({ title, papersCount, health, onTogglePanel, panelOpen }) {
  return (
    <header className="flex items-center justify-between gap-4 px-1 py-2.5 shrink-0">
      <h1 className="text-lg font-bold tracking-tight text-[var(--text)]" style={{letterSpacing:"-0.02em"}}>{title}</h1>
      <div className="flex items-center gap-2">
        {/* Badges — exact match: landing .badge style */}
        <Badge active><PulseDot active /> {papersCount} Indexed</Badge>
        <Badge active={health.graph_ready}><PulseDot active={health.graph_ready} /> {health.graph_ready?"System Ready":"Loading"}</Badge>
        <Badge active={health.llm_available}><PulseDot active={health.llm_available} warn={!health.llm_available} /> {health.llm_available?"Models Online":"Add Keys"}</Badge>
        <button type="button" onClick={onTogglePanel} title={panelOpen?"Hide papers":"Show papers"}
          className={`flex h-8 w-8 items-center justify-center rounded-[10px] border transition-all duration-200 ${panelOpen ? "border-[rgba(0,255,159,0.2)] bg-[rgba(0,255,159,0.08)] text-[var(--green)]" : "border-[var(--border-2)] bg-[rgba(255,255,255,0.02)] text-[var(--text-dim)] hover:text-[var(--text-muted)] hover:border-[var(--border-2)]"}`}
          style={{transition:"all 0.25s cubic-bezier(0.16,1,0.3,1)"}}>
          <Icon name="panels" className="w-4 h-4" />
        </button>
      </div>
    </header>
  );
}
function Badge({ children, active }) {
  return <span className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[10px] font-bold tracking-wider uppercase ${active ? "border-[rgba(0,255,159,0.2)] bg-[rgba(0,255,159,0.08)] text-[var(--text-muted)]" : "border-[var(--border)] bg-[rgba(255,255,255,0.02)] text-[var(--text-dim)]"}`}>{children}</span>;
}
function PulseDot({ active, warn }) {
  return <span className={`h-[6px] w-[6px] rounded-full ${active ? "bg-[var(--green)]" : warn ? "bg-red-400" : "bg-amber-400"}`} style={active ? {boxShadow:"0 0 8px var(--green)", animation:"pulse-dot 2s ease-in-out infinite"} : {}} />;
}

/* ════════════════════════════════════════════════════════
   FILE PANEL — Right panel, homepage card aesthetics
   ════════════════════════════════════════════════════════ */
function FilePanel({ papers, selectedIds, activeMode, fileStates, onSelect, onUploadClick, onDragOver, onDragLeave, onDrop, dragging, onDelete }) {
  const isRev = activeMode === "reviewer";
  const isComp = activeMode === "comparator";
  const isLocal = activeMode === "local";
  const isGlobal = activeMode === "global";
  const selectable = activeMode !== "writer";
  return (
    <div className="flex flex-col h-full border-l border-[var(--border)]" style={{background:"rgba(5,7,9,0.75)",backdropFilter:"blur(16px)"}}>
      <div className="flex items-center justify-between px-4 py-3 border-b border-[var(--border)] shrink-0">
        <div className="flex items-center gap-2">
          <Icon name="file" className="w-4 h-4 text-[var(--text-dim)]" />
          <span className="text-sm font-bold text-[var(--text)]">Papers</span>
          <span className="rounded-full bg-[rgba(0,255,159,0.08)] border border-[rgba(0,255,159,0.15)] px-2 py-0.5 text-[10px] font-bold text-[var(--green)]">{papers.length}</span>
        </div>
      </div>

      {/* Drop zone — matches landing pipeline-step hover */}
      <div className={`mx-3 mt-3 rounded-2xl border-2 border-dashed transition-all duration-300 cursor-pointer ${dragging ? "border-[rgba(0,255,159,0.4)] bg-[rgba(0,255,159,0.04)]" : "border-[var(--border)] hover:border-[var(--border-2)] hover:bg-[rgba(255,255,255,0.02)]"}`}
        onClick={onUploadClick} onDragOver={onDragOver} onDragLeave={onDragLeave} onDrop={onDrop}
        style={{transition:"all 0.3s cubic-bezier(0.16,1,0.3,1)"}}>
        <div className="flex items-center justify-center py-4 gap-2">
          <Icon name="upload" className={`w-4 h-4 ${dragging?"text-[var(--green)]":"text-[var(--text-dim)]"}`} />
          <span className={`text-xs font-medium ${dragging?"text-[var(--green)]":"text-[var(--text-dim)]"}`}>Drop PDFs here</span>
        </div>
      </div>

      {/* Selection hint — per-mode */}
      {selectable && (
        <div className="mx-3 mt-2 rounded-xl bg-[rgba(0,255,159,0.06)] border border-[rgba(0,255,159,0.15)] px-3 py-2">
          <span className="text-[10px] text-[var(--green)] font-bold uppercase tracking-wider">
            {(isLocal||isRev) ? `Select 1 paper · ${selectedIds.length===1?"1 selected ✓":"None selected"}`
             : isGlobal ? `Select papers for context · ${selectedIds.length} selected`
             : `Select 2–3 · ${selectedIds.length}/3 selected`}
          </span>
        </div>
      )}

      {/* Paper cards — styled like landing cards */}
      <div className="flex-1 overflow-y-auto px-3 py-3 space-y-2">
        {papers.map(paper => {
          const sel = selectedIds.includes(paper.paper_id);
          const fst = fileStates[paper.paper_id] || { status: FILE_ST.INDEXED, progress:100, chunks: paper.chunk_count||0 };
          const indexed = Number(paper.chunk_count||0) > 0;
          const isIndexing = fst.status === FILE_ST.INDEXING;
          const isUploading = fst.status === FILE_ST.UPLOADING;
          const isError = fst.status === FILE_ST.ERROR;
          return (
            <div key={paper.paper_id}
              className={`group rounded-2xl border p-3 cursor-pointer transition-all duration-300 ${
                sel && selectable
                  ? "border-[rgba(0,255,159,0.25)] bg-[var(--surface)]"
                  : isError
                  ? "border-red-400/30 bg-[rgba(248,113,113,0.04)]"
                  : "border-[var(--border)] bg-[var(--surface)] hover:border-[var(--border-2)] hover:bg-[var(--surface-2)]"
              }`}
              style={sel && selectable ? {boxShadow:"0 8px 40px rgba(0,255,159,0.06)"} : {transition:"all 0.3s cubic-bezier(0.16,1,0.3,1)"}}
              onClick={() => selectable && onSelect(paper.paper_id)}>
              <div className="flex items-start gap-2.5">
                <div className="pt-0.5">
                  {selectable ? (
                    <div className={`flex h-5 w-5 items-center justify-center rounded-md border-2 transition-all ${sel ? "border-[var(--green)] bg-[rgba(0,255,159,0.15)] text-[var(--green)]" : "border-[var(--text-dim)] bg-[rgba(255,255,255,0.03)]"}`}
                      style={sel ? {boxShadow:"0 0 8px rgba(0,255,159,0.3)"} : {}}>
                      {sel && <Icon name="check" className="w-3 h-3" />}
                    </div>
                  ) : (
                    <div className={`h-5 w-5 rounded-md flex items-center justify-center ${isUploading ? "bg-[rgba(255,255,255,0.04)]" : isIndexing ? "bg-[rgba(0,255,159,0.06)]" : "bg-[rgba(0,255,159,0.06)]"}`}>
                      {isUploading ? <span className="h-2.5 w-2.5 rounded-full border-2 border-[var(--text-dim)] border-t-transparent animate-spin" />
                       : isIndexing ? <span className="h-2.5 w-2.5 rounded-full border-2 border-[var(--green)] border-t-transparent animate-spin" />
                       : <Icon name="file" className="w-3 h-3 text-[var(--green)] opacity-50" />}
                    </div>
                  )}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-start justify-between gap-2">
                    <h4 className="text-[13px] font-medium text-[var(--text)] truncate flex-1">{paper.filename}</h4>
                    <button type="button" onClick={(e) => { e.stopPropagation(); onDelete(paper.paper_id); }} 
                      className="text-[var(--text-dim)] hover:text-red-400 opacity-0 group-hover:opacity-100 transition-all p-1 -mt-1 -mr-1" title="Remove paper">
                      <Icon name="x" className="w-[14px] h-[14px]" />
                    </button>
                  </div>
                  <div className="flex items-center gap-2 mt-1 flex-wrap">
                    {isUploading ? <StatusChip color="gray">Uploading</StatusChip>
                     : isIndexing ? <StatusChip color="green" pulse>Indexing</StatusChip>
                     : isError ? <StatusChip color="red">Error</StatusChip>
                     : <StatusChip color="green">Indexed</StatusChip>}
                    {isUploading && <span className="text-[10px] text-[var(--text-dim)] font-mono tabular-nums">{Math.round(fst.progress||0)}%</span>}
                    {isIndexing ? <span className="text-[10px] text-[var(--green)] font-mono tabular-nums" style={{animation:"shimmer 1.5s ease-in-out infinite"}}>{fst.chunks||0} chunks</span>
                     : indexed ? <span className="text-[10px] text-[var(--text-dim)]">{paper.chunk_count} chunks</span>
                     : null}
                  </div>
                  {/* Progress bar */}
                  {(isUploading||isIndexing) && (
                    <div className="mt-2 h-1.5 rounded-full bg-[rgba(255,255,255,0.04)] overflow-hidden">
                      <div className={`h-full rounded-full transition-all duration-700 ease-out ${isUploading ? "bg-[var(--text-dim)]" : "bg-[var(--green)]"}`}
                        style={{width:`${fst.progress}%`, ...(isIndexing?{boxShadow:"0 0 8px rgba(0,255,159,0.3)"}:{})}} />
                    </div>
                  )}
                </div>
              </div>
            </div>
          );
        })}
        {!papers.length && (
          <div className="flex flex-col items-center justify-center py-12 text-center">
            <div className="h-12 w-12 rounded-2xl bg-[rgba(0,255,159,0.06)] border border-[rgba(0,255,159,0.12)] flex items-center justify-center mb-3">
              <Icon name="file" className="w-6 h-6 text-[var(--text-dim)]" />
            </div>
            <p className="text-sm font-medium text-[var(--text-dim)]">No papers yet</p>
            <p className="text-xs text-[var(--text-dim)] opacity-60 mt-1">Upload PDFs to start</p>
          </div>
        )}
      </div>
    </div>
  );
}
function StatusChip({ children, color, pulse }) {
  const colors = {
    green:"bg-[rgba(0,255,159,0.08)] border-[rgba(0,255,159,0.2)] text-[var(--green)]",
    red:"bg-[rgba(248,113,113,0.08)] border-red-400/20 text-red-400",
    gray:"bg-[rgba(255,255,255,0.04)] border-[var(--border)] text-[var(--text-dim)]",
  };
  return (
    <span className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[9px] font-bold uppercase tracking-wider ${colors[color]||colors.gray}`}>
      {pulse ? <span className="h-1 w-1 rounded-full bg-[var(--green)]" style={{boxShadow:"0 0 4px var(--green)",animation:"pulse-dot 2s ease-in-out infinite"}} /> : <span className="h-1 w-1 rounded-full" style={color==="green"?{background:"var(--green)",boxShadow:"0 0 4px var(--green)"}:color==="red"?{background:"#f87171"}:{background:"var(--text-dim)"}} />}
      {children}
    </span>
  );
}

/* ════════════════════════════════════════════════════════
   CONVERSATION — Central workspace
   Task-based thinking: each task owns its mode permanently
   ════════════════════════════════════════════════════════ */
function ConversationStream({ history, activeTasks, activeMode, currentMode, copiedId, onCopy, selectedPapers, papers, onDeselect }) {
  const visible = history.filter(i=>i.role==="user"||i.role==="assistant");
  const endRef = useRef(null);
  const loading = Object.keys(activeTasks||{}).length > 0;
  useEffect(() => { endRef.current?.scrollIntoView({behavior:"smooth"}); }, [visible.length, loading]);
  const isComp = activeMode === "comparator";
  const isRev = activeMode === "reviewer";
  const isLocal = activeMode === "local";
  const isGlobal = activeMode === "global";
  const isChatMode = !isRev && !isComp;

  return (
    <div className="flex-1 overflow-y-auto relative z-10 px-6">
      <div className="max-w-[760px] mx-auto py-6 space-y-5">

        {/* ── Active Paper Context Bar ── */}
        {isChatMode && selectedPapers && selectedPapers.length > 0 && (
          <div className="flex items-center gap-2 flex-wrap anim-fade" style={{marginBottom:"-4px"}}>
            {isLocal ? (
              /* Local Brain: single paper chip */
              <div className="inline-flex items-center gap-2 rounded-xl border border-[rgba(0,255,159,0.2)] bg-[rgba(0,255,159,0.04)] px-3 py-1.5 transition-all" style={{boxShadow:"0 0 12px rgba(0,255,159,0.04)"}}>
                <span className="h-1.5 w-1.5 rounded-full bg-[var(--green)]" style={{boxShadow:"0 0 4px var(--green)"}}/>
                <span className="text-[11px] font-medium text-[var(--green)] opacity-80">Active Paper:</span>
                <span className="text-[11px] font-medium text-[var(--text)] max-w-[200px] truncate">{selectedPapers[0].filename}</span>
                <button onClick={()=>onDeselect(selectedPapers[0].paper_id)} className="text-[var(--text-dim)] hover:text-red-400 transition-colors ml-0.5"><Icon name="x" className="w-3 h-3"/></button>
              </div>
            ) : (
              /* Global Brain / Writer: multi-paper summary */
              <div className="inline-flex items-center gap-2 rounded-xl border border-[rgba(0,255,159,0.2)] bg-[rgba(0,255,159,0.04)] px-3 py-1.5 group relative" style={{boxShadow:"0 0 12px rgba(0,255,159,0.04)"}}>
                <span className="h-1.5 w-1.5 rounded-full bg-[var(--green)]" style={{boxShadow:"0 0 4px var(--green)"}}/>
                <span className="text-[11px] font-medium text-[var(--green)]">{selectedPapers.length} Paper{selectedPapers.length!==1?"s":""} Active</span>
                {/* Expand on hover */}
                <div className="hidden group-hover:flex absolute top-full left-0 mt-1 flex-col gap-1 rounded-xl border border-[var(--border-2)] p-2 min-w-[220px] z-50" style={{background:"var(--surface)",backdropFilter:"blur(16px)",boxShadow:"0 8px 32px rgba(0,0,0,0.4)"}}>
                  {selectedPapers.map(p => (
                    <div key={p.paper_id} className="flex items-center gap-2 rounded-lg px-2 py-1.5 hover:bg-[rgba(255,255,255,0.03)]">
                      <Icon name="file" className="w-3 h-3 text-[var(--green)] opacity-50 shrink-0"/>
                      <span className="text-[11px] text-[var(--text)] flex-1 truncate">{p.filename}</span>
                      <button onClick={()=>onDeselect(p.paper_id)} className="text-[var(--text-dim)] hover:text-red-400 transition-colors"><Icon name="x" className="w-3 h-3"/></button>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
        {/* Empty state */}
        {!visible.length && !loading && (
          <div className="flex flex-col items-center justify-center min-h-[60vh] text-center anim-fade">
            <div className="mb-6">
              <div className="h-20 w-20 rounded-2xl flex items-center justify-center border border-[rgba(0,255,159,0.15)]"
                style={{background:"rgba(0,255,159,0.06)", boxShadow:"0 0 40px rgba(0,255,159,0.06)"}}>
                <Icon name={activeMode} className="w-9 h-9 text-[var(--green)] opacity-40" />
              </div>
            </div>
            <h2 className="text-2xl font-bold text-[var(--text)] mb-2" style={{letterSpacing:"-0.03em"}}>{currentMode.name}</h2>
            <p className="text-sm text-[var(--text-muted)] max-w-sm leading-relaxed mb-5">{currentMode.desc}</p>
            {isRev && <p className="text-xs text-[var(--text-dim)]">Select a paper from the panel, then click <strong className="text-[var(--green)]">Run Review</strong></p>}
            {isComp && <p className="text-xs text-[var(--text-dim)]">Select 2–3 papers, then click <strong className="text-[var(--green)]">Compare Papers</strong></p>}
            {!isRev && !isComp && <p className="text-xs text-[var(--text-dim)]">Upload research papers and ask questions below</p>}
          </div>
        )}

        {/* Messages */}
        {visible.map((item, idx) => {
          const isA = item.role === "assistant";
          const showComp = isA && item.mode === "comparator";
          const showRev = isA && item.mode === "reviewer";
          return (
            <div key={item.id} className="anim-fade-up" style={{animationDelay:`${Math.min(idx*30,150)}ms`}}>
              {/* User */}
              {!isA && (
                <div className="flex justify-end mb-1">
                  <div className="max-w-[70%] rounded-2xl rounded-br-md border border-[var(--border-2)] px-5 py-3" style={{background:"var(--surface-2)"}}>
                    <p className="text-sm text-[var(--text)] leading-relaxed whitespace-pre-wrap">{item.content}</p>
                  </div>
                </div>
              )}

              {/* Assistant */}
              {isA && (
                <div className="flex justify-start mb-1">
                  <div className="max-w-[92%] w-full">
                    {/* Mode indicator */}
                    <div className="flex items-center gap-2 mb-1.5">
                      <div className="h-5 w-5 rounded-lg flex items-center justify-center" style={{background:"rgba(0,255,159,0.08)"}}>
                        <Icon name={item.mode||"global"} className="w-2.5 h-2.5 text-[var(--green)] opacity-60" />
                      </div>
                      <span className="text-[9px] font-bold uppercase tracking-[.14em] text-[var(--text-dim)]">{modeOf(item.mode).name}</span>
                      {showComp && (
                        <button onClick={()=>onCopy(item.id,item.content)} className="ml-auto flex items-center gap-1 text-[9px] text-[var(--text-dim)] hover:text-[var(--green)] transition-colors">
                          <Icon name="copy" className="w-3 h-3"/>{copiedId===item.id?"Copied ✓":"Copy"}
                        </button>
                      )}
                    </div>

                    {/* Content bubble — matches landing card style */}
                    <div className="rounded-2xl rounded-tl-md border border-[var(--border)] px-5 py-4" style={{background:"rgba(255,255,255,0.015)"}}>
                      <div className="text-sm leading-relaxed">
                        {showComp ? renderCompAns(item.content) : renderMd(item.content)}
                      </div>
                      {/* Citations */}
                      {item.citations?.length > 0 && (
                        <div className="mt-4 pt-3 border-t border-[var(--border)]">
                          <div className="flex items-center gap-2 mb-2">
                            <span className="text-[10px] font-bold uppercase tracking-[.14em] text-[var(--green)] opacity-50">Sources</span>
                            <span className="rounded-full bg-[rgba(0,255,159,0.08)] border border-[rgba(0,255,159,0.15)] px-2 py-0.5 text-[10px] font-bold text-[var(--green)]">{item.citations.length}</span>
                          </div>
                          <div className="space-y-1.5">
                            {item.citations.slice(0,3).map((c,ci) => (
                              <div key={ci} className="rounded-xl border border-[var(--border)] bg-[rgba(0,255,159,0.01)] px-3 py-2">
                                <span className="text-[11px] font-medium text-[var(--green)] opacity-50">{c.filename||"Source"}</span>
                                {c.snippet && <p className="text-[11px] text-[var(--text-dim)] mt-0.5 line-clamp-2">{c.snippet}</p>}
                              </div>
                            ))}
                            {item.citations.length > 3 && <p className="text-[10px] text-[var(--text-dim)] px-1">+{item.citations.length-3} more</p>}
                          </div>
                        </div>
                      )}
                    </div>
                    {/* Reviewer extras */}
                    {showRev && item.debug?.round_events && <div className="mt-3">{renderDebate(item.debug)}</div>}
                    {showRev && item.debug?.final_report && <div className="mt-3">{renderReport(item.debug.final_report)}</div>}
                  </div>
                </div>
              )}
            </div>
          );
        })}

        {/* Thinking — task-based, mode-locked: each task shows its originating mode */}
        {Object.entries(activeTasks||{}).map(([taskId, task]) => (
          <div key={taskId} className="flex justify-start anim-fade-up">
            <div className="flex items-center gap-3 rounded-2xl border border-[var(--border)] bg-[rgba(255,255,255,0.015)] px-5 py-3.5">
              <div className="h-5 w-5 rounded-lg flex items-center justify-center shrink-0" style={{background:"rgba(0,255,159,0.08)"}}>
                <Icon name={task.mode||"global"} className="w-2.5 h-2.5 text-[var(--green)] opacity-60" />
              </div>
              <div className="flex items-center gap-1">
                <span className="h-1.5 w-1.5 rounded-full bg-[var(--green)] animate-bounce" style={{animationDelay:"0ms",opacity:0.6}}/>
                <span className="h-1.5 w-1.5 rounded-full bg-[var(--green)] animate-bounce" style={{animationDelay:"150ms",opacity:0.4}}/>
                <span className="h-1.5 w-1.5 rounded-full bg-[var(--green)] animate-bounce" style={{animationDelay:"300ms",opacity:0.2}}/>
              </div>
              <span className="text-xs text-[var(--text-dim)]">{modeOf(task.mode).name} is thinking…</span>
            </div>
          </div>
        ))}
        <div ref={endRef} />
      </div>
    </div>
  );
}

/* ════════════════════════════════════════════════════════
   BOTTOM BAR — Matches landing .hero-search style
   Reviewer/Comparator = action panel (no text input)
   Chat modes = text input styled like landing search
   ════════════════════════════════════════════════════════ */
function BottomBar({ draft, onChange, onKeyDown, onSend, onAttach, loading, canSend, currentMode, activeMode, onCompare, onReview, canCompare, canReview, selectedPapers, reviewTargetPaper, comparatorPreset, onCompPresetChange, contextCount }) {
  const taRef = useRef(null);
  useEffect(()=>{if(taRef.current){taRef.current.style.height="auto";taRef.current.style.height=Math.min(taRef.current.scrollHeight,120)+"px";}}, [draft]);

  const isComp = activeMode === "comparator";
  const isRev = activeMode === "reviewer";

  /* ─── REVIEWER panel ─── */
  if (isRev) {
    return (
      <div className="shrink-0 px-4 pb-3 pt-1">
        <div className="max-w-[760px] mx-auto">
          <div className="rounded-[14px] border border-[var(--border-2)] p-3" style={{background:"var(--surface)",transition:"all 0.3s"}}>
            <div className="flex items-center gap-3 mb-2">
              <div className="h-8 w-8 rounded-[10px] flex items-center justify-center" style={{background:"rgba(0,255,159,0.08)",border:"1px solid rgba(0,255,159,0.15)"}}>
                <Icon name="reviewer" className="w-4 h-4 text-[var(--green)] opacity-60" />
              </div>
              <div className="flex-1 min-w-0">
                <div className="text-[10px] font-bold uppercase tracking-[.14em] text-[var(--green)] opacity-40">Review Target</div>
                <div className="text-sm font-medium text-[var(--text)] truncate">{reviewTargetPaper ? reviewTargetPaper.filename : "No paper selected"}</div>
              </div>
              {reviewTargetPaper && <span className="rounded-full bg-[rgba(0,255,159,0.08)] border border-[rgba(0,255,159,0.2)] px-2 py-0.5 text-[9px] font-bold text-[var(--green)] uppercase">Ready</span>}
            </div>

            {/* CTA — matches landing .btn-primary exactly */}
            <button onClick={onReview} disabled={!canReview||loading}
              className="w-full flex items-center justify-center gap-2 rounded-[12px] py-2.5 text-sm font-bold text-[#021a0f] disabled:opacity-20 disabled:cursor-not-allowed transition-all"
              style={{background:"linear-gradient(135deg, var(--green), #34d399)", boxShadow: canReview&&!loading ? "0 8px 32px rgba(0,255,159,0.2)" : "none", transition:"all 0.25s cubic-bezier(0.16,1,0.3,1)"}}>
              {loading ? <><span className="h-4 w-4 rounded-full border-2 border-[#021a0f]/30 border-t-[#021a0f] animate-spin"/> Analyzing…</> : <><Icon name="play" className="w-4 h-4"/> Run Full Review</>}
            </button>
          </div>
        </div>
      </div>
    );
  }

  /* ─── COMPARATOR panel ─── */
  if (isComp) {
    const count = selectedPapers.length;
    const valid = count >= 2 && count <= 3;
    return (
      <div className="shrink-0 px-4 pb-3 pt-1">
        <div className="max-w-[760px] mx-auto">
          <div className="rounded-[14px] border border-[var(--border-2)] p-3" style={{background:"var(--surface)"}}>
            <div className="flex items-center justify-between gap-3 mb-2">
              <div className="flex items-center gap-2.5">
                <div className="h-8 w-8 rounded-[10px] flex items-center justify-center" style={{background:"rgba(0,255,159,0.08)",border:"1px solid rgba(0,255,159,0.15)"}}>
                  <Icon name="comparator" className="w-4 h-4 text-[var(--green)] opacity-60" />
                </div>
                <div>
                  <div className="text-[10px] font-bold uppercase tracking-[.14em] text-[var(--green)] opacity-40">Compare</div>
                  <div className="text-sm font-medium text-[var(--text)]">{count} paper{count!==1?"s":""} selected</div>
                </div>
              </div>
              <span className={`rounded-full px-2 py-0.5 text-[9px] font-bold uppercase border ${valid?"bg-[rgba(0,255,159,0.08)] border-[rgba(0,255,159,0.2)] text-[var(--green)]":"bg-[rgba(255,255,255,0.02)] border-[var(--border)] text-[var(--text-dim)]"}`}>{valid?"Ready":count<2?"Need 2+":"Max 3"}</span>
            </div>
            {count > 0 && (
              <div className="flex flex-wrap gap-1.5 mb-2">
                {selectedPapers.map(p => <span key={p.paper_id} className="inline-flex items-center gap-1 rounded-xl bg-[rgba(0,255,159,0.04)] border border-[rgba(0,255,159,0.12)] px-2 py-0.5 text-[10px] text-[var(--green)] font-medium truncate max-w-[180px]"><Icon name="file" className="w-3 h-3 shrink-0"/>{p.filename}</span>)}
              </div>
            )}
            <div className="flex items-center gap-2 mb-2">
              <span className="text-[10px] text-[var(--text-dim)] font-medium">Mode:</span>
              {COMPARE_PRESETS.map(p => (
                <button key={p.id} onClick={()=>onCompPresetChange(p.id)}
                  className={`rounded-[10px] px-2.5 py-1 text-[11px] font-semibold border transition-all ${comparatorPreset===p.id ? "bg-[rgba(0,255,159,0.1)] text-[var(--green)] border-[rgba(0,255,159,0.2)]" : "text-[var(--text-dim)] border-[var(--border)] hover:border-[var(--border-2)]"}`}
                  style={{transition:"all 0.25s cubic-bezier(0.16,1,0.3,1)"}}>{p.label}</button>
              ))}
            </div>
            <button onClick={onCompare} disabled={!valid||loading}
              className="w-full flex items-center justify-center gap-2 rounded-[12px] py-2.5 text-sm font-bold text-[#021a0f] disabled:opacity-20 disabled:cursor-not-allowed transition-all"
              style={{background:"linear-gradient(135deg, var(--green), #34d399)", boxShadow: valid&&!loading ? "0 8px 32px rgba(0,255,159,0.2)" : "none", transition:"all 0.25s cubic-bezier(0.16,1,0.3,1)"}}>
              {loading ? <><span className="h-4 w-4 rounded-full border-2 border-[#021a0f]/30 border-t-[#021a0f] animate-spin"/> Comparing…</> : <><Icon name="zap" className="w-4 h-4"/> Compare Papers</>}
            </button>
          </div>
        </div>
      </div>
    );
  }

  /* ─── CHAT INPUT — matches landing .hero-search ─── */
  return (
    <div className="shrink-0 px-4 pb-3 pt-1">
      <div className="max-w-[760px] mx-auto">
        {/* Paper context indicator */}
        {contextCount > 0 && (
          <div className="flex items-center gap-2 mb-1.5 px-1">
            <span className="h-1.5 w-1.5 rounded-full bg-[var(--green)]" style={{boxShadow:"0 0 4px var(--green)"}} />
            <span className="text-[10px] font-medium text-[var(--green)] opacity-60">Using: {contextCount} paper{contextCount!==1?"s":""}</span>
          </div>
        )}
        <div className="flex items-end gap-3 rounded-[14px] border border-[var(--border-2)] px-4 py-3 transition-all focus-within:border-[rgba(0,255,159,0.3)]"
          style={{background:"var(--surface)", transition:"border-color 0.3s"}}>
          <button type="button" onClick={onAttach} className="flex h-8 w-8 items-center justify-center rounded-xl text-[var(--text-dim)] hover:text-[var(--green)] hover:bg-[rgba(0,255,159,0.06)] transition-all shrink-0 mb-0.5">
            <Icon name="attach" className="w-[18px] h-[18px]" />
          </button>
          <textarea ref={taRef} value={draft} onChange={e=>onChange(e.target.value)} onKeyDown={onKeyDown}
            rows={1} placeholder={`Ask ${currentMode.name} anything…`}
            className="flex-1 min-h-[28px] max-h-[120px] resize-none bg-transparent text-[15px] text-[var(--text)] outline-none placeholder:text-[var(--text-dim)] leading-relaxed" style={{fontFamily:"inherit"}} />
          <button type="button" onClick={onSend} disabled={!canSend||loading}
            className="flex h-9 w-9 items-center justify-center rounded-[10px] text-[#021a0f] disabled:opacity-15 disabled:cursor-not-allowed shrink-0 transition-all"
            style={{background:"linear-gradient(135deg, var(--green), #34d399)", boxShadow: canSend&&!loading ? "0 0 20px var(--green-dim)" : "none", transition:"all 0.25s cubic-bezier(0.16,1,0.3,1)"}}>
            {loading ? <span className="h-3 w-3 rounded-full border-2 border-[#021a0f]/30 border-t-[#021a0f] animate-spin"/> : <Icon name="send" className="w-4 h-4" />}
          </button>
        </div>
        <p className="text-center text-[10px] text-[var(--text-dim)] mt-1.5 opacity-40">Research Agent · LangGraph-powered · {currentMode.name}</p>
      </div>
    </div>
  );
}

/* ════════════════════════════════════════════════════════
   MAIN APP
   ════════════════════════════════════════════════════════ */
function ResearchAgent() {
  const [sessionId] = useState(() => sid());
  const [activeMode, setActiveMode] = useState("global");
  const [draft, setDraft] = useState("");
  const [papers, setPapers] = useState([]);
  /* Per-mode paper selection — persists across mode switches */
  const [modeSelections, setModeSelections] = useState({ local:[], global:[], writer:[], reviewer:[], comparator:[] });

  const [compPreset, setCompPreset] = useState("full");
  const [history, setHistory] = useState([]);
  const [health, setHealth] = useState({ llm_available:false, graph_ready:false, indexed_papers:0 });
  /* Task-based loading — each request permanently owns its originating mode */
  const [activeTasks, setActiveTasks] = useState({});
  const [error, setError] = useState("");
  const [panelOpen, setPanelOpen] = useState(true);
  const [dragging, setDragging] = useState(false);
  const [copiedId, setCopiedId] = useState("");
  const [fileStates, setFileStates] = useState({});
  const fileRef = useRef(null);
  const didReset = useRef(false);

  /* Derived — selectedIds is the CURRENT mode's selections */
  const currentMode = modeOf(activeMode);
  const isRev = activeMode === "reviewer";
  const isComp = activeMode === "comparator";
  const selectedIds = modeSelections[activeMode] || [];
  const loading = Object.keys(activeTasks).length > 0;
  
  const allPapers = useMemo(() => {
    const list = [...papers];
    // Only show temp entries if no real paper with the same filename exists yet
    const realFilenames = new Set(papers.map(p => p.filename));
    Object.keys(fileStates).forEach(id => {
      if (id.startsWith("temp-") && !list.some(x=>x.paper_id===id)) {
        const tempFilename = fileStates[id].filename;
        if (!realFilenames.has(tempFilename)) {
          list.unshift({ paper_id: id, filename: tempFilename, chunk_count: 0 });
        }
      }
    });
    return list;
  }, [papers, fileStates]);

  const selectedPapers = useMemo(() => papers.filter(p => selectedIds.includes(p.paper_id)), [papers, selectedIds]);
  const reviewTarget = isRev && selectedIds.length === 1 ? papers.find(p => p.paper_id === selectedIds[0]) : null;
  const canReview = isRev && selectedIds.length === 1 && Number(reviewTarget?.chunk_count||0) > 0;
  const canCompare = isComp && selectedIds.length >= 2 && selectedIds.length <= 3;

  useEffect(() => { bootstrap({reset:true}); }, []);
  useEffect(() => { setError(""); }, [activeMode]);
  /* Prune stale paper IDs when papers list changes */
  useEffect(() => { setModeSelections(s=>{ const n={...s}; for(const m in n) n[m]=n[m].filter(id=>papers.some(p=>p.paper_id===id)); return n; }); }, [papers]);

  async function bootstrap({reset=false}={}) {
    try {
      if(reset && !didReset.current) { didReset.current=true; try{await api(`${API}/papers`,{method:"DELETE"});}catch(_){} }
      const [pd,,h] = await Promise.all([api(`${API}/papers`), api(`${API}/style-profile`).catch(()=>({})), api("/health")]);
      setPapers(pd.papers||[]); setHealth(h); setError("");
    } catch(e){ setError(e.message||"Load failed."); }
  }

  function changeMode(id) { if(id!==activeMode) setActiveMode(id); }

  function selectPaper(paperId) {
    const mode = activeMode;
    setModeSelections(prev => {
      const current = prev[mode] || [];
      if (current.includes(paperId)) return { ...prev, [mode]: current.filter(id=>id!==paperId) };
      if (mode === "local" || mode === "reviewer") return { ...prev, [mode]: [paperId] }; // single select
      if (mode === "comparator" && current.length >= 3) return prev; // max 3
      return { ...prev, [mode]: [...current, paperId] };
    });
  }

  function handleUploadClick() { fileRef.current?.click(); }
  function handleDragOver(e) { e.preventDefault(); setDragging(true); }
  function handleDragLeave() { setDragging(false); }

  async function deletePaper(paperId) {
    if (paperId.startsWith("temp-")) {
      setFileStates(s => { const n={...s}; delete n[paperId]; return n; });
      return;
    }
    // Optimistic removal (instantly removes from UI and clears active context via useEffect)
    setPapers(p => p.filter(x => x.paper_id !== paperId));
    setFileStates(s => { const n={...s}; delete n[paperId]; return n; });
    
    try {
      await api(`${API}/papers/${paperId}`, { method: "DELETE" });
      await bootstrap(); 
    } catch(err) {
      setError(err.message || "Failed to remove paper.");
      await bootstrap(); // Re-sync to restore if failed
    }
  }

  /* ═══ Upload + indexing pipeline ═══ */
  async function uploadFiles(files) {
    if(!files.length)return;
    setError("");
    const form = new FormData();
    for(const f of files) form.append("files", f);
    const tempIds = Array.from(files).map(f => `temp-${f.name}-${Date.now()}`);
    tempIds.forEach((id,i) => setFileStates(s=>({...s,[id]:{status:FILE_ST.UPLOADING,progress:0,chunks:0,filename:files[i].name}})));

    // Stage 1: Upload with smooth progress 0→100
    const prog = setInterval(()=>{ tempIds.forEach(id=>setFileStates(s=>{const fs=s[id];if(!fs||fs.status!==FILE_ST.UPLOADING)return s;const step=Math.random()*8+3;return{...s,[id]:{...fs,progress:Math.min(fs.progress+step,92)}};})); },300);

    try {
      await api(`${API}/papers/upload`, { method:"POST", body:form });
      clearInterval(prog);
      // Finish upload bar to 100%
      tempIds.forEach(id => setFileStates(s=>({...s,[id]:{...(s[id]||{}),status:FILE_ST.UPLOADING,progress:100}})));
      await new Promise(r=>setTimeout(r,400));
      // Stage 2: Indexing — progress 0→100, chunk count climbing
      tempIds.forEach(id => setFileStates(s=>({...s,[id]:{...(s[id]||{}),status:FILE_ST.INDEXING,progress:0,chunks:0}})));
      let ch=0;
      const idx = setInterval(()=>{ch+=Math.floor(Math.random()*6)+2;tempIds.forEach(id=>setFileStates(s=>{const fs=s[id];if(!fs||fs.status!==FILE_ST.INDEXING)return s;const np=Math.min(fs.progress+Math.random()*10+4,96);return{...s,[id]:{...fs,progress:np,chunks:ch}};}));},400);
      await new Promise(r=>setTimeout(r,2500));
      clearInterval(idx);
      // Stage 3: Indexed — snap to 100%
      tempIds.forEach(id=>setFileStates(s=>({...s,[id]:{...(s[id]||{}),status:FILE_ST.INDEXED,progress:100}})));
      tempIds.forEach(id=>setFileStates(s=>{const n={...s};delete n[id];return n;}));
      await bootstrap();
    } catch(err) { clearInterval(prog); setError(err.message||"Upload failed."); tempIds.forEach(id=>setFileStates(s=>{const n={...s};delete n[id];return n;})); }
    setDragging(false);
    if(fileRef.current) fileRef.current.value="";
  }

  async function onFileChange(e) { await uploadFiles(Array.from(e.target.files||[])); }
  async function handleDrop(e) { e.preventDefault(); setDragging(false); await uploadFiles(Array.from(e.dataTransfer?.files||[]).filter(f=>f.name.toLowerCase().endsWith(".pdf"))); }

  function compactHistory() { return history.filter(i=>i.role==="user"||i.role==="assistant").slice(-8).map(i=>({role:i.role,content:i.content})); }

  /* ═══ Chat ═══ */
  async function sendChat() {
    const msg = draft.trim(); if(!msg||loading) return;
    const taskId = `task-${Date.now()}`;
    const taskMode = activeMode; // capture mode at invocation time
    const pids = [...(modeSelections[taskMode] || [])];
    setDraft(""); setError("");
    setActiveTasks(t=>({...t,[taskId]:{mode:taskMode,status:"thinking"}}));
    setHistory(h=>[...h,{id:`${Date.now()}-u`,role:"user",mode:taskMode,content:msg,taskId,paperCount:pids.length}]);
    try {
      const resp = await api(`${API}/chat`,{method:"POST",headers:{"Content-Type":"application/json"},
        body:JSON.stringify({session_id:sessionId,mode:taskMode,message:msg,paper_ids:pids,review_paper_id:null,intervention_mode:null,history:compactHistory()})});
      setHistory(h=>[...h,{id:`${Date.now()}-a`,role:"assistant",mode:taskMode,content:resp.answer,citations:normCites(resp.citations||[]),debug:resp.debug||{},taskId}]);
    } catch(err){setError(err.message||"Chat failed.")} finally{setActiveTasks(t=>{const n={...t};delete n[taskId];return n;})}
  }

  async function runReview() {
    if(!canReview||loading)return;
    const taskId = `task-${Date.now()}`;
    const pid=selectedIds[0];
    setError("");
    setActiveTasks(t=>({...t,[taskId]:{mode:"reviewer",status:"thinking"}}));
    setHistory(h=>[...h,{id:`${Date.now()}-u`,role:"user",mode:"reviewer",content:`Review: Full Review`,taskId}]);
    try {
      const resp=await api(`${API}/chat`,{method:"POST",headers:{"Content-Type":"application/json"},
        body:JSON.stringify({session_id:sessionId,mode:"reviewer",message:`[Start Debate] Focus lens: Full Review`,paper_ids:[],review_paper_id:pid,intervention_mode:"ask",history:[]})});
      setHistory(h=>[...h,{id:`${Date.now()}-a`,role:"assistant",mode:"reviewer",content:resp.answer,citations:normCites(resp.citations||[]),debug:resp.debug||{},taskId}]);
    } catch(err){setError(err.message||"Review failed.")} finally{setActiveTasks(t=>{const n={...t};delete n[taskId];return n;})}
  }


  async function runCompare() {
    if(!canCompare||loading)return;
    const taskId = `task-${Date.now()}`;
    const pids = [...selectedIds];
    const msg=compTextById(compPreset);
    setError("");
    setActiveTasks(t=>({...t,[taskId]:{mode:"comparator",status:"thinking"}}));
    setHistory(h=>[...h,{id:`${Date.now()}-u`,role:"user",mode:"comparator",content:`Compare: ${compLabelById(compPreset)}`,taskId}]);
    try {
      const resp=await api(`${API}/chat`,{method:"POST",headers:{"Content-Type":"application/json"},
        body:JSON.stringify({session_id:sessionId,mode:"comparator",message:msg,paper_ids:pids,review_paper_id:null,intervention_mode:null,history:[]})});
      setHistory(h=>[...h,{id:`${Date.now()}-a`,role:"assistant",mode:"comparator",content:resp.answer,citations:normCites(resp.citations||[]),debug:resp.debug||{},taskId}]);
    } catch(err){setError(err.message||"Compare failed.")} finally{setActiveTasks(t=>{const n={...t};delete n[taskId];return n;})}
  }

  function onKeyDown(e){if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();sendChat();}}
  async function copyText(mid,c){const t=String(c||"").trim();if(!t)return;try{await navigator.clipboard.writeText(t);setCopiedId(mid);setTimeout(()=>setCopiedId(x=>x===mid?"":x),1400);}catch(_){}}

  /* ═══ RENDER ═══ */
  return (
    <div className="h-screen flex overflow-hidden relative" style={{fontFamily:"'Inter',-apple-system,sans-serif"}}
      onDragOver={e=>{e.preventDefault();setDragging(true);}} onDragLeave={()=>setDragging(false)} onDrop={handleDrop}>

      {/* Living background — same as landing page */}
      <AmbientCanvas />

      {/* Sidebar */}
      <Sidebar activeMode={activeMode} onModeChange={changeMode} />

      {/* Main area */}
      <div className="flex-1 flex flex-col ml-[180px] relative z-10">
        {/* Header */}
        <div className="px-6">
          <Header title={currentMode.name} papersCount={papers.length} health={health} onTogglePanel={()=>setPanelOpen(!panelOpen)} panelOpen={panelOpen} />
        </div>

        <div className="flex-1 flex overflow-hidden">
          {/* Conversation */}
          <div className="flex-1 flex flex-col min-w-0">
            <ConversationStream history={history} activeTasks={activeTasks} activeMode={activeMode} currentMode={currentMode} copiedId={copiedId} onCopy={copyText}
              selectedPapers={selectedPapers} papers={papers} onDeselect={(pid)=>setModeSelections(s=>({...s,[activeMode]:(s[activeMode]||[]).filter(id=>id!==pid)}))} />
            <BottomBar draft={draft} onChange={setDraft} onKeyDown={onKeyDown} onSend={sendChat} onAttach={handleUploadClick}
              loading={loading} canSend={Boolean(draft.trim())} currentMode={currentMode} activeMode={activeMode}
              onCompare={runCompare} onReview={runReview} canCompare={canCompare} canReview={canReview}
              selectedPapers={selectedPapers} reviewTargetPaper={reviewTarget}
              comparatorPreset={compPreset} onCompPresetChange={setCompPreset}
              contextCount={selectedIds.length} />
          </div>

          {/* File panel */}
          {panelOpen && (
            <div className="w-[280px] shrink-0">
              <FilePanel papers={allPapers} selectedIds={selectedIds} activeMode={activeMode} fileStates={fileStates}
                onSelect={selectPaper} onUploadClick={handleUploadClick}
                onDragOver={handleDragOver} onDragLeave={handleDragLeave} onDrop={handleDrop} dragging={dragging} onDelete={deletePaper} />
            </div>
          )}
        </div>
      </div>

      {/* Error toast */}
      {error && (
        <div className="fixed top-4 left-1/2 -translate-x-1/2 z-[60] anim-slide-up">
          <div className="rounded-[14px] border border-red-400/20 px-4 py-2.5 text-sm text-red-300 flex items-center gap-3" style={{background:"var(--surface)",backdropFilter:"blur(20px)",boxShadow:"0 8px 32px rgba(0,0,0,0.5)"}}>
            <span>{error}</span>
            <button onClick={()=>setError("")} className="text-red-400/50 hover:text-red-300"><Icon name="x" className="w-4 h-4"/></button>
          </div>
        </div>
      )}

      {/* Hidden file input */}
      <input ref={fileRef} type="file" accept=".pdf,application/pdf" multiple className="hidden" onChange={onFileChange} />
    </div>
  );
}

/* ═══ Mount ═══ */
if(typeof window!=="undefined"&&window.ReactDOM){
  ReactDOM.createRoot(document.getElementById("root")).render(<ResearchAgent/>);
}
