# ui.py
import streamlit as st

CUSTOM_CSS = """
<style>
:root{
  --brand:#2563eb; --brand-2:#1d4ed8;
  --text:#0f172a; --muted:#64748b;
  --bg:#f8fafc; --panel:#ffffff; --border:#e5e7eb;
  --ring:rgba(37,99,235,.24);
  --ok:#16a34a; --warn:#d97706; --danger:#dc2626;
}
html, body, .stApp{ background:var(--bg)!important; color:var(--text)!important; }
.block-container{ padding-top:3.75rem; padding-bottom:4rem; max-width:920px; }
.stepper{
  display:flex; gap:10px; margin: 0 0 18px 0; flex-wrap:wrap;
  position:sticky; top:18px; z-index:5;
  padding-top:.25rem;
  background: linear-gradient(180deg, rgba(248,250,252,.94), rgba(248,250,252,0));
  border-radius: 12px;
}
@media (max-width: 900px){ .stepper{ position: static; top:auto; } }
.top-spacer{ height: 10px; }
.step{
  display:flex; align-items:center; gap:8px; padding:8px 12px; border-radius:9999px;
  border:1px solid var(--border); background:#fff; color:var(--muted); font-weight:700;
  box-shadow:0 2px 8px rgba(15,23,42,.06);
}
.step .dot{ width:10px; height:10px; border-radius:10px; background:#cbd5e1; }
.step.active{ border-color:var(--brand); color:var(--text);
  box-shadow:0 0 0 3px var(--ring) inset, 0 3px 10px rgba(37,99,235,.08); }
.step.active .dot{ background:var(--brand); }
.card{ background:#fff; border:1px solid var(--border); border-radius:14px; padding:18px;
       box-shadow:0 8px 28px rgba(15,23,42,.06); }
.stButton>button,.stDownloadButton>button{
  border-radius:12px; padding:10px 16px; font-weight:600; border:1px solid rgba(0,0,0,.02);
  background:linear-gradient(180deg,var(--brand),var(--brand-2)); color:#fff;
  box-shadow:0 6px 18px rgba(37,99,235,.25);
}
.stButton>button[kind="secondary"]{ background:#f3f4f6; color:var(--text);
  border:1px solid var(--border); box-shadow:none; }
.stButton>button[kind="secondary"]:hover{ background:#e5e7eb; }
div[role="radiogroup"] > div{ gap:12px !important; }
.badge{ display:inline-block; padding:8px 14px; border-radius:9999px; font-weight:700; }
.badge.ok{ background:rgba(22,163,74,.14); color:var(--ok); }
.badge.warn{ background:rgba(217,119,6,.14); color:var(--warn); }
.badge.danger{ background:rgba(220,38,38,.14); color:var(--danger); }
.upl-note{ color:var(--muted); font-size:.935rem; margin-top:-10px; }
</style>
"""

def inject_css():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

def stepper(active: str):
    seq = [("dashboard", "OSDI"), ("stories", "Story"), ("record", "Record"),
           ("predict", "Upload"), ("result", "Result")]
    st.markdown(
        '<div class="stepper">' +
        "".join([f'<div class="step {"active" if k==active else ""}"><span class="dot"></span>{label}</div>'
                 for k, label in seq]) +
        '</div>', unsafe_allow_html=True
    )
    st.markdown('<div class="top-spacer"></div>', unsafe_allow_html=True)

def recorder_html(duration_sec: int) -> str:
    html = """
    <style>
    .rec-wrap{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
    video{width:100%;max-height:360px;background:#000;border-radius:12px}
    button{padding:10px 16px;border-radius:12px;border:0;font-weight:700}
    .start{background:linear-gradient(180deg,#2563eb,#1d4ed8);color:#fff}
    .stop{background:#dc2626;color:#fff}
    .disabled{opacity:.6;pointer-events:none}
    .row{display:flex;gap:12px;margin-top:12px}
    .timer{font-weight:800;letter-spacing:.5px}
    </style>
    <div class="rec-wrap">
      <video id="preview" autoplay playsinline muted></video>
      <div class="row">
        <button class="start" id="startBtn">🎥 Start Recording</button>
        <button class="stop disabled" id="stopBtn">⏹ Stop</button>
        <span class="timer" id="timer">00:00 / 06:00</span>
      </div>
      <div id="after" style="margin-top:12px;"></div>
    </div>
    <script>
    const DURATION=___DUR___;
    const preview=document.getElementById('preview');
    const startBtn=document.getElementById('startBtn');
    const stopBtn=document.getElementById('stopBtn');
    const timerEl=document.getElementById('timer');
    const after=document.getElementById('after');
    let mediaStream,recorder,chunks=[],ticker;
    function fmt(n){return String(n).padStart(2,'0')}
    function updateTimer(e){const m=Math.floor(e/60),s=e%60; timerEl.textContent=`${fmt(m)}:${fmt(s)} / 06:00`}
    async function start(){
      try{ mediaStream=await navigator.mediaDevices.getUserMedia({video:{width:1280,height:720},audio:false}); }
      catch(e){ alert('Camera permission denied or unavailable.'); return; }
      preview.srcObject=mediaStream; chunks=[];
      const types=['video/webm;codecs=vp9','video/webm;codecs=vp8','video/webm']
        .filter(t=>window.MediaRecorder&&MediaRecorder.isTypeSupported(t));
      const mimeType=types.length?types[0]:'';
      recorder=new MediaRecorder(mediaStream,{mimeType});
      recorder.ondataavailable=e=>{ if(e.data&&e.data.size>0) chunks.push(e.data); };
      recorder.onstop=onStop; recorder.start();
      startBtn.classList.add('disabled'); stopBtn.classList.remove('disabled');
      let elapsed=0; updateTimer(0);
      ticker=setInterval(()=>{ elapsed+=1; updateTimer(elapsed); if(elapsed>=DURATION) stop(); },1000);
    }
    function stop(){
      try{ recorder&&recorder.state!=='inactive'&&recorder.stop(); }catch(_){}
      try{ mediaStream&&mediaStream.getTracks().forEach(t=>t.stop()); }catch(_){}
      clearInterval(ticker); startBtn.classList.remove('disabled'); stopBtn.classList.add('disabled');
    }
    function onStop(){
      const blob=new Blob(chunks,{type:'video/webm'}); const url=URL.createObjectURL(blob);
      preview.srcObject=null; preview.src=url; preview.controls=true; preview.muted=false; preview.play();
      const a=document.createElement('a'); a.href=url; a.download=`recording_${Date.now()}.webm`;
      a.textContent='⬇️ Download 6-minute Video';
      a.style='display:inline-block;margin-top:8px;padding:10px 16px;background:#1d4ed8;color:#fff;border-radius:12px;text-decoration:none;font-weight:700';
      after.innerHTML='<p>Recording complete! Download the video, then go to the next step to upload it for analysis.</p>'; after.appendChild(a);
    }
    startBtn.addEventListener('click',start); stopBtn.addEventListener('click',stop);
    </script>
    """
    return html.replace("___DUR___", str(duration_sec))
