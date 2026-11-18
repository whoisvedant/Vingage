const ICON={moon:'<path d="M21 12.8A9 9 0 1111.2 3a7 7 0 009.8 9.8z"/>',sun:'<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
 play:'<circle cx="12" cy="12" r="5" fill="currentColor"/>',stop:'<rect x="7" y="7" width="10" height="10" rx="1" fill="currentColor"/>',full:'<path d="M8 3H5a2 2 0 00-2 2v3M21 8V5a2 2 0 00-2-2h-3M3 16v3a2 2 0 002 2h3M16 21h3a2 2 0 002-2v-3"/>',
 down:'<path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M7 10l5 5 5-5M12 15V3"/>',trash:'<polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 01-2 2H8a2 2 0 01-2-2L5 6"/>',arrow:'<path d="M5 12h14M13 6l6 6-6 6"/>'};
const svg=n=>`<svg viewBox="0 0 24 24">${ICON[n]}</svg>`;
const TABS=[['/room','Room'],['/replays','Replays'],['/evidence','Evidence'],['/privacy','Privacy'],['/setup','Setup']];
const VG={snap:null,subs:[],on(f){this.subs.push(f);if(this.snap)f(this.snap)},
 theme(t){document.documentElement.dataset.theme=t;try{localStorage.setItem('vg-theme',t)}catch(e){}document.dispatchEvent(new Event('themechange'));const b=document.getElementById('themebtn');if(b)b.innerHTML=svg(t==='dark'?'sun':'moon')},
 async api(p,o){const r=await fetch(p,o);if(!r.ok)throw new Error(r.status);return (r.headers.get('content-type')||'').includes('json')?r.json():r.text()},
 post(p,b){return this.api(p,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b||{})})},
 pct:v=>v==null?'–':Math.round(v*100)+'%',
 dur(s){s=Math.round(s);const h=Math.floor(s/3600),m=Math.floor(s%3600/60);return h?`${h}h ${m}m`:m?`${m}m ${String(s%60).padStart(2,'0')}s`:`${s}s`},
 date(t){return new Date(t*1000).toLocaleString([], {day:'numeric',month:'short',hour:'2-digit',minute:'2-digit'})},
 enter(href,ev){const v=document.createElement('div');v.className='veil';if(ev){v.style.setProperty('--x',ev.clientX+'px');v.style.setProperty('--y',ev.clientY+'px')}v.style.clipPath='circle(0% at var(--x,50%) var(--y,50%))';document.body.append(v);
  try{sessionStorage.setItem('vg-enter','1')}catch(e){}requestAnimationFrame(()=>{v.classList.add('go');setTimeout(()=>location.href=href,640)})}};
(function(){
 let t='dark';try{t=localStorage.getItem('vg-theme')||'dark'}catch(e){}document.documentElement.dataset.theme=t;
 document.addEventListener('DOMContentLoaded',()=>{
  let entered=false;try{entered=sessionStorage.getItem('vg-enter')==='1';sessionStorage.removeItem('vg-enter')}catch(e){}
  if(entered){const v=document.createElement('div');v.className='veil';document.body.append(v);requestAnimationFrame(()=>{v.classList.add('in');setTimeout(()=>v.remove(),950)})}
  const path=location.pathname.replace(/\/$/,'')||'/';
  const bar=document.createElement('header');bar.className='bar';
  bar.innerHTML=`<a class="word" href="/" title="Back to intro">VINGAGE<i></i></a><nav class="tabs">${TABS.map(([u,l])=>`<a href="${u}" class="${u===path?'on':''}">${l}</a>`).join('')}</nav>
   <div class="right"><span class="chip" id="conn"><i></i><span>Connecting</span></span><button class="ibtn" id="themebtn" aria-label="Toggle theme"></button></div>`;
  document.body.prepend(bar);
  const tb=document.getElementById('themebtn');tb.innerHTML=svg(document.documentElement.dataset.theme==='dark'?'sun':'moon');
  tb.onclick=()=>VG.theme(document.documentElement.dataset.theme==='dark'?'light':'dark');
  const conn=document.getElementById('conn');
  function open(){const ws=new WebSocket((location.protocol==='https:'?'wss://':'ws://')+location.host+'/ws');
   ws.onmessage=e=>{const s=JSON.parse(e.data);VG.snap=s;
     const ok=!s.error;conn.className='chip '+(ok?'on':'off');conn.lastChild.textContent=ok?(s.fps?s.fps.toFixed(0)+' fps · '+(s.mode==='simulated'?'simulated':'live'):'starting'):'camera problem';
     VG.subs.forEach(f=>f(s))};
   ws.onclose=()=>{conn.className='chip off';conn.lastChild.textContent='reconnecting';setTimeout(open,1500)}}
  open();
 });
})();
