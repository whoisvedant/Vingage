document.addEventListener('DOMContentLoaded',()=>{
 const $=id=>document.getElementById(id);let range=180,last=null;
 const chart=new LineChart($('chart'),{live:true,rel:true,empty:'Chart starts once enough people are in view'});
 $('present').innerHTML=VG_icon('full')+'Presenter view';
 function VG_icon(n){return svg(n)}
 $('present').onclick=()=>{document.body.classList.toggle('presenter');if(document.body.classList.contains('presenter')){document.documentElement.requestFullscreen?.().catch(()=>{})}else document.fullscreenElement&&document.exitFullscreen()};
 document.addEventListener('keydown',e=>{if(e.key==='Escape')document.body.classList.remove('presenter')});
 document.addEventListener('fullscreenchange',()=>{if(!document.fullscreenElement)document.body.classList.remove('presenter')});
 document.querySelectorAll('#range button').forEach(b=>b.onclick=()=>{document.querySelectorAll('#range button').forEach(x=>x.classList.remove('on'));b.classList.add('on');range=+b.dataset.r;if(last)render(last)});
 document.addEventListener('themechange',()=>last&&render(last));
 $('sbtn').onclick=async()=>{const a=last&&last.session.active;await VG.post(a?'/api/session/stop':'/api/session/start',{name:$('sname').value});$('sname').value=''};
 function banners(s){const b=[];
  if(s.error)b.push(['bad','Engine error: '+s.error+'. Check that the webcam is free and the model file is in app/models.']);
  if(s.mode==='simulated')b.push(['info','Simulation mode: the audience below is generated for testing. These numbers are not measurements.']);
  if(s.k_min<3)b.push(['warn',`Privacy threshold lowered to k = ${s.k_min}. Use this only for testing: with fewer than 3 people a group statistic can reveal an individual.`]);
  $('banners').innerHTML=b.map(([c,t])=>`<div class="banner ${c}">${t}</div>`).join('')}
 function render(s){last=s;banners(s);
  const ok=s.status==='ok',st=$('state');
  st.className='pill '+(s.error?'bad':ok?'good':'warn');st.innerHTML='<i></i>'+(s.error?'Error':ok?'Live':s.status==='warming_up'?'Warming up':'Waiting for audience');
  $('k-share').textContent=ok?VG.pct(s.engaged_share):'–';ring($('k-ring'),ok?s.engaged_share:0,css('--brand'));
  const h=s.history;let d='';if(ok&&h.length>30){const old=h[Math.max(0,h.length-31)].share-0,dv=Math.round((s.engaged_share-old)*100);if(dv)d=`<span class="delta ${dv>0?'up':'dn'}">${dv>0?'▲':'▼'} ${Math.abs(dv)} pts</span>`}
  $('k-share-sub').innerHTML=ok?`of ${s.n} people${d}`:s.status==='warming_up'?'First score in about 5 seconds':`${s.faces_in_view} of ${s.k_min} people needed`;
  $('k-score').textContent=ok?s.mean_score.toFixed(2):'–';$('k-scorebar').style.width=ok?Math.round(s.mean_score*100)+'%':'0';
  $('k-n').textContent=ok?s.n:s.faces_in_view;$('k-n').style.color=ok?'':'var(--mut)';
  $('k-dots').innerHTML=Array.from({length:Math.min(s.k_min,10)},(_,i)=>`<i class="${i<s.faces_in_view?'f':''}"></i>`).join('');
  $('k-n-sub').textContent=ok?`Minimum ${s.k_min} met. Statistics are live.`:`Statistics stay hidden below ${s.k_min}.`;
  const se=s.session;$('k-sess').textContent=se.active?mmss(se.elapsed):'–';$('k-sess-sub').textContent=se.active?`Recording: ${se.name}`:'Not recording';
  $('sbtn').innerHTML=se.active?svg('stop')+'Stop':svg('play')+'Record';$('sbtn').className='btn '+(se.active?'dng':'pri');$('sname').style.display=se.active?'none':'';
  const now=h.length?h[h.length-1].t:0,pts=range?h.filter(p=>now-p.t<=range):h;chart.set(pts.map((p,i)=>{const w=pts.slice(Math.max(0,i-4),i+1);return {...p,share:w.reduce((a,q)=>a+q.share,0)/w.length}}),s.alert);
  const v=pts.map(p=>p.share),thr=s.alert;
  $('s-avg').textContent=v.length?VG.pct(v.reduce((a,b)=>a+b,0)/v.length):'–';$('s-peak').textContent=v.length?VG.pct(Math.max(...v)):'–';$('s-low').textContent=v.length?VG.pct(Math.min(...v)):'–';
  $('s-above').textContent=v.length?VG.pct(v.filter(x=>x>=thr).length/v.length):'–';
  $('feed').innerHTML=s.events.length?s.events.map(e=>`<div class="ev ${e.type}"><i></i><span>${e.msg}</span><time>${mmss(e.t)}</time></div>`).join(''):'<div class="empty sm">No events yet</div>'}
 VG.on(render);
});
