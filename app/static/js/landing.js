(function(){
 document.documentElement.dataset.theme='light';
 const $=s=>document.querySelector(s),reduce=matchMedia('(prefers-reduced-motion:reduce)').matches;
 function go(e,href){e.preventDefault();const v=document.createElement('div');v.className='veil';v.style.setProperty('--x',e.clientX+'px');v.style.setProperty('--y',e.clientY+'px');v.style.clipPath='circle(0% at var(--x) var(--y))';document.body.append(v);
  try{sessionStorage.setItem('vg-enter','1');localStorage.getItem('vg-theme')||localStorage.setItem('vg-theme','dark')}catch(_){}requestAnimationFrame(()=>{v.classList.add('go');setTimeout(()=>location.href=href,640)})}
 document.querySelectorAll('[data-enter]').forEach(a=>a.addEventListener('click',e=>go(e,a.getAttribute('href'))));
 // dot field: every dot is an anonymous seat; a slow wave of attention crosses the room, the cursor lifts the dots near it
 const cv=$('#field'),x=cv.getContext('2d');let W,H,dots=[],mx=-999,my=-999,dpr=Math.min(devicePixelRatio||1,2);
 function size(){W=cv.clientWidth;H=cv.clientHeight;cv.width=W*dpr;cv.height=H*dpr;x.setTransform(dpr,0,0,dpr,0,0);dots=[];const gap=W<700?26:34;
  for(let r=0;r*gap*.86<H+gap;r++)for(let c=0;c*gap<W+gap;c++)dots.push({x:c*gap+(r%2?gap/2:0),y:r*gap*.86,p:Math.random()*6.28})}
 addEventListener('resize',size);size();
 addEventListener('pointermove',e=>{const b=cv.getBoundingClientRect();mx=e.clientX-b.left;my=e.clientY-b.top});
 const ink=[22,19,15],ver=[224,68,42];
 function draw(t){t/=1000;x.clearRect(0,0,W,H);
  for(const d of dots){const sweep=Math.max(0,Math.sin(d.x*.0042+d.y*.0021-t*.9)),br=.5+.5*Math.sin(t*.7+d.p),dx=d.x-mx,dy=d.y-my,inf=Math.exp(-(dx*dx+dy*dy)/(2*130*130));
   const a=Math.min(1,.5*sweep+inf),rad=1.3+1.1*br+3.2*sweep+4.5*inf,al=.16+.5*a+.1*br;
   const c=[0,1,2].map(i=>Math.round(ink[i]+(ver[i]-ink[i])*a));x.fillStyle=`rgba(${c[0]},${c[1]},${c[2]},${Math.min(.9,al)})`;x.beginPath();x.arc(d.x,d.y,rad,0,6.283);x.fill()}
  if(!reduce)requestAnimationFrame(draw)}
 requestAnimationFrame(draw);
 // scroll reveal
 const io=new IntersectionObserver(es=>es.forEach(e=>{if(e.isIntersecting){e.target.classList.add('on');io.unobserve(e.target)}}),{threshold:.15});
 document.querySelectorAll('.reveal').forEach(el=>io.observe(el));
 // count-up for the evidence numbers
 const cu=new IntersectionObserver(es=>es.forEach(e=>{if(!e.isIntersecting)return;cu.unobserve(e.target);const to=+e.target.dataset.to,dec=+(e.target.dataset.dec||0),t0=performance.now();
  (function f(n){const k=Math.min(1,(n-t0)/1400),v=to*(1-Math.pow(1-k,3));e.target.firstChild.nodeValue=(dec?v.toFixed(dec):Math.round(v).toLocaleString());if(k<1)requestAnimationFrame(f)})(t0)}),{threshold:.6});
 document.querySelectorAll('[data-to]').forEach(el=>cu.observe(el));
 // viz dots for the "summarise" animation
 const m=$('.merge');if(m){for(let i=0;i<14;i++){const d=document.createElement('i');d.style.cssText=`--i:${i};--x:${(i%7)*20}px;--y:${Math.floor(i/7)*34+10}px;left:0;top:0`;m.prepend(d)}}
 const b=$('.bars');if(b){for(let i=0;i<22;i++){const e=document.createElement('i');e.style.cssText=`--i:${i};--a:${14+Math.random()*50}px;--b:${14+Math.random()*80}px`;b.append(e)}}
})();
