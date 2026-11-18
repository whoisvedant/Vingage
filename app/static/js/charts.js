const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
function mmss(s){s=Math.max(0,Math.round(s));return Math.floor(s/60)+':'+String(s%60).padStart(2,'0')}
function rel(s){return s>=-0.5?'now':'−'+mmss(-s)}
class LineChart{
  constructor(el,o={}){this.el=el;this.o=Object.assign({fmt:v=>Math.round(v*100)+'%',xfmt:mmss,gap:3.5,label:'engaged',empty:'No data yet'},o);this.pts=[];this.thr=null;this.marks=[];
    this.cv=document.createElement('canvas');this.tip=document.createElement('div');this.tip.className='tip';el.append(this.cv,this.tip);
    this.cv.addEventListener('mousemove',e=>{const r=this.cv.getBoundingClientRect();this.mx=e.clientX-r.left;this.draw()});
    this.cv.addEventListener('mouseleave',()=>{this.mx=null;this.draw()});addEventListener('resize',()=>this.draw())}
  set(pts,thr,marks){this.pts=pts;this.thr=thr;this.marks=marks||[];this.draw()}
  draw(){const c=this.cv,r=devicePixelRatio||1,W=this.el.clientWidth,H=this.el.clientHeight;if(!W)return;c.width=W*r;c.height=H*r;const x=c.getContext('2d');x.scale(r,r);
    const L=40,R=W-6,T=10,B=H-26,mut=css('--mut'),line=css('--line'),verm=css('--verm'),ink=css('--ink'),panel=css('--panel');
    x.font='11px "Geist Mono",ui-monospace,monospace';x.textBaseline='middle';x.lineWidth=1;
    for(let i=0;i<=4;i++){const y=Math.round(B-(B-T)*i/4)+.5;x.strokeStyle=line;x.beginPath();x.moveTo(L,y);x.lineTo(R,y);x.stroke();x.fillStyle=mut;x.textAlign='right';x.fillText(i*25,L-10,y)}
    const p=this.pts;this.tip.style.opacity=0;
    if(p.length<2){x.fillStyle=mut;x.textAlign='center';x.font='italic 22px "Instrument Serif",serif';x.fillText(this.o.empty,(L+R)/2,(T+B)/2);return}
    const t0=p[0].t,t1=p[p.length-1].t,sx=t=>L+(R-L)*((t-t0)/Math.max(t1-t0,1)),sy=v=>B-(B-T)*v;
    x.textAlign='center';[0,.25,.5,.75,1].forEach(f=>{const t=t0+(t1-t0)*f;x.fillStyle=mut;x.fillText(this.o.rel?rel(t-t1):this.o.xfmt(t-t0),Math.min(Math.max(sx(t),L+16),R-16),H-8)});
    if(this.thr!=null){const y=sy(this.thr);x.setLineDash([2,4]);x.strokeStyle=ink;x.globalAlpha=.55;x.beginPath();x.moveTo(L,y);x.lineTo(R,y);x.stroke();x.setLineDash([]);x.globalAlpha=1;x.fillStyle=ink;x.textAlign='left';x.fillText((this.o.thrLabel||'ALERT ')+Math.round(this.thr*100),L+6,y-9)}
    (this.marks||[]).forEach(m=>{if(m.t<t0||m.t>t1)return;const px=sx(m.t);x.fillStyle=m.type==='warn'?verm:ink;x.beginPath();x.moveTo(px,B);x.lineTo(px-4,B+7);x.lineTo(px+4,B+7);x.fill()});
    const segs=[];let cur=[p[0]];for(let i=1;i<p.length;i++){if(p[i].t-p[i-1].t>this.o.gap){segs.push(cur);cur=[]}cur.push(p[i])}segs.push(cur);
    segs.forEach(s=>{if(!s.length)return;const g=x.createLinearGradient(0,T,0,B);g.addColorStop(0,verm+'38');g.addColorStop(1,verm+'00');
      x.beginPath();s.forEach((q,i)=>i?x.lineTo(sx(q.t),sy(q.share)):x.moveTo(sx(q.t),sy(q.share)));x.strokeStyle=verm;x.lineWidth=2;x.lineJoin='round';x.stroke();
      x.lineTo(sx(s[s.length-1].t),B);x.lineTo(sx(s[0].t),B);x.closePath();x.fillStyle=g;x.fill()});
    const dot=(q,rad)=>{x.beginPath();x.arc(sx(q.t),sy(q.share),rad,0,7);x.fillStyle=verm;x.fill();x.lineWidth=2.5;x.strokeStyle=panel;x.stroke()};
    dot(p[p.length-1],4.5);
    if(this.mx!=null&&this.mx>=L&&this.mx<=R){let b=p[0],bd=1e9;p.forEach(q=>{const d=Math.abs(sx(q.t)-this.mx);if(d<bd){bd=d;b=q}});
      x.strokeStyle=css('--line2');x.lineWidth=1;x.beginPath();x.moveTo(sx(b.t)+.5,T);x.lineTo(sx(b.t)+.5,B);x.stroke();dot(b,5);
      this.tip.innerHTML=`${this.o.fmt(b.share)} ${this.o.label}<br><span style="opacity:.65">${this.o.rel?rel(b.t-t1):this.o.xfmt(b.t-t0)}${b.n?' · '+b.n+' people':''}</span>`;
      this.tip.style.left=sx(b.t)+'px';this.tip.style.top=sy(b.share)+'px';this.tip.style.opacity=1}}
}
