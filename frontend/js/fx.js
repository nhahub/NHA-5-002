(()=>{const d=document,rm=matchMedia('(prefers-reduced-motion:reduce)').matches;
// Theme (dark/light) with persistence
const root=d.documentElement,saved=(()=>{try{return localStorage.getItem('theme')}catch(e){}})();
root.dataset.theme=saved||(matchMedia('(prefers-color-scheme:light)').matches?'light':'dark');
const tb=d.createElement('button');tb.className='icon-button';tb.setAttribute('aria-label','تبديل الثيم');
const ic=()=>tb.innerHTML=`<i class="lucide-${root.dataset.theme==='dark'?'sun':'moon'}"></i>`;ic();
tb.onclick=()=>{root.dataset.theme=root.dataset.theme==='dark'?'light':'dark';try{localStorage.setItem('theme',root.dataset.theme)}catch(e){}ic()};
d.querySelector('.topbar-actions')?.insertBefore(tb,d.getElementById('notifications-button'));
if(rm)return;
// Cursor spotlight on panels + subtle 3D tilt on cards
d.addEventListener('pointermove',e=>{const p=e.target.closest?.('.panel');if(p){const r=p.getBoundingClientRect();p.style.setProperty('--mx',e.clientX-r.left+'px');p.style.setProperty('--my',e.clientY-r.top+'px')}
 const c=e.target.closest?.('.goal-card,.course-card,.stat-card,.insight-card,.knowledge-node');if(c){const r=c.getBoundingClientRect(),x=(e.clientX-r.left)/r.width-.5,y=(e.clientY-r.top)/r.height-.5;c.style.transform=`translateY(-6px) perspective(700px) rotateX(${-y*5}deg) rotateY(${x*5}deg)`}});
d.addEventListener('pointerout',e=>{const c=e.target.closest?.('.goal-card,.course-card,.stat-card,.insight-card,.knowledge-node');if(c&&!c.contains(e.relatedTarget))c.style.transform=''});
// Ripple on buttons
d.addEventListener('pointerdown',e=>{const b=e.target.closest?.('.button,.icon-button,.filter-chip');if(!b)return;const r=b.getBoundingClientRect(),s=Math.max(r.width,r.height),o=d.createElement('span');
 Object.assign(o.style,{position:'absolute',borderRadius:'50%',pointerEvents:'none',width:s+'px',height:s+'px',left:e.clientX-r.left-s/2+'px',top:e.clientY-r.top-s/2+'px',background:'currentColor',opacity:.18,transform:'scale(0)',transition:'transform .6s,opacity .6s'});
 if(getComputedStyle(b).position==='static')b.style.position='relative';b.style.overflow='hidden';b.appendChild(o);requestAnimationFrame(()=>{o.style.transform='scale(2.4)';o.style.opacity=0});setTimeout(()=>o.remove(),650)});
// Count-up for numbers + animated progress bars, applied to whatever app.js renders
const num=/^(\d+(?:\.\d+)?)(%?)$/;
function enhance(root){root.querySelectorAll?.('.stat-card strong,.knowledge-score').forEach(el=>{if(el.dataset.fx)return;el.dataset.fx=1;const m=el.textContent.trim().match(num);if(!m)return;const to=+m[1],dec=m[1].includes('.')?1:0,t0=performance.now();(function f(t){const k=Math.min(1,(t-t0)/900),v=to*(1-Math.pow(1-k,3));el.textContent=v.toFixed(dec)+m[2];if(k<1)requestAnimationFrame(f)})(t0)});
 root.querySelectorAll?.('.progress-fill').forEach(el=>{if(el.dataset.fx)return;el.dataset.fx=1;const w=el.style.width;el.style.width='0';requestAnimationFrame(()=>requestAnimationFrame(()=>el.style.width=w))});
 root.querySelectorAll?.('.task-row,.event-row,.goal-card,.course-card,.timeline-row,.insight-card,.knowledge-node,.stat-card,.planner-task').forEach((el,i)=>{if(el.dataset.st)return;el.dataset.st=1;el.animate([{opacity:0,transform:'translateY(16px)'},{opacity:1,transform:'none'}],{duration:520,delay:Math.min(i,12)*45,easing:'cubic-bezier(.2,.8,.2,1)',fill:'backwards'})})}
let q=0;new MutationObserver(m=>{if(q)return;q=requestAnimationFrame(()=>{q=0;m.forEach(x=>x.addedNodes.forEach(n=>n.nodeType===1&&enhance(n)))})}).observe(d.querySelector('.page-container')||d.body,{childList:true,subtree:true});
enhance(d.body);
// Top progress line on page change
const bar=d.createElement('div');bar.style.cssText='position:fixed;top:0;right:0;height:2px;width:0;z-index:100;background:linear-gradient(90deg,var(--accent),var(--accent2));box-shadow:0 0 12px var(--accent)';d.body.appendChild(bar);
d.addEventListener('click',e=>{if(!e.target.closest('[data-page],[data-page-link]'))return;bar.style.transition='none';bar.style.width='0';requestAnimationFrame(()=>{bar.style.transition='width .5s cubic-bezier(.2,.8,.2,1)';bar.style.width='100%';setTimeout(()=>{bar.style.transition='opacity .3s';bar.style.opacity=0;setTimeout(()=>{bar.style.cssText=bar.style.cssText.replace(/opacity:[^;]*;?/,'');bar.style.width='0';bar.style.opacity=1},300)},520)})},true);
})();
