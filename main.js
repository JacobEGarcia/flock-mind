/* FLOCK//MIND - neuroevolution flight lab
   200 fixed-topology neural nets (5-8-1) play Flappy Bird.
   Selection: elitism + tournament, uniform crossover, gaussian mutation.
   Every generation is archived (seed + champion weights) for deterministic replay. */
(() => {
'use strict';

/* ---------------- utils ---------------- */
function mulberry32(a){return function(){a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}
function gauss(){let u=0,v=0;while(u===0)u=Math.random();while(v===0)v=Math.random();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v)}
const clamp=(v,a,b)=>v<a?a:v>b?b:v;
const $=id=>document.getElementById(id);

/* ---------------- world constants ---------------- */
const W=480,H=640,GROUND=56;
const GRAV=1500,FLAP=-445,PXS=178,GAP=158,SPACING=262,PIPE_W=64;
const BIRD_X=118,BIRD_R=9;
const MAX_GEN_TIME=75;          // seconds of sim time before a generation is called

/* ---------------- network ---------------- */
const NI=5,NH=8,NO=1;
const NG=NI*NH+NH+NH*NO+NO;     // 57 genes
function randomGenome(rng){const g=new Float32Array(NG);for(let i=0;i<NG;i++)g[i]=(rng()*2-1)*0.9;return g}
function forward(g,inp,hid){
  for(let h=0;h<NH;h++){
    let s=g[NI*NH+h];
    for(let i=0;i<NI;i++)s+=inp[i]*g[i*NH+h];
    hid[h]=Math.tanh(s);
  }
  let o=g[NI*NH+NH+NH];
  for(let h=0;h<NH;h++)o+=hid[h]*g[NI*NH+NH+h];
  return o;
}

/* ---------------- ga ---------------- */
const POP=200,MUT_RATE=0.12,MUT_SIGMA=0.35,ELITES=2;
function tournament(birds){
  let best=null;
  for(let k=0;k<3;k++){
    const b=birds[(Math.random()*birds.length)|0];
    if(!best||b.fit>best.fit)best=b;
  }
  return best;
}
function crossover(a,b){
  const g=new Float32Array(NG);
  for(let i=0;i<NG;i++)g[i]=Math.random()<0.5?a[i]:b[i];
  return g;
}
function mutate(g){
  for(let i=0;i<NG;i++)if(Math.random()<MUT_RATE)g[i]+=gauss()*MUT_SIGMA;
  return g;
}

/* ---------------- sim state ---------------- */
const sim={
  mode:'live',          // live | replay
  paused:false,
  speed:4,              // substeps per frame; 0 = MAX
  gen:0,
  baseSeed:(Math.random()*0x7fffffff)|0,
  world:null,
  birds:[],
  history:[],           // {gen,best,mean,median,score,seed,weights}
  bestEver:0,
  bestEverScore:0,
  simClock:0,
  leader:null,
};
const inpBuf=new Float32Array(NI);

function makeWorld(seed){
  const rng=mulberry32(seed);
  return {seed,rng,pipes:[],nextX:W+160,t:0,dist:0,maxScore:0};
}
function spawnPipe(w){
  const m=64;
  const gapY=m+w.rng()*(H-GROUND-2*m-GAP);
  w.pipes.push({x:w.nextX,gapY});
  w.nextX+=SPACING;
}
function stepWorld(w,dt){
  w.t+=dt;w.dist+=PXS*dt;w.nextX-=PXS*dt;
  while(w.nextX<W+SPACING)spawnPipe(w);
  for(let i=w.pipes.length-1;i>=0;i--)if(w.pipes[i].x+PIPE_W<-30)w.pipes.splice(i,1);
  for(const p of w.pipes)p.x-=PXS*dt;
}
function nextPipe(w){
  for(const p of w.pipes)if(p.x+PIPE_W>BIRD_X-BIRD_R)return p;
  return w.pipes[0];
}
function stepBird(b,w,dt){
  const p=nextPipe(w);
  const gapC=p.gapY+GAP/2;
  inpBuf[0]=b.y/H;
  inpBuf[1]=b.vy/700;
  inpBuf[2]=clamp((p.x-BIRD_X)/W,-1,1);
  inpBuf[3]=(p.gapY-b.y)/H;
  inpBuf[4]=(p.gapY+GAP-b.y)/H;
  b.out=forward(b.g,inpBuf,b.act);
  if(b.out>0){b.vy=FLAP;b.flapT=0.12;}
  b.vy+=GRAV*dt;
  b.y+=b.vy*dt;
  if(b.y<BIRD_R){b.y=BIRD_R;b.vy=0;}
  if(b.flapT>0)b.flapT-=dt;
  // scoring: any pipe now fully behind this bird
  for(const q of w.pipes){
    if(q.x+PIPE_W>=BIRD_X-BIRD_R)break;
    if(!q.passedSet)q.passedSet=new Set();
    if(!q.passedSet.has(b.id)){
      q.passedSet.add(b.id);b.score++;
      if(b.score>w.maxScore)w.maxScore=b.score;
    }
  }
  // shaping: small bonus for holding the gap line while a pipe approaches
  const dx=p.x-BIRD_X;
  if(dx>-40&&dx<W*0.8){
    b.align+=dt*55*(1-clamp(Math.abs(b.y-gapC)/(H*0.45),0,1));
  }
  b.fit=w.dist+b.score*1000+b.align;
  // collisions
  if(b.y>=H-GROUND-BIRD_R){b.alive=false;b.deadT=0;return;}
  for(const q of w.pipes){
    if(q.x>BIRD_X+BIRD_R+PIPE_W)break;
    if(q.x<BIRD_X+BIRD_R&&q.x+PIPE_W>BIRD_X-BIRD_R){
      if(b.y-BIRD_R<q.gapY||b.y+BIRD_R>q.gapY+GAP){b.alive=false;b.deadT=0;return;}
    }
  }
}

function newGeneration(genomes){
  sim.gen++;
  const seed=(sim.baseSeed+Math.imul(sim.gen,2654435761))>>>0;
  sim.world=makeWorld(seed);
  for(let i=0;i<3;i++)spawnPipe(sim.world);
  sim.birds=genomes.map((g,i)=>({
    id:i,g,y:H/2,vy:0,alive:true,fit:0,score:0,out:0,flapT:0,align:0,
    act:new Float32Array(NH),deadT:-1,
  }));
  sim.leader=sim.birds[0];
}
function endGeneration(){
  const fits=sim.birds.map(b=>b.fit).sort((a,b)=>b-a);
  const best=fits[0],mean=fits.reduce((s,v)=>s+v,0)/fits.length,median=fits[fits.length>>1];
  const champ=sim.birds.reduce((a,b)=>b.fit>a.fit?b:a);
  if(best>sim.bestEver)sim.bestEver=best;
  if(sim.world.maxScore>sim.bestEverScore)sim.bestEverScore=sim.world.maxScore;
  sim.history.push({
    gen:sim.gen,best,mean,median,score:sim.world.maxScore,
    seed:sim.world.seed,weights:champ.g.slice(),
  });
  if(sim.history.length>600)sim.history.shift();
  syncScrub();
  // evolve
  const sorted=[...sim.birds].sort((a,b)=>b.fit-a.fit);
  const next=[];
  for(let e=0;e<ELITES;e++)next.push(sorted[e].g.slice());
  for(let e=0;e<4;e++)for(let k=0;k<4;k++)next.push(mutate(sorted[e].g.slice()));
  while(next.length<POP)next.push(mutate(crossover(tournament(sorted).g,tournament(sorted).g)));
  newGeneration(next);
}
function reset(){
  sim.gen=0;sim.history=[];sim.bestEver=0;sim.bestEverScore=0;sim.simClock=0;
  sim.baseSeed=(Math.random()*0x7fffffff)|0;
  sim.mode='live';
  const rng=mulberry32(sim.baseSeed);
  const genomes=[];for(let i=0;i<POP;i++)genomes.push(randomGenome(rng));
  newGeneration(genomes);
  setModeUI();
  syncScrub();
}

/* ---------------- replay (archive) ---------------- */
const replay={active:false,gen:0,bird:null,world:null};
function startReplay(entry){
  replay.active=true;replay.gen=entry.gen;
  replay.world=makeWorld(entry.seed);
  for(let i=0;i<3;i++)spawnPipe(replay.world);
  replay.bird={id:0,g:entry.weights,y:H/2,vy:0,alive:true,fit:0,score:0,out:0,flapT:0,align:0,act:new Float32Array(NH),deadT:-1};
  sim.mode='replay';
  $('archiveText').textContent='ARCHIVE // GEN '+entry.gen+' CHAMPION';
  $('archiveBanner').classList.remove('hidden');
  setModeUI();
}
function stopReplay(){
  replay.active=false;sim.mode='live';
  $('archiveBanner').classList.add('hidden');
  setModeUI();
}

/* ---------------- fixed-step tick ---------------- */
function tick(dt){
  if(sim.mode==='replay'){
    const w=replay.world,b=replay.bird;
    if(b.alive){stepWorld(w,dt);stepBird(b,w,dt);}
    sim.simClock+=dt;
    return;
  }
  const w=sim.world;
  stepWorld(w,dt);
  sim.simClock+=dt;
  let alive=0,lead=null;
  for(const b of sim.birds){
    if(!b.alive){if(b.deadT>=0)b.deadT+=dt;continue;}
    stepBird(b,w,dt);
    if(b.alive){alive++;if(!lead||b.fit>lead.fit)lead=b;}
    else b.deadT=0;
  }
  sim.alive=alive;
  sim.leader=lead;
  if(alive===0||w.t>MAX_GEN_TIME)endGeneration();
}

/* ---------------- rendering: stage ---------------- */
const stage=$('stage'),sctx=stage.getContext('2d');
const DPR=Math.min(window.devicePixelRatio||1,2);
stage.width=W*DPR;stage.height=H*DPR;sctx.scale(DPR,DPR);

function drawBackground(ctx){
  const g=ctx.createLinearGradient(0,0,0,H);
  g.addColorStop(0,'#0a0e14');g.addColorStop(.7,'#0a0c10');g.addColorStop(1,'#0c0f13');
  ctx.fillStyle=g;ctx.fillRect(0,0,W,H);
  ctx.strokeStyle='rgba(60,72,86,0.10)';ctx.lineWidth=1;
  ctx.beginPath();
  for(let y=80;y<H-GROUND;y+=80){ctx.moveTo(0,y+.5);ctx.lineTo(W,y+.5);}
  for(let x=80;x<W;x+=80){ctx.moveTo(x+.5,0);ctx.lineTo(x+.5,H-GROUND);}
  ctx.stroke();
  // altitude ruler
  ctx.strokeStyle='rgba(95,107,120,0.35)';
  ctx.beginPath();
  for(let y=40;y<H-GROUND;y+=40){ctx.moveTo(0,y+.5);ctx.lineTo(y%160===40?10:6,y+.5);}
  ctx.stroke();
  // ground
  ctx.fillStyle='#0e1218';ctx.fillRect(0,H-GROUND,W,GROUND);
  ctx.strokeStyle='#2a333f';ctx.beginPath();ctx.moveTo(0,H-GROUND+.5);ctx.lineTo(W,H-GROUND+.5);ctx.stroke();
  const off=(sim.mode==='replay'?replay.world.dist:sim.world.dist)%32;
  ctx.strokeStyle='rgba(95,107,120,0.3)';
  ctx.beginPath();
  for(let x=-off;x<W;x+=32){ctx.moveTo(x,H-GROUND+8);ctx.lineTo(x-8,H);}
  ctx.stroke();
}
function drawPipes(ctx,w){
  for(const p of w.pipes){
    ctx.fillStyle='#141a22';
    ctx.fillRect(p.x,0,PIPE_W,p.gapY);
    ctx.fillRect(p.x,p.gapY+GAP,PIPE_W,H-GROUND-p.gapY-GAP);
    ctx.strokeStyle='#2e3a48';ctx.lineWidth=1;
    ctx.strokeRect(p.x+.5,-2,PIPE_W-1,p.gapY+2);
    ctx.strokeRect(p.x+.5,p.gapY+GAP+.5,PIPE_W-1,H-GROUND-p.gapY-GAP);
    // gap edge markers
    ctx.strokeStyle='rgba(255,180,84,0.5)';
    ctx.beginPath();
    ctx.moveTo(p.x,p.gapY+.5);ctx.lineTo(p.x+PIPE_W,p.gapY+.5);
    ctx.moveTo(p.x,p.gapY+GAP+.5);ctx.lineTo(p.x+PIPE_W,p.gapY+GAP+.5);
    ctx.stroke();
  }
}
function drawBird(ctx,b,isLeader,isReplay){
  const a=clamp(b.vy/900,-0.5,1.1);
  ctx.save();
  ctx.translate(BIRD_X,b.y);ctx.rotate(a*0.6);
  if(isLeader||isReplay){
    ctx.strokeStyle='rgba(255,180,84,0.55)';ctx.lineWidth=1;
    ctx.beginPath();ctx.arc(0,0,BIRD_R+6,0,Math.PI*2);ctx.stroke();
    ctx.fillStyle='#ffb454';
  }else{
    ctx.fillStyle='rgba(190,200,214,0.30)';
  }
  const wing=b.flapT>0?-4:2;
  ctx.beginPath();
  ctx.moveTo(10,0);ctx.lineTo(-8,-6);ctx.lineTo(-5,0);ctx.lineTo(-8,6);ctx.closePath();ctx.fill();
  ctx.strokeStyle=isLeader||isReplay?'#ffd69a':'rgba(190,200,214,0.45)';
  ctx.lineWidth=1.4;
  ctx.beginPath();ctx.moveTo(-2,0);ctx.lineTo(-7,wing);ctx.stroke();
  ctx.restore();
}
function drawStage(){
  const replaying=sim.mode==='replay';
  const w=replaying?replay.world:sim.world;
  drawBackground(sctx);
  drawPipes(sctx,w);
  if(replaying){
    if(replay.bird.alive)drawBird(sctx,replay.bird,false,true);
  }else{
    for(const b of sim.birds){
      if(b.alive){if(b!==sim.leader)drawBird(sctx,b,false,false);}
      else if(b.deadT>=0&&b.deadT<0.5){
        sctx.globalAlpha=1-b.deadT*2;
        drawBird(sctx,b,false,false);
        sctx.globalAlpha=1;
      }
    }
    if(sim.leader)drawBird(sctx,sim.leader,true,false);
  }
  // world readout strip
  sctx.fillStyle='rgba(95,107,120,0.9)';
  sctx.font='10px "IBM Plex Mono",monospace';
  sctx.fillText(replaying?('GEN '+String(replay.gen).padStart(4,'0')+' // REPLAY'):('GEN '+String(sim.gen).padStart(4,'0')),12,20);
  sctx.fillText((replaying?replay.bird.score:w.maxScore)+' GATES',12,36);
  sctx.textAlign='right';
  sctx.fillText('ALT '+Math.round((H-GROUND-(replaying?replay.bird.y:(sim.leader?sim.leader.y:H/2))))+'px',W-12,20);
  sctx.textAlign='left';
}

/* ---------------- rendering: chart ---------------- */
const chart=$('chart'),cctx=chart.getContext('2d');
let chartHover=-1;
function sizeCanvas(c){
  const r=c.getBoundingClientRect();
  c.width=r.width*DPR;c.height=r.height*DPR;
  c.getContext('2d').setTransform(DPR,0,0,DPR,0,0);
  return r;
}
function drawChart(){
  const r=chart.getBoundingClientRect(),cw=r.width,ch=r.height;
  cctx.clearRect(0,0,cw,ch);
  const h=sim.history;
  if(h.length<1){
    cctx.fillStyle='#39434e';cctx.font='9px "IBM Plex Mono",monospace';
    cctx.fillText('AWAITING GENERATION 1...',8,ch/2);
    return;
  }
  const pad={l:8,r:8,t:10,b:14};
  const maxV=Math.max(...h.map(e=>e.best))*1.05;
  const x=g=>pad.l+(h.length===1?0:(g-1)/(h.length-1))*(cw-pad.l-pad.r);
  const y=v=>ch-pad.b-(v/maxV)*(ch-pad.t-pad.b);
  cctx.strokeStyle='#1a212a';cctx.lineWidth=1;
  for(let i=1;i<4;i++){const gy=pad.t+(ch-pad.t-pad.b)*i/4;cctx.beginPath();cctx.moveTo(pad.l,gy+.5);cctx.lineTo(cw-pad.r,gy+.5);cctx.stroke();}
  // mean
  cctx.strokeStyle='rgba(111,211,231,0.55)';cctx.lineWidth=1;
  cctx.beginPath();h.forEach((e,i)=>i?cctx.lineTo(x(e.gen),y(e.mean)):cctx.moveTo(x(e.gen),y(e.mean)));cctx.stroke();
  // best
  cctx.strokeStyle='#ffb454';cctx.lineWidth=1.5;
  cctx.beginPath();h.forEach((e,i)=>i?cctx.lineTo(x(e.gen),y(e.best)):cctx.moveTo(x(e.gen),y(e.best)));cctx.stroke();
  const last=h[h.length-1];
  cctx.fillStyle='#ffb454';
  cctx.beginPath();cctx.arc(x(last.gen),y(last.best),2.4,0,Math.PI*2);cctx.fill();
  // hover marker
  const hi=chartHover>=0&&chartHover<h.length?chartHover:h.length-1;
  const he=h[hi];
  cctx.strokeStyle='rgba(201,210,220,0.25)';
  cctx.beginPath();cctx.moveTo(x(he.gen)+.5,pad.t);cctx.lineTo(x(he.gen)+.5,ch-pad.b);cctx.stroke();
  cctx.fillStyle='#c9d2dc';cctx.font='9px "IBM Plex Mono",monospace';
  cctx.fillText('GEN '+he.gen+'  BEST '+Math.round(he.best)+'  MEAN '+Math.round(he.mean),pad.l,ch-3);
}

/* ---------------- rendering: network ---------------- */
const netc=$('net'),nctx=net.getContext('2d');
function drawNet(){
  const r=netc.getBoundingClientRect(),cw=r.width,ch=r.height;
  nctx.clearRect(0,0,cw,ch);
  let g,act,out,label;
  if(sim.mode==='replay'){g=replay.bird.g;act=replay.bird.act;out=replay.bird.out;label='GEN '+replay.gen;}
  else if(sim.leader){g=sim.leader.g;act=sim.leader.act;out=sim.leader.out;label='GEN '+sim.gen;}
  else{const e=sim.history[sim.history.length-1];if(!e)return;g=e.weights;act=new Float32Array(NH);forward(g,new Float32Array(NI),act);out=0;label='GEN '+e.gen;}
  $('netGen').textContent='// '+label;
  const cx=[cw*0.18,cw*0.52,cw*0.88];
  const ys=(n)=>{const arr=[];for(let i=0;i<n;i++)arr.push(ch*(n===1?0.5:(0.1+0.8*i/(n-1))));return arr;};
  const yi=ys(NI),yh=ys(NH),yo=ys(NO);
  // edges input->hidden
  for(let i=0;i<NI;i++)for(let h=0;h<NH;h++){
    const w=g[i*NH+h],aw=Math.min(Math.abs(w)/1.8,1);
    if(aw<0.06)continue;
    nctx.strokeStyle=w>0?`rgba(255,180,84,${aw*0.55})`:`rgba(111,211,231,${aw*0.45})`;
    nctx.lineWidth=aw*1.6;
    nctx.beginPath();nctx.moveTo(cx[0],yi[i]);nctx.lineTo(cx[1],yh[h]);nctx.stroke();
  }
  // edges hidden->out
  for(let h=0;h<NH;h++){
    const w=g[NI*NH+NH+h],aw=Math.min(Math.abs(w)/1.8,1);
    if(aw<0.06)continue;
    nctx.strokeStyle=w>0?`rgba(255,180,84,${aw*0.6})`:`rgba(111,211,231,${aw*0.5})`;
    nctx.lineWidth=aw*1.8;
    nctx.beginPath();nctx.moveTo(cx[1],yh[h]);nctx.lineTo(cx[2],yo[0]);nctx.stroke();
  }
  // input labels
  const inNames=['ALT','VEL','DIST','GAP-T','GAP-B'];
  nctx.fillStyle='#39434e';nctx.font='8px "IBM Plex Mono",monospace';nctx.textAlign='right';
  for(let i=0;i<NI;i++)nctx.fillText(inNames[i],cx[0]-10,yi[i]+2.5);
  nctx.textAlign='left';
  // input nodes (live values)
  const p=sim.mode==='replay'?nextPipe(replay.world):nextPipe(sim.world);
  const by=sim.mode==='replay'?replay.bird.y:(sim.leader?sim.leader.y:H/2);
  const bvy=sim.mode==='replay'?replay.bird.vy:(sim.leader?sim.leader.vy:0);
  const iv=[by/H,bvy/700,clamp(((p?p.x:W)-BIRD_X)/W,-1,1),((p?p.gapY:0)-by)/H,((p?p.gapY+GAP:0)-by)/H];
  for(let i=0;i<NI;i++){
    const v=clamp(Math.abs(iv[i]),0,1);
    nctx.fillStyle=`rgba(111,211,231,${0.25+v*0.75})`;
    nctx.beginPath();nctx.arc(cx[0],yi[i],5,0,Math.PI*2);nctx.fill();
  }
  // hidden nodes
  for(let hh=0;hh<NH;hh++){
    const v=(act[hh]+1)/2;
    nctx.fillStyle=`rgba(255,180,84,${0.15+v*0.85})`;
    nctx.beginPath();nctx.arc(cx[1],yh[hh],6,0,Math.PI*2);nctx.fill();
    nctx.strokeStyle='#2a333f';nctx.lineWidth=1;nctx.stroke();
  }
  // output node
  const firing=out>0;
  nctx.fillStyle=firing?'#ffb454':'#1a212a';
  nctx.beginPath();nctx.arc(cx[2],yo[0],8,0,Math.PI*2);nctx.fill();
  nctx.strokeStyle=firing?'#ffd69a':'#2a333f';nctx.lineWidth=1.5;nctx.stroke();
  nctx.fillStyle='#5f6b78';nctx.font='8px "IBM Plex Mono",monospace';
  nctx.fillText('FLAP',cx[2]-10,yo[0]+20);
}

/* ---------------- HUD ---------------- */
let hudT=0;
function updateHUD(){
  $('tGen').textContent=sim.gen;
  $('tAlive').textContent=sim.mode==='replay'?(replay.bird.alive?1:0):sim.alive;
  $('tScore').textContent=sim.mode==='replay'?replay.bird.score:sim.world.maxScore;
  $('tBest').textContent=sim.bestEverScore;
  const last=sim.history[sim.history.length-1];
  $('tGenBest').textContent=last?Math.round(last.best):0;
  $('tMean').textContent=last?Math.round(last.mean):0;
  const t=sim.simClock;
  $('tClock').textContent=(t<60?t.toFixed(1)+'s':(t/60).toFixed(1)+'m');
  // genome checksum
  const src=sim.mode==='replay'?replay.bird.g:(sim.leader?sim.leader.g:null);
  if(src){
    let s=0;for(let i=0;i<8;i++)s=(s*31+(src[i]*1000|0))>>>0;
    $('tChecksum').textContent='0x'+s.toString(16).padStart(8,'0').toUpperCase();
  }
}
function setModeUI(){
  const led=$('modeLed'),lab=$('modeLabel');
  led.className='led'+(sim.paused?' paused':sim.mode==='replay'?' archive':'');
  lab.textContent=sim.paused?'PAUSED':sim.mode==='replay'?'ARCHIVE':'EVOLVING';
}

/* ---------------- scrub / archive UI ---------------- */
const scrub=$('scrub'),scrubLabel=$('scrubLabel'),replayBtn=$('replayBtn');
function syncScrub(){
  const n=sim.history.length;
  scrub.max=Math.max(n,1);
  if(sim.mode==='live'){
    scrub.value=n;scrubLabel.textContent=n?('GEN '+n+' / LIVE'):'LIVE';
    scrubLabel.classList.remove('active');
  }
  const has=n>0;
  scrub.disabled=!has;
  replayBtn.disabled=!has;
}
scrub.addEventListener('input',()=>{
  const i=+scrub.value-1;
  const e=sim.history[i];
  if(!e)return;
  scrubLabel.textContent='GEN '+e.gen+'  //  BEST '+Math.round(e.best)+'  SCORE '+e.score;
  scrubLabel.classList.add('active');
  chartHover=i;
  previewGen=e.gen;
});
scrub.addEventListener('change',()=>{drawNetPreview();});
replayBtn.addEventListener('click',()=>{
  const e=sim.history[+scrub.value-1];
  if(e)startReplay(e);
});
$('backToLive').addEventListener('click',()=>{stopReplay();syncScrub();});
let previewGen=-1;
function drawNetPreview(){
  // show archived champion statically in the net panel until next frame refresh
  const e=sim.history[+scrub.value-1];
  if(!e||sim.mode!=='live')return;
  const act=new Float32Array(NH);
  forward(e.weights,new Float32Array(NI),act);
  // temporary override render
  netPreviewOverride={g:e.weights,act,gen:e.gen};
  setTimeout(()=>netPreviewOverride=null,1200);
}
let netPreviewOverride=null;

chart.addEventListener('click',ev=>{
  if(!sim.history.length)return;
  const r=chart.getBoundingClientRect();
  const frac=(ev.clientX-r.left-8)/(r.width-16);
  const i=clamp(Math.round(frac*(sim.history.length-1)),0,sim.history.length-1);
  scrub.value=i+1;scrub.dispatchEvent(new Event('input'));scrub.dispatchEvent(new Event('change'));
});
chart.addEventListener('mousemove',ev=>{
  if(!sim.history.length)return;
  const r=chart.getBoundingClientRect();
  const frac=(ev.clientX-r.left-8)/(r.width-16);
  chartHover=clamp(Math.round(frac*(sim.history.length-1)),0,sim.history.length-1);
});

/* ---------------- controls ---------------- */
const speedBtns=[...document.querySelectorAll('.btn.speed')];
function setSpeed(v){
  sim.speed=v;
  speedBtns.forEach(b=>b.classList.toggle('active',+b.dataset.speed===v));
}
speedBtns.forEach(b=>b.addEventListener('click',()=>setSpeed(+b.dataset.speed)));
$('pauseBtn').addEventListener('click',()=>{
  sim.paused=!sim.paused;
  $('pauseBtn').textContent=sim.paused?'RESUME':'PAUSE';
  setModeUI();
});
$('restartBtn').addEventListener('click',reset);
window.addEventListener('keydown',e=>{
  if(e.target.tagName==='INPUT')return;
  if(e.code==='Space'){e.preventDefault();$('pauseBtn').click();}
  else if(e.key>='1'&&e.key<='8'){
    const v=[1,2,4,8,16,32,64,0][+e.key-1];setSpeed(v);
  }
  else if(e.key==='r'||e.key==='R')reset();
  else if(e.key==='l'||e.key==='L'){stopReplay();syncScrub();}
});

/* ---------------- main loop ---------------- */
let lastT=performance.now(),fpsEMA=60;
function frame(now){
  const dtMs=now-lastT;lastT=now;
  fpsEMA=fpsEMA*0.95+(1000/Math.max(dtMs,1))*0.05;
  if(!sim.paused){
    if(sim.speed===0){
      // MAX: bounded burst of sim, render once
      const t0=performance.now();
      let guard=0;
      while(performance.now()-t0<12&&guard++<4000)tick(1/60);
    }else{
      for(let s=0;s<sim.speed;s++)tick(1/60);
    }
  }
  drawStage();
  if(netPreviewOverride){
    const keep=sim.leader;
    drawNetOverride(netPreviewOverride);
  }else drawNet();
  drawChart();
  hudT+=dtMs;
  if(hudT>120){hudT=0;updateHUD();$('fps').textContent=Math.round(fpsEMA);}
  requestAnimationFrame(frame);
}
function drawNetOverride(o){
  const saved={leader:sim.leader,mode:sim.mode};
  // lightweight: draw using override genome
  const r=netc.getBoundingClientRect(),cw=r.width,ch=r.height;
  nctx.clearRect(0,0,cw,ch);
  $('netGen').textContent='// GEN '+o.gen+' (ARCHIVED)';
  const cx=[cw*0.18,cw*0.52,cw*0.88];
  const ys=(n)=>{const arr=[];for(let i=0;i<n;i++)arr.push(ch*(n===1?0.5:(0.1+0.8*i/(n-1))));return arr;};
  const yi=ys(NI),yh=ys(NH),yo=ys(NO);
  for(let i=0;i<NI;i++)for(let h=0;h<NH;h++){
    const w=o.g[i*NH+h],aw=Math.min(Math.abs(w)/1.8,1);
    if(aw<0.06)continue;
    nctx.strokeStyle=w>0?`rgba(255,180,84,${aw*0.55})`:`rgba(111,211,231,${aw*0.45})`;
    nctx.lineWidth=aw*1.6;
    nctx.beginPath();nctx.moveTo(cx[0],yi[i]);nctx.lineTo(cx[1],yh[h]);nctx.stroke();
  }
  for(let h=0;h<NH;h++){
    const w=o.g[NI*NH+NH+h],aw=Math.min(Math.abs(w)/1.8,1);
    if(aw<0.06)continue;
    nctx.strokeStyle=w>0?`rgba(255,180,84,${aw*0.6})`:`rgba(111,211,231,${aw*0.5})`;
    nctx.lineWidth=aw*1.8;
    nctx.beginPath();nctx.moveTo(cx[1],yh[h]);nctx.lineTo(cx[2],yo[0]);nctx.stroke();
  }
  const inNames=['ALT','VEL','DIST','GAP-T','GAP-B'];
  nctx.fillStyle='#39434e';nctx.font='8px "IBM Plex Mono",monospace';nctx.textAlign='right';
  for(let i=0;i<NI;i++)nctx.fillText(inNames[i],cx[0]-10,yi[i]+2.5);
  nctx.textAlign='left';
  for(let i=0;i<NI;i++){nctx.fillStyle='rgba(111,211,231,0.45)';nctx.beginPath();nctx.arc(cx[0],yi[i],5,0,Math.PI*2);nctx.fill();}
  for(let hh=0;hh<NH;hh++){nctx.fillStyle='#20262e';nctx.beginPath();nctx.arc(cx[1],yh[hh],6,0,Math.PI*2);nctx.fill();nctx.strokeStyle='#2a333f';nctx.stroke();}
  nctx.fillStyle='#1a212a';nctx.beginPath();nctx.arc(cx[2],yo[0],8,0,Math.PI*2);nctx.fill();
  nctx.strokeStyle='#2a333f';nctx.stroke();
  void saved;
}

/* ---------------- boot ---------------- */
const bootLines=[
  'NEAT-POP ENGINE ............ <span class="ok">OK</span>',
  'POPULATION 200 x GENOME 57 . <span class="ok">OK</span>',
  'COLLISION FIELD ............ <span class="ok">OK</span>',
  'ARCHIVE VAULT .............. <span class="ok">OK</span>',
  'SEEDING GENERATION 1',
];
function boot(){
  const el=$('bootLines');
  let i=0;
  const iv=setInterval(()=>{
    if(i<bootLines.length){el.innerHTML+=bootLines[i++]+'<br>';}
    else{clearInterval(iv);setTimeout(()=>$('boot').classList.add('done'),350);}
  },110);
  $('boot').addEventListener('click',()=>{clearInterval(iv);$('boot').classList.add('done');});
}

/* ---------------- go ---------------- */
function onResize(){sizeCanvas(chart);sizeCanvas(netc);}
window.addEventListener('resize',onResize);
onResize();
reset();
setSpeed(4);
boot();
requestAnimationFrame(frame);

/* ---------------- test hooks ---------------- */
window.__FLOCK={
  sim,replay,
  step(n){for(let i=0;i<n;i++)tick(1/60);},
  setSpeed,reset,
  state(){return {gen:sim.gen,alive:sim.alive,mode:sim.mode,histLen:sim.history.length,
    best:sim.bestEver,bestScore:sim.bestEverScore,
    lastGen:sim.history[sim.history.length-1]||null};},
};
})();
