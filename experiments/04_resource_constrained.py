# Extracted from validated research notebook; run from repository root.
# Scenario renamed only; ordering and numeric parameters unchanged.

# %% cell 1
import os, json, zipfile, time
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from IPython.display import display
SEED=20261025; T=80; N=5; K=5
REQ=np.array([78.,67.,55.,43.,32.]); VALUE=np.array([10.,8.,6.,4.,2.])
TASK_XY=np.array([[0.10,.10],[.85,.15],[.80,.85],[.15,.80],[.50,.50]])
BASE_XY=np.array([[.05,.05],[.95,.05],[.95,.95],[.05,.95],[.50,.10]])
BATTERY_INITIAL=120.; ENERGY_PER_DISTANCE=18.; ENERGY_PER_TASK=0.85
# 2D unit square is an abstract mission area, NOT metres; energy units are synthetic.
SCENARIOS={'low_noise':(5.,0.),'high_noise':(22.,0.),'biased_optimism':(12.,12.),'severe_shift':(22.,-20.)}
POLICIES={'continuous':(0.,0.,0.),'no_margin_stable':(0.,1.4,10.),'full':(.25,1.4,10.)}
REPS=30; N_BOOT=1000
OUT='/content/UAV_Resource_Constraint_Audit_Results' if os.path.isdir('/content') else 'UAV_Resource_Constraint_Audit_Results'
os.makedirs(OUT,exist_ok=True)
print('Output directory:',OUT)


# %% cell 2
def make_world(seed,noise_sd,bias):
    rng=np.random.default_rng(seed)
    cap=np.empty((T,N));cap[0]=rng.uniform(68,97,N)
    onset=rng.integers(12,52,N); rate=rng.uniform(.35,1.2,N)
    for t in range(1,T):
        cap[t]=np.clip(cap[t-1]+rng.normal(0,.65,N)-(t>=onset)*rate,0,100)
    obs=np.clip(cap+rng.normal(0,noise_sd,cap.shape)+bias,0,100)
    sigma=np.full_like(obs,noise_sd if bias==0 else noise_sd*.5)
    return cap,obs,sigma

def allocate_resource(score,sigma,previous,positions,battery,margin,penalty,bonus):
    distance=np.linalg.norm(positions[:,None,:]-TASK_XY[None,:,:],axis=2)
    energy=ENERGY_PER_DISTANCE*distance+ENERGY_PER_TASK
    eligible=(score[:,None]-margin*sigma[:,None]>=REQ[None,:]) & (battery[:,None]>=energy)
    values=VALUE.copy();values[0]+=bonus
    payoff=np.where(eligible,values[None,:]+.001*score[:,None]-.08*energy,-1e5)
    if penalty:
        for j in range(K):
            if previous[j]>=0: payoff[:,j]-=np.where(np.arange(N)==previous[j],0.,penalty)
    payoff=np.vstack([payoff,np.zeros((K,K))])
    rr,cc=linear_sum_assignment(-payoff)
    chosen=np.full(K,-1,int)
    for i,j in zip(rr,cc):
        if i<N and eligible[i,j]: chosen[j]=i
    return chosen,energy

def trace(world,params,perfect=False):
    cap,obs,sigma=world
    pos=BASE_XY.copy(); battery=np.full(N,BATTERY_INITIAL)
    previous=np.full(K,-1,int)
    utility=unsafe=deferred=critical=switches=0
    spent=0.; stranded=0; executed=0
    for t in range(T):
        score=cap[t] if perfect else obs[t]
        uncert=np.zeros(N) if perfect else sigma[t]
        chosen,energy=allocate_resource(score,uncert,previous,pos,battery,*params)
        assigned=chosen>=0; safe=np.zeros(K,bool)
        if assigned.any():safe[assigned]=cap[t,chosen[assigned]]>=REQ[assigned]
        utility+=VALUE[safe].sum()
        unsafe+=int((assigned&~safe).sum())
        deferred+=int((~assigned).sum()); critical+=int(safe[0])
        switches+=int(((chosen!=previous)&assigned&(previous>=0)).sum())
        for j in np.where(assigned)[0]:
            i=chosen[j]; e=energy[i,j]
            assert battery[i]+1e-8>=e
            battery[i]-=e; spent+=e; pos[i]=TASK_XY[j];executed+=1
        stranded+=int((battery<ENERGY_PER_TASK).sum())
        previous=chosen
    return dict(safe_utility_ratio=float(utility/(T*VALUE.sum())),
        unsafe_per_opportunity=unsafe/(T*K),deferral_rate=deferred/(T*K),
        critical_safe_rate=critical/T,switches=switches,
        energy_spent=spent,mean_final_battery=float(battery.mean()),
        executed=executed,near_empty_uav_steps=stranded)

w=make_world(SEED,12,0)
for params in POLICIES.values():
    a=trace(w,params);b=trace(w,params,perfect=True)
    assert a['safe_utility_ratio']<=1 and b['safe_utility_ratio']<=1
    assert b['unsafe_per_opportunity']==0
print('Sanity checks PASS')


# %% cell 4
rows=[]
t0=time.time()
for sidx,(scenario,(sd,bias)) in enumerate(SCENARIOS.items()):
    for rep in range(REPS):
        seed=SEED+1400000+10000*sidx+rep
        world=make_world(seed,sd,bias)
        mae=float(np.abs(world[1]-world[0]).mean())
        signed_bias=float((world[1]-world[0]).mean())
        for name,params in POLICIES.items():
            actual=trace(world,params,False)
            perfect=trace(world,params,True)
            row=dict(scenario=scenario,rep=rep,seed=seed,policy=name,estimation_mae=mae,
                     estimation_bias=signed_bias)
            row.update(actual)
            row.update({'perfect_'+k:v for k,v in perfect.items()})
            row['signed_information_utility_gap']=perfect['safe_utility_ratio']-actual['safe_utility_ratio']
            rows.append(row)
runs=pd.DataFrame(rows)
assert len(runs)==len(SCENARIOS)*REPS*len(POLICIES)
runs.to_csv(os.path.join(OUT,'resource_audit_runs.csv'),index=False)
print('Finished',len(runs),'policy runs in',round(time.time()-t0,1),'seconds')


# %% cell 5
METRICS=['safe_utility_ratio','unsafe_per_opportunity','deferral_rate','critical_safe_rate',
         'switches','energy_spent','mean_final_battery','signed_information_utility_gap']
summary=runs.groupby(['scenario','policy'],as_index=False)[METRICS].agg('mean')
summary.to_csv(os.path.join(OUT,'resource_audit_summary.csv'),index=False)
display(summary.round(4))
rng=np.random.default_rng(SEED+14)
paired=[]
for scenario,g in runs.groupby('scenario'):
    wide=g.set_index(['rep','policy'])
    for left,right in [('full','no_margin_stable'),('full','continuous')]:
        for metric in METRICS:
            a=wide.xs(left,level='policy')[metric].sort_index().to_numpy()
            b=wide.xs(right,level='policy')[metric].sort_index().to_numpy()
            d=a-b
            boot=np.mean(d[rng.integers(0,len(d),size=(N_BOOT,len(d)))],axis=1)
            paired.append(dict(scenario=scenario,comparison=f'{left} minus {right}',metric=metric,
                               mean_difference=float(d.mean()),ci_low=float(np.percentile(boot,2.5)),
                               ci_high=float(np.percentile(boot,97.5))))
paired=pd.DataFrame(paired)
paired.to_csv(os.path.join(OUT,'resource_audit_paired.csv'),index=False)
display(paired[paired.metric.isin(['safe_utility_ratio','unsafe_per_opportunity','signed_information_utility_gap'])].round(4))


# %% cell 6
assoc=[]
for scenario,g in runs[runs.policy=='full'].groupby('scenario'):
    for metric in ['signed_information_utility_gap','unsafe_per_opportunity','deferral_rate']:
        r,p=spearmanr(g.estimation_mae,g[metric])
        assoc.append(dict(scenario=scenario,metric=metric,rho=float(r),p_value_descriptive=float(p),n_worlds=len(g)))
assoc=pd.DataFrame(assoc)
assoc.to_csv(os.path.join(OUT,'resource_audit_within_scenario.csv'),index=False)
display(assoc.round(4))
fig,ax=plt.subplots(figsize=(10,5))
for name,g in runs.groupby('policy'):
    x=g.groupby('scenario')['safe_utility_ratio'].mean().reindex(SCENARIOS)
    ax.plot(range(len(x)),x.values,marker='o',label=name)
ax.set_xticks(range(len(SCENARIOS)),list(SCENARIOS),rotation=15)
ax.set_ylabel('Safe utility / theoretical task value')
ax.set_title('Resource-constrained task allocation (synthetic simulation)')
ax.legend();fig.tight_layout()
fig.savefig(os.path.join(OUT,'resource_audit_utility.png'),dpi=160)
plt.show()


# %% cell 7
# Strict methodological checks and stop/go flags
# This verifies that the resource constraint is active and that the perfect-information
# comparator no longer automatically reaches 100% safe utility.
perfect_cols=runs['perfect_safe_utility_ratio']
active=bool((runs.mean_final_battery<BATTERY_INITIAL-1).all())
nontrivial=bool((perfect_cols<.999999).mean()>.5)
gap_nonzero=bool((runs.signed_information_utility_gap.abs()>1e-9).mean()>.5)
# No 'publication GO' can be inferred automatically; these are only model sanity checks.
verdict={'resource_constraint_active':active,'perfect_information_not_trivial':nontrivial,
         'nonzero_information_gaps_in_majority':gap_nonzero,
         'n_worlds':len(SCENARIOS)*REPS,'n_policy_runs':len(runs),
         'model_limitations':['Synthetic energy units and positions','No flight dynamics',
          'No empirical health-to-capability mapping','No global dynamic oracle',
          'Perfect and imperfect policies follow different endogenous battery paths'],
         'next_step':'Review evidence and stop model development; write with bounded claims or stop project.'}
with open(os.path.join(OUT,'resource_audit_verdict.json'),'w') as f:json.dump(verdict,f,indent=2)
print(json.dumps(verdict,indent=2))
archive='/content/UAV_Resource_Constraint_Audit_Results.zip' if os.path.isdir('/content') else 'UAV_Resource_Constraint_Audit_Results.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for name in sorted(os.listdir(OUT)):z.write(os.path.join(OUT,name),arcname=name)
print('ZIP:',archive)
if os.path.isdir('/content'):
    from google.colab import files
    files.download(archive)

