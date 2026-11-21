from pathlib import Path
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score,davies_bouldin_score,adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
p=Path(__file__).parent
rfm=pd.read_csv(p/'checkpoint_rfm_customer_features.csv')
X=StandardScaler().fit_transform(np.log1p(rfm[['Recency','Frequency','Monetary']].clip(lower=0)))
base=KMeans(n_clusters=2,random_state=42,n_init=10).fit_predict(X)
rfm['Cluster']=base
# Original-variable chart.
fig,ax=plt.subplots(figsize=(9,6))
for cluster,color in [(0,'#1779a7'),(1,'#d87d37')]:
 d=rfm[rfm.Cluster==cluster]
 ax.scatter(d.Recency,d.Monetary,alpha=.35,s=14,c=color,label=f'Cluster {cluster} (n={len(d):,})',rasterized=True)
ax.set(yscale='log',xlabel='Recency (days since last purchase)',ylabel='Total positive spending (£, log scale)',title='Customer Segments in Original Purchasing Variables')
ax.legend();fig.tight_layout();fig.savefig(p/'final_customer_clusters_rfm.png',dpi=190);plt.close(fig)
pr=rfm.groupby('Cluster').agg(Customers=('Recency','size'),MedianRecency=('Recency','median'),MedianFrequency=('Frequency','median'),MedianMonetary=('Monetary','median'),MedianOrderValue=('AverageOrderValue','median'),MedianBasketUnits=('BasketSizeUnits','median'),MedianCancellationLines=('CancellationLineCount','median'))
pr.to_csv(p/'final_cluster_profile.csv')
fig,axes=plt.subplots(1,3,figsize=(13.5,4))
for ax,col,title in zip(axes,['MedianRecency','MedianFrequency','MedianMonetary'],['Recency (days, lower is more recent)','Frequency (invoices)','Spending (£)']):
 vals=pr[col]
 ax.bar([f'Cluster {i}' for i in vals.index],vals.values,color=['#1779a7','#d87d37']);ax.set_title(title)
 for j,v in enumerate(vals):ax.text(j,v+max(vals)*.03,f'{v:,.2f}' if col=='MedianMonetary' else f'{v:,.0f}',ha='center',fontsize=10)
 ax.set_ylim(0,max(vals)*1.23)
fig.suptitle('Interpretable RFM Customer Profiles');fig.tight_layout();fig.savefig(p/'final_interpretable_rfm_profiles.png',dpi=200);plt.close(fig)
st=[]
for seed in [0,1,2,7,21,42,99]:
 trial=KMeans(n_clusters=2,random_state=seed,n_init=10).fit_predict(X)
 st.append({'Seed':seed,'ARI_vs_seed_42':adjusted_rand_score(base,trial)})
st=pd.DataFrame(st);st.to_csv(p/'final_seed_stability.csv',index=False)
fig,ax=plt.subplots(figsize=(7,3.5));ax.plot(st.Seed,st.ARI_vs_seed_42,marker='o');ax.set(xlabel='Random seed',ylabel='ARI vs seed 42',title='K-Means Random Initialization Stability',ylim=(-.03,1.05));fig.tight_layout();fig.savefig(p/'final_cluster_seed_stability.png',dpi=200);plt.close(fig)
candidates={'K-Means':base,'Agglomerative (Ward)':AgglomerativeClustering(n_clusters=2,linkage='ward').fit_predict(X),'Gaussian mixture':GaussianMixture(n_components=2,random_state=42,n_init=3).fit_predict(X)}
rows=[]
for method,labs in candidates.items():rows.append({'Method':method,'Clusters':len(np.unique(labs)),'SampledSilhouette':silhouette_score(X,labs,sample_size=1500,random_state=42),'DaviesBouldin':davies_bouldin_score(X,labs),'ARIvsKMeans':adjusted_rand_score(base,labs)})
comp=pd.DataFrame(rows);comp.to_csv(p/'final_method_comparison.csv',index=False)
fig,ax=plt.subplots(figsize=(8,4));ax.bar(comp.Method,comp.SampledSilhouette,color=['#1779a7','#798c9c','#d87d37']);ax.set(ylabel='Sampled silhouette (higher is better)',title='Comparison of Clustering Methods');ax.tick_params(axis='x',rotation=8);fig.tight_layout();fig.savefig(p/'final_method_comparison.png',dpi=200);plt.close(fig)
print('PROFILE\n',pr.to_string());print('\nSEED STABILITY\n',st.to_string(index=False));print('\nCOMPARISON\n',comp.to_string(index=False))
