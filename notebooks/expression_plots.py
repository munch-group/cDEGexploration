import pandas as pd
import numpy as np
import random
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from functools import cache
#import anndata as ad
import scanpy as sc

# adata_file = "../data/adata_susbet_kasper.h5ad"
# adata = sc.read(adata_file)

adata = None

def load_data(adata_file):
    global adata
    adata = sc.read(adata_file, cache=True)

@cache
def _cell_categories_data(layer):
    clusters = ['Spermatogonia', 'Leptotene/Zygotene', 'Pachytene', 'Spermatids']    
    try:
        # df = adata[adata.obs.SPECIES == "Human"].to_df(layer=layer)
        df = adata.to_df(layer=layer)        
    except NameError:
        print('''Use 'load_data(<data_file>)' to load data before plotting''')
        return
    df['cluster'] = adata.obs.cluster
    df.set_index('cluster', inplace=True)
    df.columns.name = 'gene'
    return df
    
def cell_categories(genes, layer=None, markersize=4, 
                    #errorbar=False
                    errorbar=False,
                    ax=None):
    # layer is 'norm_sct' or 'raw_counts'
    if layer is None:
        return _double_cell_categories(genes, errorbar=errorbar, markersize=markersize, ci=ci)
    axes = ax
    
    df = _cell_categories_data(layer)
    
    genes = pd.Series(genes)
    has_data = genes.isin(df.columns)
    missing = genes[~has_data]
    if missing.size:
        print("missing:", missing.tolist())
    genes = genes[has_data]
    plot_df = df.loc[:, genes].stack().to_frame('expression').reset_index()
    if len(plot_df.index):    
        # plt.figure(figsize=(10,5))
        ax = sns.pointplot(data=plot_df, x='cluster', y='expression', hue='gene', order=clusters, 
                           errorbar=('ci', 95) if errorbar else None, 
                           ax=axes, err_kws={'linewidth': 1})
        plt.setp(ax.collections, sizes=[markersize])
        ax.xaxis.set_major_formatter('{x:.2f}')
        ax.xaxis.set_tick_params(rotation=90)  
        ax.set_ylabel(layer)
        ax.set_xlabel('Pseudo-time')
        ax.legend(frameon=False, borderaxespad=0.)
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))       
    del df
    gc.collect()
    sns.despine()
    return ax

def _double_cell_categories(genes, markersize=4, errorbar=False):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
    cell_categories(genes, 'raw_counts', markersize, errorbar=errorbar, ax=ax1)
    ax1.legend().set_visible(False)    
    cell_categories(genes, 'norm_ssct', markersize, errorbar=errorbar, ax=ax2)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc=2, frameon=False, borderaxespad=0.)    
    return (ax1, ax2)

@cache
def _trajectory_data(layer):
    try:
        # df = adata[adata.obs.SPECIES == "Human"].to_df(layer=layer)
        df = adata.to_df(layer=layer)
    except NameError:
        print('''Use 'load_data(<data_file>)' to load data before plotting''')
        return    
    df['ptime'] = pd.cut(adata.obs.Pseudotime_scaled, include_lowest=True, bins=30)
    df.set_index('ptime', inplace=True)
    df.columns.name = 'gene'
    return df
    
def trajectory(genes, layer=None, markersize=4, errorbar=False, min_max_norm=False, ax=None):
    if layer is None:
        return _double_trajectory(genes, errorbar=errorbar, markersize=markersize)
    axes = ax

    df = _trajectory_data(layer)
    # print(df)

    genes = pd.Series(genes)
    has_data = genes.isin(df.columns)
    missing = genes[~has_data]
    if missing.size:
        print("missing:", missing.tolist())
    genes = genes[has_data]
    plot_df = df[genes].stack().to_frame('expression').reset_index()

    if min_max_norm:
        plot_df = plot_df.groupby(['ptime', 'gene'], observed=True).mean().reset_index()
        plot_df['expression'] = plot_df.groupby('gene')[['expression']].transform(lambda x: (x - x.min())/(x.max()-x.min()) ).expression
                       
    if len(plot_df.index):
        ax = sns.pointplot(data=plot_df, x='ptime', y='expression', hue='gene', markersize=markersize, linewidth=1, errorbar=('ci', 95) if errorbar else None, err_kws={'linewidth': 1}, ax=axes)
        plt.setp(ax.collections, sizes=[markersize])
        ax.xaxis.set_major_formatter('{x:.2f}')
        ax.xaxis.set_tick_params(rotation=90)        
        ax.axvline(6.5, color='grey', linestyle='dashed')
        ax.axvline(12, color='grey', linestyle='dashed')
        ax.axvline(14.5, color='grey', linestyle='dashed')
        ax.set_ylabel(layer)
        ax.set_xlabel('Pseudo-time')
        ax.legend(frameon=False, borderaxespad=0.)
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1)) 
    del df
    gc.collect()
    sns.despine()
    return ax

def _double_trajectory(genes, markersize=4, errorbar=False):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
    trajectory(genes, 'raw_counts', markersize, errorbar=errorbar, ax=ax1)
    ax1.legend().set_visible(False)
    trajectory(genes, 'norm_sct', markersize, errorbar=errorbar, ax=ax2)
    ax2.legend(frameon=False, borderaxespad=0.)
    sns.move_legend(ax2, "upper left", bbox_to_anchor=(1, 1))     
    return (ax1, ax2)

@cache
def _split_trajectory_data(layer):
    try:
        # df = adata[adata.obs.SPECIES == "Human"].to_df(layer=layer)
        df = adata.to_df(layer=layer)
    except NameError:
        print('''Use 'load_data(<data_file>)' to load data before plotting''')
        return    
    # df['ptime'] = pd.cut(adata.obs.Pseudotime_scaled, include_lowest=True, bins=30)
    # df = df.merge(adata.obs["cell_class"].to_frame('cell_class'), how='left', left_index=True, right_index=True)
    # df.set_index(['ptime', 'cell_class'], inplace=True)    
    df['ptime'] = pd.cut(adata.obs.Pseudotime_scaled, include_lowest=True, bins=30)
    df = df.merge(adata.obs["cell_class"].to_frame('cell_class'), how='left', left_index=True, right_index=True)
    df.set_index(['ptime', 'cell_class'], inplace=True)
    return df

def split_trajectory(gene, layer=None, markersize=4, errorbar=False, ax=None):
    assert type(gene) is str
    genes = [gene]    
    if layer is None:
        return _double_split_trajectory(gene, errorbar=errorbar, markersize=markersize)
    axes = ax

    df = _split_trajectory_data(layer)
    
    #df.columns.name = 'gene'
    genes = pd.Series(genes)
    has_data = genes.isin(df.columns)
    missing = genes[~has_data]
    if missing.size:
        print("missing:", missing.tolist())
    genes = genes[has_data]
    df = df[genes] 
    df = df.reset_index()
    plot_df = df.melt(id_vars=['ptime', 'cell_class'], value_name='expression')#.reset_index()
    if len(plot_df.index):
        ax = sns.pointplot(data=plot_df, x='ptime', y='expression', 
                           hue='cell_class', #linestyles=['--', '-', '-'],
                           errorbar=('ci', 95) if errorbar else None, ax=axes, markersize=markersize, err_kws={'linewidth': 1})
        plt.setp(ax.collections, sizes=[markersize])
        ax.xaxis.set_major_formatter('{x:.2f}')
        ax.xaxis.set_tick_params(rotation=90)        
        ax.axvline(6.5, color='grey', linestyle='dashed')
        ax.axvline(12, color='grey', linestyle='dashed')
        ax.axvline(14.5, color='grey', linestyle='dashed')
        ax.set_ylabel(layer)
        ax.set_xlabel('Pseudo-time')
        ax.legend(frameon=False, borderaxespad=0.)
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1)) 

    del df
    gc.collect()
    sns.despine()
    return ax
    

def _double_split_trajectory(gene, markersize=4, errorbar=False):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
    split_trajectory(gene, 'raw_counts', markersize, errorbar=errorbar, ax=ax1)
    ax1.legend().set_visible(False)
    split_trajectory(gene, 'norm_sct', markersize, errorbar=errorbar, ax=ax2)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc=2, frameon=False, borderaxespad=0.)
    return (ax1, ax2)    

@cache
def _x_ratio_trajectory_data(layer):
    try:
        # df = adata[adata.obs.SPECIES == "Human"].to_df(layer=layer)
        df = adata.to_df(layer=layer)
    except NameError:
        print('''Use 'load_data(<data_file>)' to load data before plotting''')
        return
    # df['ptime'] = pd.cut(adata.obs.Pseudotime_scaled, include_lowest=True, bins=8)
    df['ptime'] = adata.obs.Pseudotime_scaled
    df = df.merge(adata.obs["cell_class"].to_frame('cell_class'), how='left', left_index=True, right_index=True)
    df = df.loc[df.cell_class.isin(['X', 'Y'])]
    df['ptime'] = pd.qcut(df.ptime, q=10)
    df.set_index(['ptime', 'cell_class'], inplace=True)
    return df

def x_ratio_trajectory(genes, layer=None, markersize=4, errorbar=False, ax=None, normalize=False):
    if layer is None:
        return _double_x_ratio_trajectory(genes, errorbar=errorbar, markersize=markersize)
    axes = ax

    df = _x_ratio_trajectory_data(layer)
    
    genes = pd.Series(genes)
    has_data = genes.isin(df.columns)
    missing = genes[~has_data]
    if missing.size:
        print("missing:", missing.tolist())
    genes = genes[has_data]
    df = df[genes] 
    df = df.reset_index()

    def x_ratio_resamping(df):
        lst = []
        x_sr = df.loc[df.cell_class == 'X', 'expression']
        y_sr = df.loc[df.cell_class == 'Y', 'expression']
        # print('min max', y_sr.min(), y_sr.max())
        # print('\n'.join(map(str, sorted(x_sr.tolist()))))
        # print()
        # print('\n'.join(map(str, sorted(y_sr.tolist()))))
        for _ in range(100):
            x = x_sr.sample(x_sr.index.size, replace=True).mean()
            y = y_sr.sample(y_sr.index.size, replace=True).mean()
            # print('means', x, y)
            if x + y:
                lst.append(x / (x + y))
            else:
                lst.append(np.nan)
        # print()
        return pd.DataFrame(dict(x_ratio=lst))

    def x_ratio(df):
        x = df.loc[df.cell_class == 'X', 'expression'].mean() 
        y = df.loc[df.cell_class == 'Y', 'expression'].mean()
        if x + y:
            val = x / (x + y)
        else:
            val = np.nan
        return pd.DataFrame(dict(x_ratio=[val]))

    assert errorbar in [True, False]
    # assert bool(errorbar) == bool(normalize)
    
    if errorbar:
        df = (df.melt(id_vars=['ptime', 'cell_class'], value_name='expression', var_name='gene')
         .groupby(['ptime', 'gene'])
         .apply(x_ratio_resamping)
         .reset_index()
        )
    else:
        df = (df.melt(id_vars=['ptime', 'cell_class'], value_name='expression', var_name='gene')
         .groupby(['ptime', 'gene'])
         .apply(x_ratio)
         .reset_index()
        )
        # if normalize:
        #     df['x_ratio'] = df.groupby(['gene']).x_ratio.transform(lambda x: x / x.sum())

    # print(df)

    df['ptime'] = df['ptime'].cat.remove_unused_categories()

    # def std(x):
    #     return x.quantile(0.25), x.quantile(0.75)
        
    
    plot_df = df#.melt(id_vars=['ptime', 'cell_class'], value_name='expression')#.reset_index()
    if len(plot_df.index):
        ax = sns.pointplot(data=plot_df, x='ptime', y='x_ratio', 
                           hue='gene', #linestyles=['--', '-', '-'],
                           # errorbar=std,
                           # errorbar=errorbar,
                           err_kws={'linewidth': 1},
                           ax=axes, markersize=markersize)
        plt.setp(ax.collections, sizes=[markersize])
        ax.xaxis.set_major_formatter('{x:.2f}')
        ax.xaxis.set_tick_params(rotation=90)        
        # ax.axvline(6.5, color='grey', linestyle='dashed')
        # ax.axvline(12, color='grey', linestyle='dashed')
        # ax.axvline(14.5, color='grey', linestyle='dashed')
        ax.set_ylabel(layer)
        ax.set_xlabel('Pseudo-time')
        ax.legend(frameon=False, borderaxespad=0.)
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1)) 
        # ax.set_xlim(left=14.5)

    del df
    gc.collect()
    sns.despine()


    
def _double_x_ratio_trajectory(genes, errorbar=False, markersize=4, normalize=False):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
    x_ratio_trajectory(genes, 'raw_counts', markersize, errorbar=errorbar, normalize=normalize, ax=ax1)
    ax1.legend().set_visible(False)
    x_ratio_trajectory(genes, 'norm_sct', markersize, errorbar=errorbar, normalize=normalize, ax=ax2)
    ax2.legend(frameon=False, borderaxespad=0.)
    sns.move_legend(ax2, "upper left", bbox_to_anchor=(1, 1)) 
    return (ax1, ax2)     


def facet_plot(genes, plot_fun, *args, sharey=False, ncols=3, **kwargs):
    nr_genes = len(genes)
    nr_rows = nr_genes // ncols + int(nr_genes % ncols)
    fig, axes = plt.subplots(nr_rows, ncols, figsize=(10, min(50, 2 + 1.6*nr_rows)), sharey=sharey, sharex=True)
    axes_flat = axes.flatten()
    for gene, ax in zip(genes, axes_flat):
        plot_fun(gene, *args, **kwargs, ax=ax, )
        ax.set_title(gene, loc='right')
        try:
            ax.get_legend().remove()
        except AttributeError:
            # if the gene cannot be found
            pass
    if plot_fun == split_trajectory:
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.05, 0.5), frameon=False)        
    for ax in axes_flat[nr_genes:]:
        ax.remove()
    plt.tight_layout()
