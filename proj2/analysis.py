#!/usr/bin/env python3.8
# coding=utf-8

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os
# muzete pridat libovolnou zakladni knihovnu ci knihovnu predstavenou na prednaskach
# dalsi knihovny pak na dotaz
import gzip, pickle
from pympler import asizeof

# Ukol 1: nacteni dat
def get_dataframe(filename: str, verbose: bool = False) -> pd.DataFrame:
    orig_size = 0
    new_size = 0
    #read pickled data
    df = pd.read_pickle(filename)
    if verbose:
        orig_size = asizeof.asizeof(df)/1_048_576

    #add date column
    df['date'] = pd.to_datetime(df['p2a'])

    #get columns to convert to categories
    # We don't want to categorize:
    #   p1(id), date and region
    #   float data
    #   p2a(date), p2b(time)
    #   p13a(deaths), p13b(heavy injuries), p13c (light injuries)
    #   p14(total material damage(in czk))
    #   p34(nubmer of vehicles)
    #   p47(vehicle make year)(maybe?)

    # We also don't want to automaticaly categorize:
    #   p12(accident cause) - we want to bin this by main causes
    #   p53(vehicle damage (in hundreds of czk)) - bin

    #get column names
    convertColsNames = list(df.select_dtypes(exclude='float32').columns)
    convertColsNames.remove('p1')
    convertColsNames.remove('date')
    convertColsNames.remove('region')
    #convertColsNames.remove('p2a')
    #convertColsNames.remove('p2b')
    convertColsNames.remove('p12')
    convertColsNames.remove('p13a')
    convertColsNames.remove('p13b')
    convertColsNames.remove('p13c')
    convertColsNames.remove('p14')
    #convertColsNames.remove('p34')
    convertColsNames.remove('p53')

    #Automatic categorization
    df[convertColsNames] = df[convertColsNames].astype('category')

    #Manual categorizations
    # accidentCauseBins = pd.IntervalIndex.from_tuples((100,200), (201,300), (301,400), (401,500), (501,600), (601,700), closed='both')
    accidentCauseBins = [100,200,300,400,500,600,700]
    accidentCauseLabels = [
        "nezaviněná řidičem",
        "nepřiměřená rychlost jízdy",
        "nesprávné předjíždění",
        "nedání přednosti v jízdě",
        "nesprávný způsob jízdy",
        "technická závada vozidla"
    ]
    df["p12"] = pd.cut(df["p12"], bins=accidentCauseBins, labels=accidentCauseLabels, include_lowest=True, right=False, ordered=False)

    vehicleDamageBins =  [0,50,200,500,1000,float("inf")]
    df["p53"] = pd.cut(df["p53"], bins=vehicleDamageBins, include_lowest=True)

    if verbose:
        new_size = asizeof.asizeof(df)/1_048_576
        print(f"{orig_size=:.2f} MB\n{new_size=:.2f} MB")

    return df

# Ukol 2: následky nehod v jednotlivých regionech
def plot_conseq(df: pd.DataFrame, fig_location: str = None,
                show_figure: bool = False):
    sns.set()
    sns.color_palette('tab10')

    #get regions
    # regions = list(df['region'].unique())

    fig, axes = plt.subplots(4, 1, sharex=False, sharey=False)
    fig.set_size_inches(w=8.2, h=20)
    fig.set_tight_layout({"h_pad":1})

    dfByRegion = df.groupby(['region'])

    #total accidents
    totalAccidents = dfByRegion['p1'].agg('count')
    #order regions by total accidents
    regions = list(totalAccidents.sort_values(ascending=False).keys())
    
    sns.barplot(ax=axes[3], x=totalAccidents.index, y=totalAccidents.values, order=regions)
    axes[3].set_title("Total Accidents")

    #deaths
    deaths = dfByRegion['p13a'].agg('sum')
    sns.barplot(ax=axes[0], x=deaths.index, y=deaths.values, order=regions)
    axes[0].set_title("Deaths by Region")

    #heavy injuries
    heavyInjuries = dfByRegion['p13b'].agg('sum')
    sns.barplot(ax=axes[1], x=heavyInjuries.index, y=heavyInjuries.values, order=regions)
    axes[1].set_title("Heavy Injuries by Region")

    #light injuries
    lightInjuries = dfByRegion['p13c'].agg('sum')
    sns.barplot(ax=axes[2], x=lightInjuries.index, y=lightInjuries.values, order=regions)
    axes[2].set_title("Light Injuries")

    if fig_location:
        plt.savefig(fig_location)

    if show_figure:
        plt.show()

    return


# Ukol3: příčina nehody a škoda
def plot_damage(df: pd.DataFrame, fig_location: str = None,
                show_figure: bool = False):
    sns.set()
    chosenRegions = ['PHA', 'HKK', 'JHM', 'PLK']
    colRenameDict = {
        "region":"Región",
        "p12":"Príčina Nehody",
    }

    #get binned damage
    dmg = (tmp:=df[["region","p53", "p12"]])[tmp.region.isin(chosenRegions)]

    dmg.rename(columns=colRenameDict, inplace=True)
    grid = sns.catplot(data=dmg, kind='count', x='p53', hue=colRenameDict['p12'], row=colRenameDict['region'], legend_out=True)
    grid.set(yscale="log", ylabel="Počet Nehod")

    if fig_location:
        plt.savefig(fig_location)

    if show_figure:
        plt.show()

    return

# Ukol 4: povrch vozovky
def plot_surface(df: pd.DataFrame, fig_location: str = None,
                 show_figure: bool = False):
    roadSurfaceDict = {
        0:"jiný stav povrchu vozovky v době nehody",
        1:"povrch suchý, neznečištěný",
        2:"povrch suchý, znečištěný",
        3:"povrch mokrý",
        4:"na vozovce je bláto",
        5:"na vozovce je náledí, ujetý sníh - posypané",
        6:"na vozovce je náledí, ujetý sníh - neposypané",
        7:"na vozovce je rozlitý olej, nafta apod.",
        8:"souvislá sněhová vrstva, rozbředlý sníh",
        9:"náhlá změna stavu vozovky"
    }
    sns.set()
    cross = pd.crosstab([df["region"],df["date"]],[df["p16"]])
    # cross.rename(roadSurfaceDic)

    chosenRegions = ['PHA', 'HKK', 'JHM', 'PLK']

    regDataset = []
    for region in chosenRegions:
        tmp = cross.xs(region, level=0).resample('M').sum()
        # tmp.rename(roadSurfaceDict)
        regDataset.append((region,tmp))

    fig, axes = plt.subplots(4)
    fig.set_size_inches(w=8.2, h=20)
    fig.tight_layout()

    for ax, (region, regData) in zip(axes, regDataset):
        sns.lineplot(data=regData, ax=ax, legend="full")
        ax.set_ylabel("Počet Nehod")
        ax.set_xlabel(region)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, roadSurfaceDict.values(), ncol=2, loc='upper center', borderaxespad=0.)
    # handles, labels = axes[0].get_legend_handles_labels()
    # plt.legend(handles, roadSurfaceDict.values(), bbox_to_anchor=(1, 1), loc=2, ncol=1)
    # fig.legend()

    # fig

    if fig_location:
        plt.savefig(fig_location, bbox_inches='tight')

    if show_figure:
        plt.show()

    return

if __name__ == "__main__":
    pass
    # zde je ukazka pouziti, tuto cast muzete modifikovat podle libosti
    # skript nebude pri testovani pousten primo, ale budou volany konkreni ¨
    # funkce.

    df = get_dataframe("accidents.pkl.gz", verbose=False)
    plot_conseq(df, fig_location="01_nasledky.png", show_figure=True)
    #plot_damage(df, "02_priciny.png", True)
    plot_surface(df, "03_stav.png", True)

    #print(df.info())
