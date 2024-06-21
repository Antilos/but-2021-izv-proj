#!/usr/bin/python3.8
# coding=utf-8
import pandas as pd
import geopandas
import matplotlib.pyplot as plt
import contextily as ctx
import sklearn.cluster
import numpy as np
# muzeze pridat vlastni knihovny

def categorize(df: pd.DataFrame) -> pd.DataFrame:
    """Categorizis certain collumns and adds a date column"""
    #add date column
    df['date'] = pd.to_datetime(df['p2a'])

    ##Convert to categories##
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

    return df

def make_geo(df: pd.DataFrame) -> geopandas.GeoDataFrame:
    """ Konvertovani dataframe do geopandas.GeoDataFrame se spravnym kodovani"""
    #drop rows with regions we are not interested in
    df = df[df["region"]=="JHM"]

    #drop rows with missing coordinates
    df = df.dropna(how='any', subset=['d','e'])


    #categorize values
    df = categorize(df)

    #convert to GeoDataFrame and set projection to Krovak East-North (EPSG:5514)
    gdf = geopandas.GeoDataFrame(df, crs="EPSG:5514", geometry=geopandas.points_from_xy(df.d, df.e))

    return gdf

def plot_geo(gdf: geopandas.GeoDataFrame, fig_location: str = None,
             show_figure: bool = False):
    """ Vykresleni grafu s dvemi podgrafy podle lokality nehody """
    fig, ax = plt.subplots(1, 2, figsize=(8,6))

    #set ax titles
    ax[0].set_title("JHM : Nehody v obci")
    ax[1].set_title("JHM : Nehody mimo obce")


    #convert to web mercator (EPSG:3857) for presentation
    gdf = gdf.to_crs('EPSG:3857')

    # #limit axes
    padding = 100
    xmin = gdf['geometry'].x.min()
    xmax = gdf['geometry'].x.max()
    ymin = gdf['geometry'].y.min()
    ymax = gdf['geometry'].y.max()

    ax[0].set_xlim(xmin-padding, xmax+padding)
    ax[0].set_ylim(ymin-padding, ymax+padding)
    ax[1].set_xlim(xmin-padding, xmax+padding)
    ax[1].set_ylim(ymin-padding, ymax+padding)

    #Accidents inside city limits
    gdf[gdf["p5a"]==1].plot(ax=ax[0], markersize=2, label="Nehody v obci")

    #Accidents outside city limits
    gdf[gdf["p5a"]==2].plot(ax=ax[1], markersize=2, label="Nehody mimo obce")

    #add base maps
    ctx.add_basemap(ax[0], crs=gdf.crs, source=ctx.providers.Stamen.TonerLite)
    ctx.add_basemap(ax[1], crs=gdf.crs, source=ctx.providers.Stamen.TonerLite)

    plt.legend()

    if fig_location:
        plt.savefig(fig_location)
    
    if show_figure:
        plt.show()


def plot_cluster(gdf: geopandas.GeoDataFrame, fig_location: str = None,
                 show_figure: bool = False):
    """ Vykresleni grafu s lokalitou vsech nehod v kraji shlukovanych do clusteru """
    #training data
    X = np.dstack([gdf.geometry.x, gdf.geometry.y]).reshape(-1,2)

    #train
    n_clusters = 50
    model = sklearn.cluster.MiniBatchKMeans(n_clusters=n_clusters).fit(X)

    #add cluster labels to data
    gdf["cluster"] = model.labels_

    #cluster
    gdf_c = gdf.dissolve(by="cluster", aggfunc={"p1":"count"}).rename(columns={"p1":"count"})
    #get cluster centers
    centers = geopandas.GeoDataFrame(geometry=geopandas.points_from_xy(model.cluster_centers_[:,0], model.cluster_centers_[:,1]), crs='EPSG:5514')
    gdf_c = gdf_c.merge(centers, left_on="cluster", right_index=True).set_geometry("geometry_y", crs='EPSG:5514')
    
    #convert to web mercator (EPSG:3857) for presentation
    gdf_c = gdf_c.to_crs('EPSG:3857')
    gdf = gdf.to_crs('EPSG:3857')

    #plot
    fig, ax = plt.subplots(1,1,figsize=(8,6))

    #plot points
    gdf.plot(ax=ax, markersize=0.1, alpha=0.7, color="tab:gray")
    #plot clusters
    gdf_c.plot(ax=ax, alpha=0.5, markersize=gdf_c["count"]/5, column="count", legend=True)

    #add base map
    ctx.add_basemap(ax, crs=gdf_c.crs, source=ctx.providers.Stamen.TonerLite)

    if fig_location:
        plt.savefig(fig_location)
    
    if show_figure:
        plt.show()


if __name__ == "__main__":
    # zde muzete delat libovolne modifikace
    gdf = make_geo(pd.read_pickle("accidents.pkl"))
    #plot_geo(gdf, "geo1.png", True)
    plot_cluster(gdf, "geo2.png", True)

