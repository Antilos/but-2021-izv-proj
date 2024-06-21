import logging
import os, re, io, requests, datetime, operator
import gzip, pickle, csv, zipfile, argparse
from bs4 import BeautifulSoup
import numpy as np
import matplotlib.pyplot as plt

from download import DataDownloader

logging.basicConfig(level=logging.INFO)

def plot_stat(data_source, fig_location = None, show_figure = False):
    names, data = data_source

    regions = np.unique(data[names.index('region')])
    yearArr = np.array(list(map(lambda x : getattr(x,"year"), data[names.index('p2a')])))
    years = np.unique(yearArr) #get years

    fig, ax = plt.subplots(years.shape[0], sharex=False, sharey=True)
    #force ax to be an array
    if not isinstance(ax, np.ndarray):
        ax = np.array(ax)

    #fig size
    fig.set_size_inches(w=8.2, h=20)
    fig.set_tight_layout({"h_pad":1.5})

    #fig title
    fig.suptitle("Accident counts in Czech Republic", y=1)

    for i, year in enumerate(years):
        logging.debug(regions)
        #graph title
        ax[i].set_title(str(year))

        yearMask = np.nonzero(yearArr == year)
        accidentCounts = list()
        for region in regions:
            accidentCounts.append(data[names.index('region')][np.nonzero(data[names.index('region')][yearMask] == region)].shape[0])
        
        #plot accidents
        rects = ax[i].bar(regions, accidentCounts)
        #rects = ax[i].bar(*zip(*sorted(zip(regions, accidentCounts), reverse=True, key=operator.itemgetter(1))))

        #regions ordered by accident count
        #orderedRegions, _ = zip(*sorted(zip(regions, accidentCounts), reverse=True, key=operator.itemgetter(1)))
        orderedAccidentCounts = tuple(zip(*sorted(zip(regions, accidentCounts), reverse=True, key=operator.itemgetter(1))))

        #Anotate bars with accident position in country and the exact accident count
        for rect, region in zip(rects, regions):
            regionIndex = orderedAccidentCounts[0].index(region)+1 #add 1 to text because indexes start at 0
            accidentCount = orderedAccidentCounts[1][regionIndex-1]
            #position
            ax[i].annotate(regionIndex, xy=(rect.get_x()+(rect.get_width()/2), rect.get_y()+rect.get_height()))

            #accident count
            ax[i].annotate(accidentCount, xy=(rect.get_x(), rect.get_y()+(rect.get_height()/2)), color='white')

        ax[i].set_xlabel("Regions")
        ax[i].set_ylabel("Accident Count")
        ax[i].legend(["Accident Count"])

    #save the graph
    if fig_location:
        if not os.path.isdir(os.path.dirname(fig_location)):
            os.makedirs(os.path.dirname(fig_location))
        fig.savefig(fig_location)
    
    #show thegraph
    if show_figure:
        plt.show()
    else:
        return



if __name__ == "__main__":
    #Argument parsing
    argParser = argparse.ArgumentParser()

    argParser.add_argument("--fig_location", default=None, required=False)
    argParser.add_argument("--show_figure", default=False, action='store_true')

    args = argParser.parse_args()

    logging.debug(args.show_figure)
    #
    plot_stat(DataDownloader().get_list(), fig_location=args.fig_location, show_figure=args.show_figure)