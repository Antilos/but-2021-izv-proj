import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

if __name__ == "__main__":
    sns.set()

    ## get the data ##
    df = pd.read_pickle("accidents.pkl")

    #pick only personal vehicles
    df = df.query("p44 == 3 or p44 == 4")

    #categorize p44
    df["p44"] = df["p44"].astype('category')

    #categorize p12
    accidentCauseBins = [100,200,300,400,500,600,700]
    # accidentCauseLabels = [
    #     "nezaviněná řidičem",
    #     "nepřiměřená rychlost jízdy",
    #     "nesprávné předjíždění",
    #     "nedání přednosti v jízdě",
    #     "nesprávný způsob jízdy",
    #     "technická závada vozidla"
    # ]
    accidentCauseLabels = [
        100,
        200,
        300,
        400,
        500,
        600
    ]
    df["p12c"] = pd.cut(df["p12"], bins=accidentCauseBins, labels=accidentCauseLabels, include_lowest=True, right=False, ordered=False)

    df = df.rename(columns={'p44':'trailer'})
    df['trailer'] = df['trailer'].cat.rename_categories({3:False, 4:True})

    #get accident seriousness
    df["accident_seriousness"] = df.apply(lambda row : (row['p13a']+row['p13b']) > 0, axis=1)

    ## Determine wether accident seriousness and trailer use are related ##
    alpha = 0.05

    cross = pd.crosstab(index=df['trailer'], columns=df['accident_seriousness'])
    print(cross)

    stat, p, dof, expected = chi2_contingency(cross)
    print(f"{p=}\n{stat=}")

    # if p <= alpha:
    #     print('Dependent')
    # else:
    #     print('Independent')

    ## Plot accident seriousness for vehicles with and without trailer ##
    #normalize values (there are so many more accidents without trailer)
    df_group = df.groupby(['accident_seriousness','p12c','trailer'])['p1'].agg('count').unstack(level=2)

    #split by trailer use
    df_trailer = df_group[True].unstack(level=0)
    df_no_trailer = df_group[False].unstack(level=0)

    #compute proportions
    df_trailer[False] = df_trailer[False].map(lambda x : x/df_trailer[False].sum())
    df_trailer[True] = df_trailer[True].map(lambda x : x/df_trailer[True].sum())
    df_no_trailer[False] = df_no_trailer[False].map(lambda x : x/df_no_trailer[False].sum())
    df_no_trailer[True] = df_no_trailer[True].map(lambda x : x/df_no_trailer[True].sum())

    #melt and concat the tables so we end up with long form data
    df_trailer_unstacked = df_trailer.unstack().reset_index().rename(columns={0:'prop'})
    df_no_trailer_unstacked = df_no_trailer.unstack().reset_index().rename(columns={0:'prop'})
    df_trailer_unstacked['trailer'] = True
    df_no_trailer_unstacked['trailer'] = False
    df_c = pd.concat([df_trailer_unstacked, df_no_trailer_unstacked])

    #plot
    sns.catplot(x='p12c', y='prop', hue='accident_seriousness', col='trailer', data=df_c, kind='bar')

    plt.savefig('fig.eps')

    ## Make some observations about accident causes ##

    