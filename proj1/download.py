import os, re, io, requests, datetime
import gzip, pickle, csv, zipfile
from bs4 import BeautifulSoup
import numpy as np
import matplotlib.pyplot as plt


class UnexpectedDataFormatException(Exception):
    def __init__(self, msg):
        super.__init__(f"Unexpected data format: {msg}")

class DataDownloader:
    def __init__(self, url="https://ehw.fit.vutbr.cz/izv/", folder="data", cache_filename="data_{}.pkl.gz"):
        self._url = url
        self._folder = folder
        self._cache_filename = cache_filename

        #Dictionary for translating czech month names to their respective numbers
        self._monthDict = {
            "leden":"01",
            "únor":"02",
            "březen":"03",
            "duben":"04",
            "květen":"05",
            "červen":"06",
            "červenec":"07",
            "srpen":"08",
            "září":"09",
            "říjen":"10",
            "listopad":"11",
            "prosinec":"12",
        }

        #Dictionary for translating file names to their region acronyms
        self._file2regionDict = {
            "00.csv":"PHA",
            "01.csv":"STC",
            "02.csv":"JHC",
            "03.csv":"PLK",
            "04.csv":"ULK",
            "05.csv":"HKK",
            "06.csv":"JHM",
            "07.csv":"MSK",
            "14.csv":"OLK",
            "15.csv":"ZLK",
            "16.csv":"VYS",
            "17.csv":"PAK",
            "18.csv":"LBK",
            "19.csv":"KVK"
        }

        self._region2fileDict = {v:k for (k,v) in self._file2regionDict.items()}

        #region data cache
        self._regionCache = {k:None for k in self._region2fileDict.keys()}

        self._colNames = [
            "p1",
            "p36",
            "p37",
            "p2a",
            "weekday",
            "p2b",
            "p6",
            "p7",
            "p8",
            "p9",
            "p10",
            "p11",
            "p12",
            "p13a",
            "p13b",
            "p13c",
            "p14",
            "p15",
            "p16",
            "p17",
            "p18",
            "p19",
            "p20",
            "p21",
            "p22",
            "p23",
            "p24",
            "p27",
            "p28",
            "p34",
            "p35",
            "p39",
            "p44",
            "p45a",
            "p47",
            "p48a",
            "p49",
            "p50a",
            "p51",
            "p52",
            "p53",
            "p55a",
            "p57",
            "p58",
            "a",
            "b",
            "d",
            "e",
            "f",
            "g",
            "h",
            "i",
            "j",
            "k",
            "l",
            "n",
            "o",
            "p",
            "q",
            "r",
            "s",
            "t",
            "p5a"
        ]

        self._nonIntCols = {
            "p1":np.int64,
            # "p2a":np.datetime64,
            "p2a":datetime.date,
            "p2b":datetime.time,
            "p14":float,
            "a":'S100', #semantics unknown
            "b":'S100', #semantics unknown
            "d":float,#decimal comma
            "e":float,#decimal comma
            "f":'S100',#decimal comma #semantics unknown
            "g":'S100',#decimal comma #semantics unknown
            "h":'S100', #semantics unknown
            "i":'S100', #semantics unknown
            "j":'S100', #semantics unknown
            "k":'S100', #semantics unknown
            "l":'S100', #semantics unknown
            "o":'S100', #semantics unknown
            "p":'S100', #semantics unknown
            "q":'S100', #semantics unknown
            "t":'S100', #semantics unknown
        }

        self._colTypes = list(zip(self._colNames, [self._nonIntCols[col] if col in self._nonIntCols.keys() else np.int32 for col in self._colNames]))


    def download_data(self):
        datePattern = re.compile(r'(\w+)\ (\d{4})') #pattern for parsing date

        #create the target directory if it doesn't exist
        if not os.path.isdir(self._folder):
            os.mkdir(self._folder)

        with requests.Session() as s:
            #set headers
            s.headers.update({'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64)'})
            #get html from url
            r = s.get(self._url)
            r.raise_for_status()

            #parse html
            soup = BeautifulSoup(r.text, 'html.parser')
            if soup:
                links = soup.find_all("a", string="ZIP") #find all links with zip text

                #find latest month in each year(redundant data in earlier months)
                yearLatestMonthDict = dict() #contains the latest month (and the soup object of it's zip href) in a given year for which we have any data
                for link in links:
                    #get a month+year
                    monthYearString = link.parent.previous_sibling.string
                    m = re.match(r'(\w+) (\d+)', monthYearString)
                    if m:
                        month = self._monthDict[m.group(1).lower()]
                        year = m.group(2)
                    else:
                        raise UnexpectedDataFormatException("Wrong month year format")

                    #check if this is a newer month
                    if year in yearLatestMonthDict.keys():
                        if int(month) > int(yearLatestMonthDict[year][0]):
                            yearLatestMonthDict[year] = (month, link)
                    else:
                        yearLatestMonthDict[year] = (month, link)

                for year, (month, link) in yearLatestMonthDict.items():
                    #download data
                    href = link.get('href')
                    r = s.get(self._url+link.get('href'))#get archive
                    r.raise_for_status()

                    #open archive
                    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
                        for filename in zf.namelist():
                            if filename in self._file2regionDict.keys(): #check if file we are interested in
                                with zf.open(filename) as f: #open file
                                    #construct new filename
                                    newFileName = f"data/data_{self._file2regionDict[filename]}.csv"
                                    #save the file
                                    open(newFileName, "ab").write(f.read())
            else:
                raise UnexpectedDataFormatException("Wrong html format, Beautiful Soup failed.")

    def parse_region_data(self, region:str) -> (list, list):
        filename = f"{self._folder}/data_{region}.csv"
        if not os.path.isfile(filename):
            self.download_data()

        descriptiveColnames = [
            "id",
            "roadType",
            "roadNumber",
            "date",
            "weekday",
            "time",
            "accidentType",
            "collisionDirection",
            "barrierType",
            "isLethal",
            "offender",
            "accidentCause",
            "deaths",
            "seriousInjuries",
            "lightInjuries",
            "materialCost",
            "roadSurfaceType",
            "roadSurfaceState",
            "roadState",
            "weather",
            "light",
        ]

        def timeConverter(x):
            m = re.search(r'(\d{2})(\d{2})', str(x)).groups()
            return datetime.time(int(m[0]) if m[0] != "25" else -1, int(m[1]) if m[1] != "60" else -1)

        converters = {
            "p1":lambda x : int(x.decode().strip('"')),
            "p2a":lambda x : datetime.date(*map(int, re.search(r'(\d{4})-(\d{2})-(\d{2})', str(x)).groups())),
            #"p2a":lambda x: np.datetime64(x.decode().strip('"')),
            "p2b":timeConverter,
            "d":lambda x : float(x.decode().replace(",",".").strip('"')),
            "e":lambda x : float(x.decode().replace(",",".").strip('"')),
            "f":lambda x : float(x.decode().replace(",",".").strip('"')),
            "g":lambda x : float(x.decode().replace(",",".").strip('"')),
        }

        with open(filename, encoding="latin1") as fin:
            arr = np.genfromtxt(fin, names=self._colNames, dtype=self._colTypes, delimiter=";", converters=converters, invalid_raise=False)

        shape = arr.shape

        return (["region"]+self._colNames, [np.full(shape[0], region, dtype="S3")]+[arr[name] for name in self._colNames])

    def get_list(self, regions=None):
        #if regions==None, we consider all regions
        regions = regions or list(self._region2fileDict.keys())

        result = [np.empty(0, dtype="S3")] + [np.empty(0, dtype=colType[1]) for colType in self._colTypes] #don't forget region collumn

        for region in regions:
            cacheFilename = self._cache_filename.format(region)

            #get region data
            if self._regionCache[region]: #get from memory
                data = self._regionCache[region]
            elif os.path.isfile(cacheFilename): #load from cache file
                with gzip.open(cacheFilename) as fin:
                    self._regionCache[region] = pickle.load(fin)
                    data = self._regionCache[region]
            else: #read from data file
                with gzip.open(cacheFilename, "w") as fout:
                    print(f"Parsing region {region}")
                    self._regionCache[region] = self.parse_region_data(region)
                    pickle.dump(self._regionCache[region], fout)
                    data = self._regionCache[region]

            for i, col in enumerate(data[1]):
                result[i] = np.concatenate((result[i], col))

        return ["region"] + self._colNames, result

def main():
    downloader = DataDownloader()
    regions = ["HKK", "JHC", "JHM"]
    names, data = downloader.get_list(regions)

    print(f"Accident data for {regions[0]}, {regions[1]}, and {regions[2]}.")
    print("------------------")
    print("Collumn headers:")
    print(names)
    print("------------------")
    print(f"Row Count:{data[0].shape[0]}")

if __name__ == "__main__":
    main()