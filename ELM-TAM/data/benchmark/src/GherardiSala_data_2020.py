"""Extract data from Gherardi and Sala (2020) PNAS for the benchmark.

This work compiled a list of studies that measured belowground net carbon fixation (BNF) in terrestrial ecosystems.

Reference:
    Gherardi LA and Sala OE (2020) Global patterns and climatic controls of belowground net carbon fixation.
PNAS 117(33), 20038-20043. DOI: 10.1073/pnas.2006715117
"""

import csv

# Raw text data for Supplementary Table 5, extracted from the PDF with Gemini.
# This data is spread across multiple pages and needs careful parsing.
# Note: Manual inspection and minor corrections might be needed due to OCR imperfections or complex table structures.

table_data_raw = """
Site #,Latitude,Longitude,Country,Location,Data Source,Disturbance/Management,Dominant species,Cover Type,MAT,MAP,Reference
1,49.40,-110.69,Canada,Alberta,Publication,Protected from grazing,"Stipa comata, Boutelkoua gracilis",grasslands,4.5,343,(53)
2,44.22,-122.24,"United States",Oregon,Ecotrends,Old growth,"Tsuga heterophylla, Rhododendron macrophyllum, Berberis nervosa","evergreen needleleaf forest",9.3,2500,(54)
3,44.23,-122.15,"United States",Oregon,Ecotrends,Old growth,"Tsuga heterophylla, Acer circinatum, Polystichum munitum","evergreen needleleaf forest",9.3,2500,(54)
4,66.63,-149.60,"United States",Alaska,Ecotrends,"No evidence of disturbance","Eriophorum vaginatum","woody savannas",-12.5,680,(55)
5,-17.30,145.61,Australia,Atherton,ORNL,"No evidence of disturbance","Dendrocnide photinophylla, Rockinghamia angustifolia, Toona australis, Aleurites moluccana","evergreen broadleaf forest",19.8,1560,(56)
6,42.38,71.23,"United States","Massachusetts",Publication,"Old Field","Dactylis glomerata, Elymus repens. Phleum pratense, Poa spp",grasslands,9.3,1190,(57)
7,32.53,-106.79,"United States","New Mexico",LTER,Undisturbed,"Bouteloua eriopoda","open shrublands",14.7,247,(58)
8,35.68,62.00,Turkmenistan,Badkhyz,ORNL,NA,"Poa bulbosa, Carex pachystylis, Onobrychis pulchella","open shrublands",12.8,292,(59)
9,-29.10,26.95,"South Africa",Bloemfontein,Publication,"History of grazing","Themeda triandra, Cymbopogon plurinodis",grasslands,25,560,(60)
10,64.72,-148.15,"United States",Alaska,LTER,"Old growth","Populus balsamifera-Picea glauca","evergreen needleleaf forest",-2.3,194,(61)
11,64.68,-148.24,"United States",Alaska,LTER,"Old growth","Betula papyrifera","evergreen needleleaf forest",-2.3,175,(62)
12,64.70,-148.36,"United States",Alaska,LTER,"Old growth","Picea glauca Betula papyrifera","evergreen needleleaf forest",-2.3,201,(63)
13,9.15,-79.85,Panama,"Barro Colorado",ORNL,None,"Tachigali versicolor","evergreen broadleaf forest",26.3,2626,(64)
14,45.40,-93.20,"United States",Minnesota,LTER,"Disturbed by thoroughly disking the area prior to establishment of the experiment.","Agrostis scabra. Oenothera biennis. Agropyron repen",grasslands,6.7,841,(65)
15,45.40,-93.20,"United States",Minnesota,LTER,"Disturbed by thoroughly disking the area prior to establishment of the experiment.","Agrostis scabra, Oenothera biennis, Agropyron repen",grasslands,6.7,841,(65)
16,41.18,104.83,"United States",Wyoming,Publication,NA,"Bouteloua gracilis",grasslands,7.6,384,(66)
17,50.20,-115.50,Canada,"British Columbia",ORNL,"Natural regeneration after wildfire","Pinus contorta","evergreen needleleaf forest",1.4,630,(67)
18,50.98,-115.12,Canada,"British Columbia",ORNL,"Natural regeneration after wildfire","Pinus contorta","evergreen needleleaf forest",1.4,630,(67)
19,18.30,-65.75,"United States","US Virgin Islands",ORNL,"Undisturbed since 1900","Tabebuia heterophylla","evergreen broadleaf forest",26.6,1130,(68)
20,8.65,-78.12,Panama,Darien,ORNL,"Not cultivated for previous 400 years","Cavanillesia platanifolia","evergreen broadleaf forest",5.6,2000,(69)
21,44.00,-112.00,"United States",Idaho,Publication,"Livestock excluded","Agropyron spicatum, Koeleria cristata, Stipa comata",grasslands,6,254,(70)
22,42.39,116.56,China,"Inner Mongolia",Publication,Fenced,"Stipa krylovii, Agropyron cristatum, Artemisia frigida, Potentilla acaulis",grasslands,3.3,382,(71)
23,49.33,46.78,Kazakhstan,Dhzanybek,ORNL,NA,"Agropyron desertorum",grasslands,5,283,(72)
24,48.88,119.80,China,"Ewenke Qi",Publication,NA,"Stipa baicalensis",grasslands,-1.9,330,(73)
25,64.12,19.45,Sweden,Uppsala,ORNL,"Planted 4-yr old seedlings in 1963 after clear felling (Contol treatments)","Picea abies",grasslands,1.8,600,(74)
26,4.02,114.80,Malaysia,"Gonung Mulu",ORNL,NA,"Dipterocarpaceae/Euphorbiaceae","evergreen broadleaf forest",27,5090,(75)
27,4.02,114.85,Malaysia,"Gonung Mulu",ORNL,NA,"Dipterocarpaceae/Euphorbiaceae","evergreen broadleaf forest",27,5110,(75)
28,4.14,114.86,Malaysia,"Gonung Mulu",ORNL,NA,"Dipterocarpaceae/Euphorbiaceae","evergreen broadleaf forest",27,5700,(75)
29,4.12,114.88,Malaysia,"Gonung Mulu",ORNL,NA,"Dipterocarpaceae/Euphorbiaceae","evergreen broadleaf forest",27,5700,(75)
30,35.69,-83.42,"United States",Tennessee,ORNL,"Sheltered from major disturbance","Halesia-Acer-Tsuga-Aesculus","mixed forests",13.5,1400,(76)
31,35.69,-83.42,"United States",Tennessee,ORNL,"Sheltered from major disturbance",Liriodendron,"mixed forests",13.5,1400,(76)
32,35.68,-83.39,"United States",Tennessee,ORNL,"Sheltered from major disturbance","Halesia-Acer-Aesculus","mixed forests",13.5,1400,(76)
33,35.68,-83.46,"United States",Tennessee,ORNL,"Sheltered from major disturbance","Tsuga-Halesia-Fagus-Acer","mixed forest",13.5,1400,(76)
34,37.48,101.20,China,Qinghai,Publication,NA,"Kobresia homilis",grasslands,0.6,589,(77)
35,39.80,-99.30,"United States",Kansas,Publication,NA,"Bouteloua gracilis, Buchloe dactyloides",grasslands,12,606,(78)
36,37.75,101.12,China,Qinghai,Publication,NA,"Kobresia humilis",grasslands,-4.7,560,(77)
37,43.94,-71.75,"United States","New Hampshire",Ecotrends,"Reference watershed","Acer saccharum, Fagus grandifolia, Betula alleghaniensis, Picea rubens","mixed forests",4.5,1150,(54)
38,42.53,-72.19,"United States","Massachusetts",Ecotrends,NA,"Quercus rubra-Acer rubrum-Betula lenta-Pinus strobus -Tsuga canadensis","mixed forests",7.3,1195,(54)
39,49.75,15.98,"Czech Republic",Hlinsko,Publication,NA,"Cirsium palustre, Deschampsia ceaspitosa, Agrostis capilaris","mixed forests",6.3,762,(79)
40,32.59,-106.84,"United States","New Mexico",LTER,Undisturbed,"Bouteloua eriopoda","open shrublands",14.7,247,(58)
41,60.84,16.50,Sweden,Uppsala,ORNL,"Regrowth forest harvested in 1957 and thinned in 1962","Pinus sylvestris","evergreen needleleaf forest",3.5,730,(80)
42,18.08,-76.65,Jamaica,"John Crow Ridge",ORNL,"Appeared completely undisturbed","Chaetocarpus globosus-Clusia cf. havetioides- Lyonia cf. octandra association","evergreen broadleaf forest",17.5,2230,(81)
43,42.40,-85.40,"United States",Michigan,LTER,Undisturbed,"Andropogon gerardii","grasslands",10.1,791,(82)
44,6.15,-0.92,Ghana,Kade,ORNL,"Not cultivated for previous 40-50 years","Diospyros spp","evergreen broadleaf forest",26.5,1650,(83)
45,39.08,-96.56,"United States",Kansas,LTER,"Annually burned","Andropogon gerardii, Sorghastrum nutans, Panicum virgatum, and Schizachyrium scoparium",grasslands,12.5,849,(84)
46,39.07,-96.58,"United States",Kansas,LTER,"Burned every 20 years","Andropogon gerardii, Sorghastrum nutans, Panicum virgatum, and Schizachyrium scoparium",grasslands,12.5,849,(84)
47,51.67,36.50,Russia,Kursk,ORNL,"Preserved in natural state","Bromus riparius, Stipa pennata, Poa angustifolia. Agropyron intermedium, Filipendula hexapetala, Fragaria viridis",grasslands,6.1,583,(85)
48,48.82,15.99,"Czech Republic",Znojmo,Publication,NA,"Festuca ovina, Avenella flexuosa, Anthoxantum odoratum","mixed forests",9,587,(79)
49,18.32,-65.75,"Puerto Rico",Luquillo,ORNL,"Undisturbed since 1930","Dacryodes excelsa","evergreen broadleaf forest",23,3500,(86)
50,18.32,-65.82,"Puerto Rico",Luquillo,ORNL,"Not cultivated for previous 40-50 years","Tabebuia rigida","evergreen broadleaf forest",23,3810,(86)
51,46.79,-100.92,"United States","North Dakota",Publication,"Undisturbed since 1915","Bouteloua gracilis, Stipa comata, Agropyron smithii",grasslands,5.8,415,(87)
52,6.39,-73.56,Colombia,"Magdalena Valley",ORNL,NA,"Jessenia polycarpa","evergreen broadleaf forest",27.8,3000,(88)
53,46.40,-105.95,"United States",Kansas,Publication,"Cattle grazing excluded","Schizachyrium scoparium",grasslands,12.1,583,(89)
54,20.81,-156.26,"United States",Hawaii,ORNL,NA,"Metrosideros polymorpha","evergreen broadleaf forest",23,2200,(90)
55,-3.00,-59.70,Brazil,Manaus,ORNL,"mature forest",NA,"evergreen broadleaf forest",26.7,1771,(91)
56,49.55,20.15,"Czech Republic","Bily Kriz",Publication,NA,"Nardus stricta, Avenella flexuosa, Festuca rubra, and Agrostis capillaris","mixed forests",6.5,947,(79)
57,-6.00,145.18,"Papua New Guinea",Marafunga,ORNL,"Mostly untouched before 1962","Dacrycarpus cinctus","evergreen broadleaf forest",13,4010,(92)
58,46.35,-83.38,Canada,Mississagi,ORNL,"fire every 100? years","Pinus banksiana","mixed forests",4.1,798,(93)
59,43.43,116.07,China,"Inner Mongolia",Publication,Ungrazed,"Leymus chinensis",grasslands,0.9,297,(73)
60,46.40,-105.95,"United States",Montana,Publication,"Ungrazed since 1999.","Hesperostipa comata, Pascopyrum smithii, Carex filifolia",grasslands,7.8,342,(94)
61,-1.33,36.83,Kenya,Nairobi,ORNL,"Cattle grazing ceased in 1946","Pennisetum menzianum, Themeda triandra",grasslands,19.1,677,(95)
62,34.98,-97.52,"United States",Oklahoma,Publication,"Abandoned from field cropping in 1970 with light grazing until 5 years before the study","Ambrosia trifida, Solanum carolinense. Euphorbia dentata",grasslands,16.3,914,(96)
63,45.05,-123.96,"United States",Oregon,ORNL,"Old growth","Tsuga heterophylla-Alnus rubra","evergreen needleleaf forest",10.1,2510,(97)
64,44.29,-121.33,"United States",Oregon,ORNL,None,"Juniperus occidentalis","closed shrublands",9.1,220,(97)
65,44.42,-121.44,"United States",Oregon,ORNL,Control,"Pinus ponderosa","closed shrublands",7.4,540,(97)
66,44.42,-121.84,"United States",Oregon,ORNL,None,"Tsuga mertensiana","evergreen needleleaf forest",6,1810,(97)
67,44.60,-123.27,"United States",Oregon,ORNL,None,"Pseudotsuga menziesii","evergreen needleleaf forest",11.2,980,(97)
68,46.77,-99.47,"United States","North Dakota",Publication,"Plowed before 1950, moderately grazed until study began","Bromus inermis, Symphoricarpos occidentalis. Oligoneuron rigidum",grasslands,5.1,454,(8)
69,46.77,-99.47,"United States","North Dakota",Publication,"Plowed before 1950, moderately grazed until study began","Poa pratensis, Nassella viridula, Carexinops, Pascopirum smithii",grasslands,5.1,454,(8)
70,-45.68,-70.27,Argentina,"Santa Cruz",Publication,Ungrazed,"Stipa speciosa, Mulinum Spinosum","open shrublands",8.5,165,(8)
71,40.35,-123.00,"United States",California,ORNL,None,"Sequoia sempervirens","evergreen needleleaf forest",12.6,1230,(8)
72,1.93,-66.96,Venezuela,"San Carlos de Rio Negro",ORNL,"Apparently undisturbed","Aspidosperma spp.","evergreen broadleaf forest",26.1,3565,(8)
73,1.90,-67.00,Venezuela,"San Carlos de Rio Negro",ORNL,"Apparently undisturbed","Aspidosperma spp.","evergreen broadleaf forest",26.1,3565,(8)
74,8.60,-71.18,Venezuela,"San Eusebio",ORNL,"Primary or secondary forest, at least 25 years old","Podocarpus rospigliosii","evergreen broadleaf forest",18.9,1752,(8)
75,34.34,-106.97,"United States","New Mexico",Ecotrends,Undisturbed,"Bouteloua gracilis, bouteloua eriopoda","open shrublands",13.4,251,(8)
76,34.34,-106.97,"United States","New Mexico",Ecotrends,Undisturbed,"Larrea tridentata","open shrublands",13.4,251,(8)
77,34.34,-106.97,"United States","New Mexico",Ecotrends,Undisturbed,"Larrea tridentata","open shrublands",13.4,251,(8)
78,39.26,-121.32,"United States",California,Publication,NA,"Avena spp., Bromus spp., Lolium spp., Hordeum spp.","woody savannas",16,750,(8)
79,40.84,-104.76,"United States",Colorado,Publication,Ungrazed,"Bouteloua gracilis",grasslands,8.6,321,(94)
80,40.84,-104.76,"United States",Colorado,Publication,Ungrazed,"Bouteloua gracilis",grasslands,8.6,321,(89)
81,40.84,-104.76,"United States",Colorado,Ecotrends,Ungrazed,"Bouteloua gracilis",grasslands,8.6,321,(8)
82,40.84,-104.76,"United States",Colorado,Publication,Ungrazed,"Bouteloua gracilis",grasslands,8.6,321,(9)
83,40.84,-104.76,"United States",Colorado,Ecotrends,Ungrazed,"Bouteloua gracilis",grasslands,8.6,321,(9)
84,37.09,-119.73,"United States",California,Publication,"Protected from grazing","Erodium cicutarium, Hordeum gussonianum, Poa scabrella, Bromus rigidus, Medicago hispida, Hypochoeris glabra, Bromus mollis",savannas,15.8,495,(9)
85,10.43,-83.98,"Costa Rica","La Selva",ORNL,"Light shifting cultivation for previous 3000 years","Pentaclethra macroloba","evergreen broadleaf forest",25.9,3962,(9)
86,48.07,-92.04,"United States",Minnesota,ORNL,NA,"Populus tremuloides","mixed forests",2.7,634,(9)
87,48.07,-92.04,"United States",Minnesota,ORNL,NA,"Picea strobus","mixed forests",2.7,604,(9)
88,58.00,83.00,Russia,"Tomsk Region",ORNL,"25-122 year old stands","Pinus sylvestris","mixed forests",0.9,501,(9)
89,39.08,-96.56,"United States",Kansas,LTER,"Annually burned","Andropogon gerardii, Sorghastrum nutans, Panicum virgatum, and Schizachyrium scoparium",grasslands,12.5,849,(94)
90,50.65,11.34,Germany,"Thüringer Schiefergebirge",Publication,Ungrazed,NA,"mixed forests",7.1,560,(9)
91,46.10,123.00,China,Tumugi,ORNL,"Fenced site","Filifolium sibiricum",grasslands,2.1,410,(9)
92,46.10,123.00,China,Tumugi,ORNL,"Fenced site","Stipa baicalensis",grasslands,2.1,410,(9)
93,46.10,123.00,China,Tumugi,ORNL,"Fenced site","Leymus chinense",grasslands,2.1,410,(9)
94,-24.90,28.35,"South Africa",Towoomba,ORNL,"all trees were removed","Themeda triandra, Cymbopogon pospischilli, Heteropogon contortus",savannas,18.7,629,(9)
95,43.64,116.70,China,"Xilin River Basin",Publication,"Previously grazed","Stipa grandis, Leymus chinensis, Cleistogenes squarrosa, Agropyron cristatum",grasslands,0.7,320,(9)
96,44.55,117.61,China,"Xiwu Qi",Publication,Fenced,"Stipa grandis",grasslands,1.1,329,(73)
97,43.72,116.63,China,Xilingol,ORNL,Fenced,"Leymus chinense",grasslands,-2,334,(10)
98,42.31,116.10,China,"Zhenglan Qi",Publication,Fenced,"Stipa krylovii",grasslands,1.7,364,(73)
99,55.879,-98.48,Canada,Manitoba,ForC,"established 1850","Picea glauca","boreal coniferous forest",-3,510,(119)
100,55.906,-98.525,Canada,Manitoba,ForC,"established 1930","Picea glauca","boreal coniferous forest",-3,510,(119)
101,-13.28,-71.6,Peru,Acjanaco,ForC,unmanaged,"Weinmannia crassifolia","tropical evergreen forest",6.8,760,(120)
102,-3.95,-73.43,Peru,Loreto,ForC,unmanaged,"Eschweilera, Guatteria, Inga","tropical rain forest",26.3,2762,(121)
103,-12.495,131.152,Australia,"Northern Territory",ForC,unmanaged,"Eucalyptus tetrodonta","tropical evergreen savanna",NA,1750,(122)
104,46.87,-87.9,"United States",Michigan,ForC,unmanaged,"Tsuga canadensis","temperate forest",NA,1102,(123)
105,62.85,30.88,Finland,Ilomantsi,Publication,NA,"Picea abies, Pinus sylvestris, Betula pubescens","boreal forest",NA,NA,(124)
106,53,103,Russia,Irkutsk,Publication,NA,"Pinus sylvestris","boreal forest",NA,NA,(124)
107,35.65,-84.28,"United States",Tennessee,ForC,unmanaged,"Quercus alba","broadleaf forest",14.31,1400,(125)
108,26.18,117.43,China,Fujian,ForC,unmanaged,"Castanopsis kawakamii","subtropical humid forest",19.1,1749,(126)
109,46.717,7.767,Switzerland,Beatenberg,Publication,"stand established in 1801","Picea abies","temperate evergreen forest",4.7,1454,(127)
110,5.1,-61,Venezuela,"Canaima National Park",ForC,NA,NA,"tropical forest",20.7,2000,(128)
111,45.83,-121.99,"United States",Washington,ForC,unmanaged,"Pseudotsuga menziesii and Tsuga heterophylla","Temperate Forest",8.7,2467,(129)
"""

# Split the raw data into lines and remove any leading/trailing whitespace
lines = [line.strip() for line in table_data_raw.strip().split('\n')]

# The first line is the header
header = lines[0].split(',')
# The rest are data rows
data_rows_text = lines[1:]

# Prepare data for CSV writing
# Each item in the list will be a row in the CSV
csv_output_data = [header]

for row_text in data_rows_text:
    # Use csv.reader to handle quoted fields correctly
    # We need to pass it as a list containing a single string
    reader = csv.reader([row_text])
    for parsed_row in reader:
        csv_output_data.append(parsed_row)

# Define the output CSV file name
csv_file_name = "../productivity/globe/supplementary_table_5.csv"

# Write the data to a CSV file
try:
    with open(csv_file_name, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(csv_output_data)
    print(f"Successfully wrote data to {csv_file_name}")
    # You can offer the file for download if this script is run in a web environment
    # For local execution, the file is saved in the current directory.
except IOError:
    print(f"Error: Could not write to file {csv_file_name}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

# For verification, print a few rows of the parsed data
print("\nFirst 5 rows of the parsed data (including header):")
for i in range(min(5, len(csv_output_data))):
    print(csv_output_data[i])

