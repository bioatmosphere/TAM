#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Plot the locations of the five sites across the gradient
#
# By Bin Wang
# 09/16/2021
# 09/28/2021: SPRUCE point added
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Helpful guide:https://www.nceas.ucsb.edu/~frazier/RSpatialGuides/ggmap/ggmapCheatsheet.pdf

library(ggmap)

#downloading the raster:
MapImage <- get_map(location = c(left=-95.00, bottom=25.0, right=-65.0, top=50.0),
                    zoom=7, color = "color",source="google",maptype = "terrain", force=TRUE)

#site locations: MOFLUX, MORTON, Howland, SPRUCE
Longitude = c(-92.12,-88.07,-68.74,-93.48)
Latitude  = c(38.44, 41.82, 45.20, 47.50)
data = data.frame(Longitude, Latitude)


#Plotting the map
ggmap(MapImage) +
  geom_point(aes(x = Longitude, y = Latitude), data = data, alpha = .5, color="blue", size = 3) +
  annotate('text', x=data[1,1], y=data[1,2]-1, label = 'MOFLUX',  color = "black", size = 6) +
  annotate('text', x=data[2,1], y=data[2,2]-1, label = 'MORTON',  color = "black", size = 6) +
  annotate('text', x=data[3,1], y=data[3,2]-1, label = 'Howland', color = "black", size = 6) +
  annotate('text', x=data[4,1]+1, y=data[4,2]+1, label = 'SPRUCE',  color = "black", size = 6) +
  labs(x='Longitude',y='Latitude')+
  theme_bw()+
  theme(axis.text = element_text(size=10,face='plain'),axis.title = element_text(size=12,face='bold'))
  
