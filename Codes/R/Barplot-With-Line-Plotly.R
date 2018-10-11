library(plotly)

# Subsetting the data for the month of May
airquality_sept <- airquality[which(airquality$Month == 5),]
airquality_sept$Date <- as.Date(paste(airquality_sept$Month, airquality_sept$Day, sep = "."), format = "%m.%d")

# Using plot_ly() to generate the chart
p <- plot_ly(airquality_sept) %>%

  # Adding the bars by using add_trace() 
  add_trace(x = ~Date, y = ~Wind, type = 'bar', name = 'Wind',
            marker = list(color = '#e88686'),
            hoverinfo = "text",
            text = ~paste(Wind, ' mph')) %>%
  # Adding the line by using add_trace() 
  add_trace(x = ~Date, y = ~Temp, type = 'scatter', mode = 'lines', name = 'Temperature', yaxis = 'y2',
            line = list(color = '#183cbf'),
            hoverinfo = "text",
            text = ~paste(Temp, '°F')) %>%
  layout(title = 'Wind and Temperature Measurements for September - NY',
         yaxis = list(side = 'left', title = 'Wind', showgrid = TRUE),
         yaxis2 = list(side = 'right', overlaying = "y", title = 'Temperature in F', showgrid = TRUE))

# Visualizing the plot
p
