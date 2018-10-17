# Packge for generating the plots
library(plotly)

# Using gapminder dataset
library(gapminder)


p <- gapminder %>% # Reads the data 
  plot_ly(
    x = ~lifeExp,  # xasix variable
    y = ~gdpPercap, # yaxis variable
    color = ~continent, # different colors for different countries
    frame = ~year, # This command results into animation
    text = ~country, # hovering text
    hoverinfo = "text", # on hoevr what should you get
    type = 'scatter' # plot type is scatter. 
  ) %>%
  layout(
    xaxis = list(
      type = "log" # taking log 
    )
  )