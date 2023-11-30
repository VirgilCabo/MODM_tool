library(ggplot2)
library(ggridges)
library(tidyr)

data <- read.csv("C://Users//Virgi//OneDrive//Bureau//MODM_tool_project//Tool//Data//results//PROMETHEE//PROMETHEE_main_optimal_pareto_points2.csv_20231122_163259//sensitivity_net_flows.csv")

data <- data[,-1]

data_long <- pivot_longer(data, cols = everything(),
                          names_to = "alternative",
                          values_to = "score")

data_long$alternative <- factor(sub("X",
                                    "",
                                    as.character(data_long$alternative)))

head(data)

data_long$alternative_numeric <- as.numeric(data_long$alternative)

data_long$alternative <- factor(data_long$alternative,
                                levels = unique(data_long$alternative))

p <- ggplot(data_long, aes(x = score, y = alternative, fill = after_stat(x))) +
  geom_density_ridges_gradient(scale = 10, size = 0.3, rel_min_height = 0.01) +
  scale_fill_viridis_c(name = "Net Flow", alpha = 0.8) +
  labs(title = "Net Outranking Flow Distributions for each alternative",
       x = "Net Outranking Flow",
       y = "Alternative") +
  theme_ridges() +
  theme(
    plot.background = element_rect(fill = "white", colour = "white"),
    panel.background = element_rect(fill = "white", colour = "white")
  )

print(p)

output_directory <- "C://Users//Virgi//OneDrive//Bureau//MODM_tool_project//Tool//Data//results//ridgeplots_R//PROMETHEE//"
output_filename <- "normalization=minmax_weights=5,5,5_ranges=37,37,37_numsets=10000.png"
output_filepath <- paste0(output_directory, output_filename)
ggsave(output_filepath, plot = p, width = 8, height = 5, dpi = 500)

