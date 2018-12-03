dat <- read.csv("directed_alliance_data.csv", header=TRUE)
countrys <- c("USA", "CAN", "CUB", "SLV", "IRQ")
year <- c("1950", "1960", "1970", "1980", "1990", "2000")
dt=xtabs(defense ~ country_a + country_b + year, dat)

gr<-graph_from_adjacency_matrix(dt[countrys, countrys, "1950"], mode=c("undirected"))
plot(gr)
