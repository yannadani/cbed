dir.create(Sys.getenv("R_LIBS_USER"), recursive = TRUE)  # create personal library
.libPaths(Sys.getenv("R_LIBS_USER"))  # add to the path

if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager", repos='http://cran.us.r-project.org')
    BiocManager::install()
    BiocManager::install(c("graph", "RBGL", "Rgraphviz"))
if (!require("rlang"))
    install.packages('rlang',repos='http://cran.us.r-project.org')
if (!require("fastICA"))
    install.packages("https://cran.r-project.org/src/contrib/Archive/fastICA/fastICA_1.1-16.tar.gz")
if (!require("pcalg"))
    install.packages('pcalg', repos='http://cran.us.r-project.org')

if (!require("SID"))
    install.packages('https://cran.r-project.org/src/contrib/Archive/SID/SID_1.0.tar.gz')
