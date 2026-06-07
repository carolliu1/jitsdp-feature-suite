
args <- commandArgs(trailingOnly = TRUE)



if (length(args) != 4) {

  stop("Usage: Rscript run_scottknottesd.R <input_csv> <output_csv> <version> <alpha>")

}



input_csv <- args[[1]]

output_csv <- args[[2]]

sk_version <- args[[3]]

alpha <- as.numeric(args[[4]])



if (!requireNamespace("ScottKnottESD", quietly = TRUE)) {

  stop("The R package ScottKnottESD is not installed.")

}



data <- read.csv(input_csv, check.names = FALSE)



sk <- ScottKnottESD::sk_esd(data, version = sk_version, alpha = alpha)

groups <- sk$groups



if (is.data.frame(groups)) {

  rank_column <- NULL

  for (candidate in c("groups", "rank", "ranks", "Group", "Rank")) {

    if (candidate %in% names(groups)) {

      rank_column <- candidate

      break

    }

  }

  if (is.null(rank_column)) {

    numeric_columns <- names(groups)[vapply(groups, is.numeric, logical(1))]

    if (length(numeric_columns) == 0) {

      stop("Cannot find a numeric ranking column in sk$groups.")

    }

    rank_column <- numeric_columns[[1]]

  }

  concepts <- rownames(groups)

  raw_groups <- as.integer(groups[[rank_column]])

} else {

  concepts <- names(groups)

  raw_groups <- as.integer(groups)

}



if (is.null(concepts) || any(is.na(concepts)) || any(concepts == "")) {

  stop("ScottKnottESD returned unnamed groups; cannot map ranks back to features.")

}



result <- data.frame(

  concept = concepts,

  raw_group = raw_groups,

  rank = raw_groups,

  stringsAsFactors = FALSE

)



result <- result[order(result$rank, result$concept), ]

write.csv(result, output_csv, row.names = FALSE)

