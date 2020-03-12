#!/usr/bin/Rscript

# friedman.post.hoc.simple(t(amg2$mean))$ranks
# friedman.post.hoc.simple(t(r$succRatio[1:4,]),minFlag=F)
# friedman.post.hoc.simple(t(r$mean[1:4,]))

#" 
#Useful commands:
#source("friedman.r"); r <- friedman.post.hoc.meth.bench("intsp.csv", minFlag=FALSE)
#source("friedman.r"); r <- friedman.post.hoc.meth.bench("strp.csv", minFlag=FALSE)
#sort(r$ranks)
#r$p.value
#r$cmp.matrix
#r$cmp.p.values
#r$cmp.p.values
#r$p.value
#r$cmp.matrix
#"





friedman.post.hoc.meth.bench <- function(fname, minFlag=TRUE)
{
# KK
# reads a csv file of the form: methods x benchmarks
# transforms a table into format expected by friedman.post and calls friedman.post
# set minFlag to true if the variable is minimized, otherwise to false

    options(width=10000) # prevent the wrapping of output
    #options(width=300)

    options(digits=2)
	  b <- read.csv(file=fname, header=TRUE, sep=";", row.names=1, check.names=FALSE)
    b <- b[rownames(b) != "All", ]
    b <- t(b)
	print(summary(b))
	r <- friedman.post.hoc.simple(b,minFlag)
    print(sort(r$ranks))
    r #sort(r.ranks)
}

## Does Friedman from dataframe
friedman.post.hoc.simple <- function(df, minFlag=T,...)  {
	res <- data.frame(var=rep(NA, 0), alg=rep("", 0),  
		block=rep("", 0), stringsAsFactors=FALSE, check.names=FALSE)
    i = 1
   print(df)
	for(r in 1:nrow(df))
		for(c in 1:ncol(df))
		{
			res[i,1] = df[r,c]
			res[i,2] = row.names(df)[r]
			res[i,3] = colnames(df)[c]
			i=i+1			
		}
	friedman.post.hoc (var ~ alg | block, data=res,minimize=minFlag, ...)
}

friedman.post.hoc <- function(formu, data, minimize=TRUE, p.signif=0.05, post.signif=p.signif, ctrl=NULL, method=NULL)
{
	#                               
	# formu is a formula of the shape: 	Y ~ X | block   i.e.:  Value ~ Algorithm | Problemf
	# data is a long data.frame with three columns:	[[ Y (numeric), X (factor), block (factor) ]]

	# get the names out of the formula
	formu.names <- all.vars(formu)
	Y.name <- formu.names[1]
	X.name <- formu.names[2]
	block.name <- formu.names[3]
	
	X <- factor(data[,X.name ])
	Y <- data[,Y.name]
	block <- factor(data[,block.name ])
			
	n <- nlevels(block)
	k <- nlevels(X)
	
	if(sum(is.na(Y)) > 0) stop("Function stopped: This function doesn't handle NA's. In case of NA in Y in one of the blocks, then that entire block should be removed.")
	if(k == 2) { warning(paste("'",X.name,"'", "has only two levels. Consider using paired wilcox.test instead of friedman test"))}

	# convert data.frame to matrix
	values <- matrix(nrow=n, ncol=k, dimnames=list(levels(block), levels(X)))
	values[cbind(block, X)] <- Y
	
	ranks.in.blocks <- t(apply(if (minimize) values else -values, 1, rank))
	rmean <- mean(ranks.in.blocks)
	ranks <- apply(ranks.in.blocks, 2, mean)
	
	Ff <- n*(k-1)*(n*sum((ranks-rmean)^2))/(sum((ranks.in.blocks-rmean)^2))
	p.value <- pchisq(Ff, k-1, lower.tail=F)
	
	if (p.value < p.signif)
	{
		if (is.null(ctrl)) {
			# NxN
			#stop("NxN posthoc not implemented yet")
			method <- match.arg(method, c('shaffer', p.adjust.methods))
			
			#z <- outer(1:k,1:k, function(a,b) abs(ranks[a]-ranks[b]) / sqrt((k*(k+1))/(6*n)))
			rowidx <- unlist(lapply(2:k, function(x) x:k))
			colidx <- rep(1:(k-1), (k-1):1)
			z <- abs(ranks[rowidx] - ranks[colidx]) / sqrt((k*(k+1))/(6*n))			
			p <- 2*pnorm(-z)
		
			cmp.p.values <- switch(method,
					shaffer = {
						ShafferH <- function(k) {
							if (k <= 1) return(0)
							S <- vector("list", k)
							S[[1]] <- 0
							
							for (i in 2:k) {
								res <- choose(i, 2)
								for (j in (i-1):1) res <- union(res, choose(j, 2) + S[[i-j]])
								S[[i]] <- res
							}
							return(sort(S[[k]]))
						}

												
						i <- seq_len(length(p))
						o <- order(p)
						ro <- order(o)
						m <- k*(k-1)/2
						
						val <- matrix(nrow=k,ncol=k)
						#val[lower.tri(val)] <- p.adjust(p) # Homel
						#val[lower.tri(val)] <- pmin(1, cummax( (m - i + 1L) * p[o] ))[ro] # Homel
						
						H <- ShafferH(k)
						val[lower.tri(val)] <- pmin(1, cummax( unlist(lapply(0:(m-1), function(x) max(H[H<=(m-x)]))) * p[o] ))[ro] # Shaffer's static S1
						val
					},
					{
						val <- matrix(nrow=k,ncol=k)
						val[lower.tri(val)] <- p.adjust(p)
						val
					}
				)
			
			cmp.p.values[upper.tri(cmp.p.values)] <- t(cmp.p.values)[upper.tri(cmp.p.values)]
			cmp.matrix <- matrix(ifelse(cmp.p.values < p.signif, outer(1:k,1:k, function(a,b) sign(ranks[b]-ranks[a])), 0), nrow=k,ncol=k)
			dimnames(cmp.p.values) <- dimnames(cmp.matrix) <- list(levels(X), levels(X))			
		} else {
			# 1xN
			if (is.character(ctrl))
				ctrl <- match(ctrl, names(ranks))
			z <- abs(ranks[ctrl]-ranks[-ctrl]) / sqrt((k*(k+1))/(6*n))
			p <- 2*pnorm(-z)
			
			if (is.null(method)) method <- 'holm';
			cmp.p.values <- matrix(p.adjust(p, method=method), nrow=1, dimnames=list(names(ranks)[ctrl], names(p)))
			cmp.matrix <- ifelse(cmp.p.values < p.signif, sign(ranks[-ctrl] - ranks[ctrl]), 0)
			
		}
		
		return(list(statistic=Ff, parameter=k-1, p.value=p.value, ranks=ranks, cmp.p.values=cmp.p.values, cmp.matrix=cmp.matrix, cmp.method=method))
	} else return(list(statistic=Ff, parameter=k-1, p.value=p.value, ranks=ranks))
}
# friedman.post.hoc(successes ~ method | problem, sstab, minimize=FALSE, ctrl=c('X_A0.2','X_A0.3'))
# 
# friedman.post.hoc(successes ~ method | problem, sstab, minimize=FALSE)
# 
# friedman.post.hoc(successes ~ method | problem, sstab, minimize=FALSE, ctrl='X_A0.2')
# 
# friedman.post.hoc(successes ~ method | problem, sstab, minimize=FALSE, ctrl=c('X_A0.2','X_A0.3'))
# 
# friedman.post.hoc(successes ~ method | problem, sstab)


#################################

args <- commandArgs(trailingOnly = TRUE)
friedman.post.hoc.meth.bench(args[1], minFlag=args[2])

#friedman.post.hoc.meth.bench("tmp.csv", minFlag=FALSE)

#r <- friedman.post.hoc.meth.bench("table-success-small.csv", minFlag=FALSE)
#sort(r$ranks)
#r$p.value
#r$cmp.matrix
#r$cmp.p.values
#source("friedman.r"); r <- friedman.post.hoc.meth.bench("strp.csv", minFlag=FALSE)

