build/satellites.bdb: satellites/data/satellites.utf8.csv build
	rm -f $@
	(echo ".csv satellites $<" && \
	 echo "CREATE DEFAULT GENERATOR satellites_cc FOR satellites \
	       USING crosscat (GUESS(*), name IGNORE);" && \
	 echo "INITIALIZE 16 MODELS FOR satellites;" && \
	 echo "ANALYZE satellites FOR 4 MINUTES WAIT;") \
	| bayeslite $@ --seed 0

build:
	mkdir -p build
