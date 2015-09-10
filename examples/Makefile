build/satellites/satellites.html: satellites/satellites.bql | build
	# Set the working directory because the .bql script depends on it
	cd satellites && \
	(echo ".hook ../../bdbcontrib/contrib_util.py" && \
	 echo ".hook ../../bdbcontrib/contrib_plot.py" && \
	 echo ".hook ../../bdbcontrib/contrib_math.py" && \
	 echo ".readtohtml satellites.bql ../build/satellites/") \
	| PYTHONPATH=../../:$$PYTHONPATH bayeslite -m --seed 0 -j1

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
